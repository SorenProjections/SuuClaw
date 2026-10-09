"""The assembled supervisor → verify graph, compiled and ready to run.

::

    START ──▶ planner ──▶ verifier ──┬─ "final" ──▶ final ──▶ END
                 ▲                   │
                 └──── "planner" ────┘

Stage 3 has no separate actor node: the planner is the supervisor, and the work
happens inside its ``call_search_agent`` / ``call_code_agent`` tool calls — each
delegation runs a full specialist loop *within* the planner's superstep.

Usage::

    from suuclaw.core import RuntimeState
    from suuclaw.graph import build_workflow, initial_state

    app = build_workflow()                     # no API key needed to build it
    runtime = RuntimeState.create("./sandbox")
    result = app.invoke(initial_state("add a --version flag", runtime, max_attempts=3))
    print(result["final_answer"])

Progress streams as it happens: the specialists' ReAct steps and the planner's
delegations are forwarded to LangGraph's custom stream, tagged with the node that
produced them (``planner``, ``search``, ``code``, ``verifier``)::

    for chunk in app.stream(state, stream_mode="custom"):
        print(chunk["node"], chunk["type"])

Front ends do not have to hold a graph at all —
:func:`~suuclaw.core.agent.stream_agent_events` compiles one, runs it, and
re-emits both stream modes as a single flat event sequence (``node_start``,
``node_output``, plus the live inner-loop events above) which is what the CLI
renders. It compiles :func:`build_complex_workflow`, so the CLI gets the context
monitor and compressor without asking for them.

Two things worth knowing before running this for real.

**Budget.** Which graph you run sets the floor. One pass around the simple graph
(:func:`build_workflow`) costs *two* supersteps (planner, verifier); the
context-aware one costs *four* (planner, monitor, verifier, monitor) and one
more per compression detour. Size ``recursion_limit`` for the worst case rather
than the average — the driver's own math does
(:data:`~suuclaw.core.agent.SUPERSTEPS_PER_ATTEMPT`), and a manual
``app.invoke`` should too, or the run dies with ``GraphRecursionError`` instead
of reaching ``final``::

    app.invoke(state, config={"recursion_limit": 60})

Note the budget asymmetry the topology hides: a superstep here can be long. One
planner visit may run several specialist loops back to back, each up to
``code_agent_loops`` / search-loops of its own.

**Persistence.** ``runtime`` is a live ``RuntimeState`` object, not data. An
in-memory checkpointer does not care, but a real one has to serialize it, and
LangGraph warns that the type is unregistered and will refuse it in a future
version. Either register the type when compiling, or keep ``runtime`` out of the
checkpointed state and hand it to the nodes through ``config["configurable"]``.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..core.agent import DEFAULT_RETRIES, DEFAULT_RETRY_BACKOFF
from ..core.state import RuntimeState
from .nodes import (
    DEFAULT_CODE_AGENT_LOOPS,
    DEFAULT_PLANNER_LOOPS,
    DEFAULT_VERIFIER_LOOPS,
    chat_responder_node,
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    final_node,
    intent_route_fn,
    intent_router_node,
    make_chat_responder_node,
    make_context_compressor_node,
    make_context_monitor_node,
    make_final_node,
    make_intent_router_node,
    make_planner_node,
    make_verifier_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from .state import SuuClawGraphState

__all__ = [
    "CONTEXT_COMPRESSOR_ROUTE_TARGETS",
    "CONTEXT_MONITOR_ROUTE_TARGETS",
    "INTENT_ROUTE_TARGETS",
    "ROUTE_TARGETS",
    "build_complex_state_graph",
    "build_complex_workflow",
    "build_entry_workflow",
    "build_state_graph",
    "build_workflow",
    "initial_state",
]

#: Where ``verifier_route`` may send the run. Spelled out rather than relying on
#: the route names happening to match the node names: the router returns a
#: *destination* ("finish" vs "retry"), and this is the one place that binds those
#: destinations to the nodes that implement them.
ROUTE_TARGETS: dict[str, str] = {"final": "final", "planner": "planner"}

#: Where ``context_monitor_route`` may send the run: finish, detour through
#: compression, or continue to the node upstream nominated in
#: ``context_next_node``.
CONTEXT_MONITOR_ROUTE_TARGETS: dict[str, str] = {
    "context_compressor": "context_compressor",
    "verifier": "verifier",
    "planner": "planner",
    "final": "final",
}

#: Where ``context_compressor_route`` may send the run: straight back to the
#: interrupted work. No self-loop — the *next* monitor visit re-measures.
CONTEXT_COMPRESSOR_ROUTE_TARGETS: dict[str, str] = {
    "verifier": "verifier",
    "planner": "planner",
    "final": "final",
}

#: Where ``intent_route_fn`` may send the run *inside the entry graph*. The
#: ``"planner"`` destination binds to ``END`` on purpose: the main workflow is
#: a separate graph the caller compiles and drives, so the entry graph's only
#: job is to decide whose graph owns the run and get out of the way.
INTENT_ROUTE_TARGETS: dict[str, Any] = {"chat_responder": "chat_responder", "planner": END}


def build_state_graph(
    model: Any | None = None,
    *,
    summarize: bool = True,
    planner_loops: int = DEFAULT_PLANNER_LOOPS,
    code_agent_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    verifier_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> StateGraph:
    """Wire the three nodes and the loop's edges into an uncompiled graph.

    Returned uncompiled so a caller can add nodes — a cancellation path, a
    human-approval gate — before calling ``.compile()``.

    Args:
        model: Shared chat model for the working nodes — the supervisor passes it
            down to every delegated specialist, so one instance serves the run.
            When omitted, the bare nodes are registered and each builds its own
            model from the runtime's metadata at call time, which is what lets
            this graph be imported, compiled and tested without an API key.
        summarize: When a ``model`` is given, let the final node word its report
            with the model. ``False`` keeps the final node deterministic. Ignored
            when ``model`` is ``None``, which is already deterministic.
        planner_loops: Supervisor tool rounds per planner visit (plan, a few
            delegations, the closing summary).
        code_agent_loops: ReAct loops each delegated code agent gets.
        verifier_loops: ReAct loops for the verifier's read-only inspection.
        retries: Extra attempts per model call for transient failures, inside the
            supervisor's calls, the specialists' loops and the verifier's loop.
            Threaded through the constructors because the nodes own those calls —
            without it a caller's retry policy would be silently dropped.
        retry_backoff: Base delay in seconds; doubles after each retry.
    """
    graph = StateGraph(SuuClawGraphState)

    if model is None:
        graph.add_node("planner", planner_node)
        graph.add_node("verifier", verifier_node)
        graph.add_node("final", final_node)
    else:
        graph.add_node(
            "planner",
            make_planner_node(
                model,
                planner_loops=planner_loops,
                code_agent_loops=code_agent_loops,
                retries=retries,
                retry_backoff=retry_backoff,
            ),
        )
        graph.add_node(
            "verifier",
            make_verifier_node(
                model,
                max_loops=verifier_loops,
                retries=retries,
                retry_backoff=retry_backoff,
            ),
        )
        graph.add_node("final", make_final_node(model if summarize else None))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    # The router's only two destinations, and the last decision the run makes:
    # finish, or go around again with a repaired plan.
    graph.add_conditional_edges("verifier", verifier_route, ROUTE_TARGETS)
    graph.add_edge("final", END)
    return graph


def build_workflow(
    model: Any | None = None,
    *,
    checkpointer: Any | None = None,
    summarize: bool = True,
    planner_loops: int = DEFAULT_PLANNER_LOOPS,
    code_agent_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    verifier_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Any:
    """Build and compile the graph — the entry point callers should use.

    Calling ``.invoke()`` directly is equally valid when a caller wants the final
    state rather than a stream of progress. This is the context-blind topology,
    though: :func:`~suuclaw.core.agent.stream_agent_events` compiles
    :func:`build_complex_workflow`, which brackets the same nodes with the
    monitor and the compressor. Reach for this one when the loop is all you want.

    Args:
        model: See :func:`build_state_graph`.
        checkpointer: Passed straight to ``compile()``. Without one, the graph
            still runs end to end; it just cannot be resumed or inspected by
            thread id.
        summarize: See :func:`build_state_graph`.
        planner_loops: See :func:`build_state_graph`.
        code_agent_loops: See :func:`build_state_graph`.
        verifier_loops: See :func:`build_state_graph`.
        retries: See :func:`build_state_graph`.
        retry_backoff: See :func:`build_state_graph`.
    """
    return build_state_graph(
        model,
        summarize=summarize,
        planner_loops=planner_loops,
        code_agent_loops=code_agent_loops,
        verifier_loops=verifier_loops,
        retries=retries,
        retry_backoff=retry_backoff,
    ).compile(checkpointer=checkpointer)


def build_complex_state_graph(
    model: Any | None = None,
    *,
    summarize: bool = True,
    planner_loops: int = DEFAULT_PLANNER_LOOPS,
    code_agent_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    verifier_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> StateGraph:
    """Wire the context-aware topology: working nodes bracketed by the monitor.

    The monitor sits on *both* sides of the working loop — after the planner's
    plan (before the verifier burns tokens judging it) and after every verifier
    round (before the planner burns tokens replanning) — so the compression
    decision is made against the real next prompt, not a stale one::

        START ──▶ planner ──▶ monitor ──┬─ "verifier" ──▶ verifier ──┐
                     ▲                  │                            │
                     │                  ├─ "context_compressor" ─┐   │
                     │                  └─ "final" ──▶ final ─┐   │   │
                     │                    ▲                  │   ▼   ▼
                     │                    │                  │ monitor
                     │              "final"│                  │   │
                     │                    └── "planner" ──┐   ▼   │
                     │                                    │  compressor
                     └────────────────────────────────────┴◀──┘

    (The compressor hands control back to ``context_next_node`` — verifier or
    planner — so the run resumes exactly where the monitor interrupted it.)

    Args:
        model: See :func:`build_state_graph`. Also drives the monitor's token
            estimate and the compressor's LLM summary; with ``None`` both fall
            back to their deterministic paths (chars/4 estimate, digest), which
            is what lets this graph compile and be tested without an API key.
        summarize: See :func:`build_state_graph` — ``False`` keeps the final
            node deterministic. Only the final node reads it: the monitor and
            the compressor keep the model either way, because measuring and
            compressing are housekeeping rather than the report's prose.
        planner_loops / code_agent_loops / verifier_loops / retries /
            retry_backoff: See :func:`build_state_graph`.
    """
    graph = StateGraph(SuuClawGraphState)

    if model is None:
        graph.add_node("planner", planner_node)
        graph.add_node("context_monitor", context_monitor_node)
        graph.add_node("context_compressor", context_compressor_node)
        graph.add_node("verifier", verifier_node)
        graph.add_node("final", final_node)
    else:
        graph.add_node(
            "planner",
            make_planner_node(
                model,
                planner_loops=planner_loops,
                code_agent_loops=code_agent_loops,
                retries=retries,
                retry_backoff=retry_backoff,
            ),
        )
        graph.add_node("context_monitor", make_context_monitor_node(model))
        graph.add_node("context_compressor", make_context_compressor_node(model))
        graph.add_node(
            "verifier",
            make_verifier_node(
                model,
                max_loops=verifier_loops,
                retries=retries,
                retry_backoff=retry_backoff,
            ),
        )
        graph.add_node("final", make_final_node(model if summarize else None))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges(
        "context_monitor", context_monitor_route, CONTEXT_MONITOR_ROUTE_TARGETS
    )
    graph.add_conditional_edges(
        "context_compressor", context_compressor_route, CONTEXT_COMPRESSOR_ROUTE_TARGETS
    )
    # After verification the run re-enters the monitor: it re-measures against
    # whatever the verifier just wrote and routes on (final, planner, compress).
    graph.add_edge("verifier", "context_monitor")
    graph.add_edge("final", END)
    return graph


def build_complex_workflow(
    model: Any | None = None,
    *,
    checkpointer: Any | None = None,
    summarize: bool = True,
    planner_loops: int = DEFAULT_PLANNER_LOOPS,
    code_agent_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    verifier_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Any:
    """Build and compile the context-aware graph.

    Same contract as :func:`build_workflow`, plus the context monitor and the
    compressor wired into the loop. This is the topology
    :func:`~suuclaw.core.agent.stream_agent_events` compiles, so it is what the
    CLI runs.

    Both writers of ``context_next_node`` are in place: the planner hands off to
    ``"verifier"`` and the verifier hands off to its verdict's destination
    (``"planner"`` while attempts remain, ``"final"`` once they are spent), so
    the loop terminates on its own and a hand-built state does not have to seed
    the channel just to keep it from spinning.

    Args:
        model / checkpointer / summarize / planner_loops / code_agent_loops /
            verifier_loops / retries / retry_backoff:
            See :func:`build_complex_state_graph` and :func:`build_workflow`.
    """
    return build_complex_state_graph(
        model,
        summarize=summarize,
        planner_loops=planner_loops,
        code_agent_loops=code_agent_loops,
        verifier_loops=verifier_loops,
        retries=retries,
        retry_backoff=retry_backoff,
    ).compile(checkpointer=checkpointer)


def build_entry_workflow(model: Any | None = None) -> Any:
    """Build and compile the intent-routing entry graph.

    The cheap gate in front of the main workflow::

        START ──▶ intent_router ──┬─ "chat_responder" ──▶ chat_responder ──▶ END
                                  └─ "planner" ──▶ END   (hand-off, see below)

    One LLM call classifies the input (:func:`~suuclaw.graph.nodes.
    intent_router_node`). Chat-shaped input goes to the lightweight responder,
    which answers with no tools and writes ``final_answer`` itself — the chat
    path never enters the main workflow, so the ``final`` node is not on it.
    Workflow-shaped input (or a low-confidence verdict, which defaults to
    workflow) ends this graph: the ``"planner"`` destination is bound to
    ``END`` via :data:`INTENT_ROUTE_TARGETS`, and *the caller* is the one who
    runs the main graph next — a LangGraph edge cannot cross a ``compile()``
    boundary, so the hand-off is the driver's job, reading
    ``state["intent_route"]`` (and ``intent_route_fn``'s answer) to know
    whether to invoke :func:`build_complex_workflow` with the same state.

    Args:
        model: Shared chat model for both nodes. When omitted, the bare nodes
            are registered and each builds its own model from the runtime's
            metadata at call time, which is what lets this graph be imported,
            compiled and tested without an API key.

    Usage::

        entry = build_entry_workflow()
        state = entry.invoke(initial_state("hi, who are you?", runtime))
        if state["intent_route"] == "chat":
            print(state["final_answer"])      # chat_responder wrote it
        else:
            result = build_complex_workflow().invoke(state)   # main graph
    """
    graph = StateGraph(SuuClawGraphState)
    if model is None:
        graph.add_node("intent_router", intent_router_node)
        graph.add_node("chat_responder", chat_responder_node)
    else:
        graph.add_node("intent_router", make_intent_router_node(model))
        graph.add_node("chat_responder", make_chat_responder_node(model))
    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges("intent_router", intent_route_fn, INTENT_ROUTE_TARGETS)
    graph.add_edge("chat_responder", END)
    return graph.compile()


def initial_state(task: str, runtime: RuntimeState, **overrides: Any) -> dict[str, Any]:
    """The seed dict a run needs: the task, and the sandbox policy.

    Only these two are required — every other channel is filled by the nodes — but
    both are: a node cannot invent the workspace it is allowed to touch, and it
    raises :class:`~suuclaw.graph.nodes.GraphStateError` rather than guessing.

    Args:
        task: The user's instruction, verbatim.
        runtime: Sandbox policy built once per run, e.g.
            ``RuntimeState.create("./sandbox")``.
        **overrides: Any other channel to seed, most usefully ``max_attempts``.

    Returns:
        A state dict ready for ``app.invoke(...)``.
    """
    state: dict[str, Any] = {"task": task, "runtime": runtime}
    state.update(overrides)
    return state
