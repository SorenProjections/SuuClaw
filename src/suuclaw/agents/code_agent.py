"""The codeAgent node: the implementation specialist of the multi-agent flow.

Stage 3's second specialist takes what :mod:`~suuclaw.agents.search_agent`
researched and turns it into working files: it edits the workspace, runs
checks, and reports progress on the plan as it goes.

Where it sits
-------------
::

    planner ──▶ search_agent ──▶ code_agent ──▶ verifier ──▶ final

Both specialists are optional steps; a workflow that skips the search step
hands the code agent the planner's instruction alone.

Tools are the sandbox
---------------------
Unlike the search agent — whose *one* tool defines it — the code agent gets
the full workspace set from :func:`~suuclaw.tools.registry.build_tools`
(``file_read`` / ``file_write`` / ``file_edit`` / ``grep`` / ``bash``), plus:

* ``todo_update`` — progress on the plan. The model reports ``in_progress`` /
  ``completed`` / ``blocked`` per step, and the node folds the updated plan
  back into its return value so the caller can put it in graph state.
* ``notepad_append`` / ``notepad_read`` — durable notes that survive context
  compression (:mod:`~suuclaw.tools.notepad_tools`).

What it emits
-------------
On ``writer`` — LangGraph's custom stream, so a front end can render progress
live (same contract as the search agent, tagged ``node="code"``):

===================  =========================================================
``tool_call``        A tool was requested. Keys: ``node``, ``loop``, ``id``,
                     ``name``, ``args``.
``tool_result``      The tool finished. Keys: ``node``, ``loop``, ``id``,
                     ``name``, ``ok``, ``output`` (clipped preview), ``error``.
``final_answer``     The loop closed its own turn. The workflow driver relabels
                     this to ``node_summary`` — only the final node writes the
                     run's answer.
``retry``, ``error``  A transient model failure being retried, or the loop
                     could not continue.
===================  =========================================================

What it returns
---------------
``{"ok", "summary", "todos", "messages", "tool_events", "error"}`` — a dict,
never a state update. ``todos`` is the post-run plan snapshot (the store is
seeded from ``state["todos"]`` and mutated by the model's own ``todo_update``
calls), ready to drop into the ``todos`` channel. ``messages`` holds only the
loop's own turns; the scaffolding is rebuilt from state on every call.

Layered memory
--------------
:func:`~suuclaw.graph.memory.build_layered_memory` assembles the full
three-layer snapshot (rules, working memory, compressed history) from the
graph state, and the runtime announces it on the stream as a ``memory``
event before the first model call — same contract as the planner and the
verifier. The hand-rolled two-layer snapshot this module used to build is
gone: the shared assembler already carries research notes and the notepad,
so keeping a second copy was two sources of truth waiting to drift.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..core.agent import (
    DEFAULT_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    is_transient_error,
    message_text,
)
from ..core.state import RuntimeState
from ..prompts.stage3 import CODE_AGENT_PROMPT
from ..tools.notepad_tools import NotepadAppendTool, NotepadReadTool
from ..tools.registry import as_langchain_tools, build_tools
from ..tools.todo_tools import TodoStore, TodoUpdateTool

__all__ = [
    "CODE_AGENT_PROMPT",
    "CODE_NODE_NAME",
    "DEFAULT_CODE_LOOPS",
    "CodeAgentError",
    "make_code_agent",
    "run_code_agent",
]

#: ReAct loops one implementation step gets. Mirrors the stage-2 actor's own
#: budget for a single attempt: implement one instruction, not the whole task.
DEFAULT_CODE_LOOPS = 10

#: Tag on every event this agent emits, so a renderer can label its section.
CODE_NODE_NAME = "code"

#: Per-block cap on what state hands the agent. The task, the instruction and
#: the memory snapshot each get their own budget so none crowds out the rest.
MAX_BLOCK_CHARS = 4000

#: How much of one tool's output a ``tool_result`` event carries. The full text
#: already went to the model as the ``ToolMessage`` body; the event is for a
#: live front end, which needs the shape, not every byte.
MAX_EVENT_OUTPUT_CHARS = 600


class CodeAgentError(ValueError):
    """Raised when the state handed to the agent is missing something it cannot invent."""


# --------------------------------------------------------------------------- #
# state access
# --------------------------------------------------------------------------- #
def _runtime(state: Mapping[str, Any]) -> RuntimeState:
    """The ``RuntimeState`` every tool needs, or a clear complaint about it."""
    if not isinstance(state, Mapping):
        kind = type(state).__name__
        raise CodeAgentError(f"code agent needs a graph state mapping, got {kind}")
    runtime = state.get("runtime")
    if runtime is None:
        raise CodeAgentError(
            "code agent state has no 'runtime' — seed the run with "
            "{'task': ..., 'runtime': RuntimeState.create(workspace)}"
        )
    return runtime


def _default_model(state: Mapping[str, Any]) -> Any:
    """Build a model from the runtime's metadata (the CLI records the model id there).

    Imported lazily so that merely importing this module never requires an API key.
    """
    from ..providers.openai_provider import create_model

    metadata = getattr(_runtime(state), "metadata", None) or {}
    return create_model(metadata.get("model"), temperature=0.0)


def _stream_writer() -> Any | None:
    """LangGraph's custom-stream writer, or ``None`` outside a running graph.

    Mirrors the helper in :mod:`suuclaw.graph.nodes` without importing it —
    an agent package reaching into a graph module's privates is the coupling
    that would make stage-3 agents hard to lift out of their stage-2 home.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001 - no run, no stream, no problem
        return None


def _emit(writer: Any | None, event: Mapping[str, Any]) -> None:
    """Forward one event to the custom stream, tagged with this agent's node.

    A broken consumer must not abort real work, hence the swallow.
    """
    if writer is None:
        return
    payload = dict(event)
    payload.setdefault("node", CODE_NODE_NAME)
    try:
        writer(payload)
    except Exception:  # noqa: BLE001 - a dead stream is not a reason to stop
        pass


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #
def _clip(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (+{len(text) - limit} more character(s) withheld)"


def _session_context(state: Mapping[str, Any], runtime: RuntimeState, store: TodoStore) -> str:
    """Where this step sits in the run: the workspace, the plan, and past failures."""
    lines = [f"- workspace: {runtime.name}"]

    last_error = str(state.get("last_error") or "").strip()
    if last_error:
        lines.append(f"- previous attempt failed: {_clip(last_error, 1500)}")

    lines.append(f"- current plan:\n{store.render()}")
    return "\n".join(lines)


def _code_agent_input(
    state: Mapping[str, Any],
    instruction: str,
    memory: Mapping[str, Any],
    *,
    runtime: RuntimeState,
    store: TodoStore,
) -> str:
    """The agent's brief: task, this step's instruction, session context, memory.

    ``memory`` is the assembled three-layer snapshot the caller built (see
    :func:`~suuclaw.graph.memory.build_layered_memory`); it rides last, under
    the     same ``# Layered memory`` header every other node's brief uses, so one
    prompt reader handles all of them.
    """
    from ..graph.memory import append_memory_block  # lazy: see the note in _implement

    task = _clip(str(state.get("task") or ""), 2000) or "(no task given)"
    ask = _clip(str(instruction or ""), 2000)
    ask = ask or "(no specific instruction — implement the task above)"
    return append_memory_block(
        "\n\n".join(
            [
                f"# Task\n{task}",
                f"# What to implement now\n{ask}",
                f"# Session context\n{_session_context(state, runtime, store)}",
                "# Your job\n"
                "Implement the instruction above inside the workspace. Work the plan "
                "one todo at a time: mark a todo in_progress before starting it and "
                "completed once it is done (blocked, with a note, if you cannot "
                "proceed). BashTool already runs inside the workspace, so use "
                "relative paths — never `cd /workspace`. Record durable findings in "
                "the notepad as you go. End with a concise summary of the files "
                "changed and the checks you ran.",
            ]
        ),
        memory,
    )


# --------------------------------------------------------------------------- #
# one loop
# --------------------------------------------------------------------------- #
def _positive_int(raw: Any, default: int) -> int:
    """A count read off loose input: the caller's number when it is usable."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def _invoke(
    agent: Any,
    messages: list[Any],
    *,
    loop: int,
    writer: Any | None,
    retries: int,
    retry_backoff: float,
) -> tuple[Any | None, str]:
    """One model call, retrying only *transient* failures.

    Returns ``(response, "")`` or ``(None, reason)`` — the same policy as the
    search agent: a dropped connection or a 429 is worth another attempt, a
    bad key fails identically every time and is reported immediately.
    """
    for attempt in range(retries + 1):
        try:
            return agent.invoke(messages), ""
        except Exception as exc:  # noqa: BLE001 - surfaced as an event, not a traceback
            message = f"{type(exc).__name__}: {exc}"
            if attempt >= retries or not is_transient_error(exc):
                _emit(writer, {"type": "error", "loop": loop, "message": message})
                return None, message

            delay = retry_backoff * (2**attempt)
            _emit(
                writer,
                {
                    "type": "retry",
                    "loop": loop,
                    "attempt": attempt + 1,
                    "max_retries": retries,
                    "delay": delay,
                    "message": message,
                },
            )
            time.sleep(delay)

    return None, "the model call did not complete"  # pragma: no cover - loop returns or raises


def _execute_tool(tools_by_name: Mapping[str, Any], name: str, args: Mapping[str, Any]) -> str:
    """Run one bound tool by name and return its text output — never raising.

    A model-supplied tool name is trusted no further than the binding: an
    unknown or misspelled name becomes an error *message* listing the real
    ones, which the model can read and correct on its next loop. Tool
    exceptions are caught the same way — the registry's contract is that
    failures are text the model can act on, not exceptions that end the step.
    """
    tool = tools_by_name.get(name)
    if tool is None:
        known = ", ".join(sorted(tools_by_name)) or "(none)"
        return f"Error: unknown tool {name!r}. Available tools: {known}"
    try:
        return str(tool.invoke(dict(args)))
    except Exception as exc:  # noqa: BLE001 - the ToolMessage body is how a caller hears about it
        return f"{type(exc).__name__}: {exc}"


def _fallback_summary(
    *,
    calls_made: int,
    calls_ok: int,
    store: TodoStore,
    error: str,
    loops: int,
    answered: bool,
) -> str:
    """The step's outcome in deterministic words, for when the model produced none.

    A step that burned its loops mid-edit still owes the next node an honest
    picture: how much of the plan moved, how many tool calls landed, and why
    the turn never closed.
    """
    lines: list[str] = []
    if error:
        lines.append(f"The implementation step failed: {error}")

    counts = store.counts()
    moved = counts.get("completed", 0) + counts.get("blocked", 0)
    lines.append(
        f"Plan progress: {moved}/{len(store.items)} todo(s) completed or blocked "
        f"(in progress: {counts.get('in_progress', 0)}, pending: {counts.get('pending', 0)})."
    )
    lines.append(f"Ran {calls_made} tool call(s), {calls_ok} succeeded.")
    if not answered and not error:
        lines.append(
            f"The agent used all {loops} of its loops without a closing summary; "
            "the workspace may be in a partially edited state."
        )
    return "\n".join(lines)


def _implement(
    state: Mapping[str, Any],
    instruction: str,
    *,
    model: Any,
    writer: Any | None,
    max_loops: int,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> dict[str, Any]:
    """The loop itself. See :func:`run_code_agent` for the contract."""
    runtime = _runtime(state)
    store = TodoStore.from_items(state.get("todos"))

    # The assembled three-layer snapshot, announced before the first call so a
    # front end can show what the agent is about to see. Imported here rather
    # than at module level: graph/__init__ pulls in this module via the nodes,
    # so a top-level import would be the cycle the lazy-import rule exists for.
    from ..graph.memory import build_layered_memory, memory_event

    memory = build_layered_memory(state, node="codeAgent")
    # The stream tag is this agent's rendering identity (``code``), so the
    # driver's node tracking does not flicker mid-section; the snapshot's own
    # ``working_memory.node`` stays "codeAgent" — the handoff naming.
    _emit(writer, memory_event(memory, node=CODE_NODE_NAME))

    bound = build_tools(runtime)
    extras = as_langchain_tools(
        [
            TodoUpdateTool(runtime, store),
            NotepadAppendTool(runtime),
            NotepadReadTool(runtime),
        ]
    )
    tools = [*bound, *extras]
    tools_by_name = {tool.name: tool for tool in tools}
    agent = model.bind_tools(tools)

    messages: list[Any] = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(
            content=_code_agent_input(state, instruction, memory, runtime=runtime, store=store)
        ),
    ]

    produced: list[Any] = []  # the loop's own turns; the scaffolding is not replayed
    tool_events: list[dict[str, Any]] = []
    calls_made = 0
    calls_ok = 0
    last_content = ""
    last_error: str | None = None
    answered = False
    loops_used = 0

    for loop in range(1, max_loops + 1):
        loops_used = loop
        response, failure = _invoke(
            agent,
            messages,
            loop=loop,
            writer=writer,
            retries=retries,
            retry_backoff=retry_backoff,
        )
        if response is None:
            last_error = failure
            break

        messages.append(response)
        produced.append(response)
        content = message_text(getattr(response, "content", "")).strip()
        if content:
            last_content = content

        calls = list(getattr(response, "tool_calls", None) or [])
        invalid = list(getattr(response, "invalid_tool_calls", None) or [])
        if not calls:
            if invalid:
                # The model's tool-call JSON did not survive the hop. Say so rather
                # than pass the turn off as a clean close.
                preview = [str(item)[:200] for item in invalid]
                last_error = f"model produced malformed tool call(s): {preview}"
                _emit(writer, {"type": "error", "loop": loop, "message": last_error})
            else:
                answered = True
            break

        for position, call in enumerate(calls, start=1):
            name = str(call.get("name") or "")
            raw_args = call.get("args")
            call_id = str(call.get("id") or f"{CODE_NODE_NAME}-{loop}-{position}")
            shown_args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
            _emit(
                writer,
                {
                    "type": "tool_call",
                    "loop": loop,
                    "id": call_id,
                    "name": name,
                    "args": shown_args,
                },
            )

            output = _execute_tool(tools_by_name, name, shown_args)
            calls_made += 1
            ok = not output.startswith("Error:")
            if ok:
                calls_ok += 1

            _emit(
                writer,
                {
                    "type": "tool_result",
                    "loop": loop,
                    "id": call_id,
                    "name": name,
                    "ok": ok,
                    "output": output[:MAX_EVENT_OUTPUT_CHARS],
                    "error": "" if ok else output,
                },
            )

            reply = ToolMessage(content=output, tool_call_id=call_id)
            messages.append(reply)
            produced.append(reply)
            tool_events.append(
                {
                    "loop": loop,
                    "id": call_id,
                    "name": name,
                    "args": shown_args,
                    "ok": ok,
                    "output": output,
                }
            )

    if last_error is not None:
        ok, error = False, last_error
    else:
        ok, error = True, ""

    # A closing turn is a report; anything the model said *before* an unfinished
    # tool call is not — replaying that out of context reads as a conclusion the
    # step never reached. So an exhausted loop is summarised from the data it
    # actually collected instead.
    summary = (
        last_content
        if (answered and last_content)
        else _fallback_summary(
            calls_made=calls_made,
            calls_ok=calls_ok,
            store=store,
            error=error,
            loops=loops_used,
            answered=answered,
        )
    )

    _emit(
        writer,
        {
            "type": "final_answer",
            "content": summary,
            "loops": loops_used,
            "exhausted": not answered and last_error is None,
            "error": last_error or "",
            "interrupted": last_error is not None,
        },
    )

    return {
        "ok": ok,
        "summary": summary,
        "todos": store.snapshot(),
        "messages": produced,
        "tool_events": tool_events,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def run_code_agent(
    state: Mapping[str, Any],
    instruction: str,
    *,
    writer: Any | None = None,
    max_loops: int = DEFAULT_CODE_LOOPS,
    model: Any | None = None,
) -> dict[str, Any]:
    """Implement ``instruction`` in the workspace and report the outcome.

    Args:
        state: The graph state. ``runtime`` (the policy every tool needs),
            ``task`` (the user's original request), ``todos`` (the plan whose
            progress the model reports via ``todo_update``) and
            ``research_notes`` (findings from an earlier search step) are read
            from it.
        instruction: What *this* step should implement. Usually one planner
            todo, e.g. "write tests/test_life.py covering the step function".
        writer: LangGraph's custom-stream writer. When omitted it is picked up
            from the running graph, so a node can call this without passing
            one, and a plain script call works with no stream at all.
        max_loops: Model⇄tool iterations before the step gives up.
        model: Pre-built chat model. Built from the runtime's metadata when
            omitted; the supervisor passes its own so one model instance serves
            the whole graph, and a test can inject a fake.

    Returns:
        ``{"ok", "summary", "todos", "messages", "tool_events", "error"}``.

        ``ok`` is ``False`` only when the model call itself failed; a tool
        error is data the model already saw and had the chance to correct.
        ``summary`` is the model's closing prose when it wrote one (the
        files-changed/checks-run report the prompt asks for), and a
        deterministic digest of plan progress and tool calls when it did not.
        ``todos`` is the post-run plan snapshot — fold it into the graph's
        ``todos`` channel. ``messages`` holds only this loop's own turns.

    Raises:
        CodeAgentError: when ``state`` has no ``runtime``. Tool failures are
            *not* errors of this kind — they travel as text and events, so the
            step can still close with an honest report.
    """
    return _implement(
        state,
        instruction,
        model=model if model is not None else _default_model(state),
        writer=writer if writer is not None else _stream_writer(),
        max_loops=_positive_int(max_loops, DEFAULT_CODE_LOOPS),
    )


def make_code_agent(
    model: Any,
    *,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Any:
    """Code agent bound to one pre-built model (the form a compiled graph wants).

    Mirrors the ``make_*_node`` factories in :mod:`suuclaw.graph.nodes`: a
    graph builds its model once and hands it to every node, and a test can
    hand it a fake without an API key or a network.
    """

    def agent(
        state: Mapping[str, Any],
        instruction: str,
        *,
        writer: Any | None = None,
        max_loops: int = DEFAULT_CODE_LOOPS,
    ) -> dict[str, Any]:
        return _implement(
            state,
            instruction,
            model=model,
            writer=writer if writer is not None else _stream_writer(),
            max_loops=_positive_int(max_loops, DEFAULT_CODE_LOOPS),
            retries=retries,
            retry_backoff=retry_backoff,
        )

    agent.__name__ = "run_code_agent"
    return agent
