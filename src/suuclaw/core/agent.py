"""The workflow driver, exposed as a stream of UI-friendly events.

SuuClaw's work happens in the plan → act → verify graph
(:mod:`suuclaw.graph.workflow`). This module is the seam between that graph and a
front end: it runs the compiled workflow and flattens LangGraph's stream into one
sequence of plain ``dict`` events, so a renderer never has to know about LangGraph,
``StreamWriter``, checkpoints or LangChain message objects::

    for event in stream_agent_events("list the files", workspace="./ws"):
        print(event["type"], event)

Two layers, two functions
-------------------------
========================  ======================================================
``stream_agent_events``   The public entry point. Builds the workflow, feeds it
                          one task, and re-emits its progress in the catalogue
                          below.
``stream_session_events`` The session wrapper: records each input and reply as a
                          turn under ``.suuclaw/session/``, routes through the
                          entry graph (chat answers now, work delegates) and
                          re-emits ``stream_agent_events`` for the work.
``stream_react_events``   The inner ReAct loop — what the verifier node and the
                          delegated specialists run *inside* their own node.
                          Not a front-end entry point.
========================  ======================================================

The split is load-bearing rather than stylistic: the graph's nodes drive
``stream_react_events``, so if the driver and the inner loop were one function a
node would end up invoking itself.

Recording, not re-shaping
-------------------------
While it streams, the driver also writes a checkpoint
(:mod:`suuclaw.core.checkpoint`) on every node switch and a trace
(:mod:`suuclaw.core.trace`) for the whole run, both under the workspace's
``.suuclaw/``. That bookkeeping is *recorded*, never yielded: the catalogue
below is the front-end contract, and a renderer that predates checkpoints and
traces renders this stream exactly as before.

Event catalogue
---------------
Workflow events, produced by :func:`stream_agent_events`:

=================  ==========================================================
``node_start``     A node began. Keys: ``node``.
``node_output``    A node finished. Keys: ``node``, ``data`` (the update it
                   returned into the graph state).
``node_summary``   An inner ReAct loop closed its own turn. Keys: ``node``,
                   ``content``, ``loops``, ``exhausted``, ``error``,
                   ``interrupted``. These are a *node's* last words, not the
                   run's answer.
``final_answer``   Always the last event. Keys: ``content``, ``passed``,
                   ``attempts``, ``max_attempts``, ``loops``, ``exhausted``,
                   ``error``, ``interrupted``.
``heartbeat``      No event for ``interval`` seconds while the graph runs.
                   Keys: ``node``, ``loop``, ``quiet_for``, ``interval``. A
                   liveness ping, not progress: a long tool call produces
                   several; a run gone genuinely dark now shows *how* dark.
=================  ==========================================================

Inner-loop events, forwarded live and tagged with the node that produced them:

=================  ==========================================================
``ai_message``     The model replied. Keys: ``node``, ``loop``, ``content``,
                   ``tool_calls`` (names), ``invalid_tool_calls``.
``tool_call``      The model asked for a tool. Keys: ``node``, ``loop``, ``id``,
                   ``name``, ``args``.
``tool_result``    The tool ran. Keys: ``node``, ``loop``, ``id``, ``name``,
                   ``ok``, ``result`` (the dict that was serialised into the
                   ``ToolMessage``).
``error``          A loop could not continue. Keys: ``node``, ``loop``,
                   ``message``.
``retry``          A transient model failure is being retried. Keys: ``node``,
                   ``loop``, ``attempt``, ``max_retries``, ``delay``, ``message``.
=================  ==========================================================
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..prompts.stage2 import ACTOR_PROMPT
from .approval import ApprovalHandler
from .checkpoint import CheckpointError, CheckpointManager
from .session import (
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)
from .state import DEFAULT_COMMAND_TIMEOUT, RuntimeState
from .trace import DEFAULT_TRACE_MODE, TraceRecorder

__all__ = [
    "ACTOR_PROMPT",
    "DEFAULT_RETRIES",
    "DEFAULT_RETRY_BACKOFF",
    "Event",
    "execute_tool",
    "index_tools",
    "is_transient_error",
    "iter_workflow_events",
    "message_text",
    "stream_agent_events",
    "stream_react_events",
    "stream_session_events",
]

Event = dict[str, Any]

DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 1.0

#: Supersteps one pass around the graph can cost, used to size a run's
#: ``recursion_limit``. Stage 3 has no separate actor — the specialists run
#: *inside* the planner's superstep — so the working pass is four visits
#: (planner → monitor → verifier → monitor), and compression can add a detour on
#: either side of the verifier, so two more. The final node adds one more on top
#: of the product, which :data:`RECURSION_HEADROOM` covers.
SUPERSTEPS_PER_ATTEMPT = 6

#: Headroom over the computed minimum, so an interrupted run reports the real
#: problem rather than dying on the recursion guard.
RECURSION_HEADROOM = 5

#: How long the front end may hear nothing before the driver speaks up. LangGraph's
#: sync stream wakes its consumer through a chain of futures (task done → queue
#: semaphore → waiter future → ``concurrent.futures.wait``), and that chain has
#: been observed going deaf mid-node: the task keeps working while every event it
#: writes silently piles up, which is how a run goes dark for seven minutes
#: (trace ``428377``, 2026-09-13). The pump thread plus this heartbeat turns such
#: a blackout into visible, timed ``heartbeat`` events instead of silence.
HEARTBEAT_SECONDS = 30.0
ENTRY_TIMEOUT_SECONDS = 30.0

#: Sentinel the pump thread pushes when the graph stream ends normally.
_PUMP_DONE = object()


class _PumpFailure:
    """A BaseException from the pump thread, carried to the consumer thread.

    The pump runs against a compiled workflow that may raise anything; wrapping
    keeps the sentinel channel homogeneous (one object per queue item, never a
    raw exception racing the event loop).
    """

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _invoke_with_timeout(call: Any, timeout: float) -> Any:
    """Run a side-effect-free entry call with a hard wall-clock deadline.

    Python cannot safely kill a thread blocked in an HTTP client. The entry
    graph only classifies and drafts chat text, so abandoning its daemon thread
    is safe; the main workflow can fail open instead of leaving the caller
    silent through several nested transport retries.
    """
    out_q: queue.Queue[Any] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            out_q.put((True, call()))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            out_q.put((False, exc))

    threading.Thread(
        target=invoke,
        name="suuclaw-entry-call",
        daemon=True,
    ).start()
    try:
        ok, value = out_q.get(timeout=max(0.1, timeout))
    except queue.Empty as exc:
        raise TimeoutError(f"entry route exceeded {timeout:g}s") from exc
    if ok:
        return value
    raise value


def _iter_entry_events(
    build: Any,
    inputs: Mapping[str, Any],
    timeout: float,
) -> Iterator[Event]:
    """Stream entry progress under one deadline, isolating late worker events.

    Each item carries its arrival time. A slow UI consumer may still drain
    events produced before the deadline; an entry model that finishes after it
    cannot turn a timed-out workflow fallback back into a chat result. The
    private queue is abandoned on timeout and is never shared with the main run.
    """
    out_q: queue.Queue[tuple[float, Any]] = queue.Queue()
    deadline = time.monotonic() + max(0.1, timeout)

    def pump() -> None:
        try:
            for item in build().stream(dict(inputs), stream_mode=["updates", "custom"]):
                out_q.put((time.monotonic(), item))
            out_q.put((time.monotonic(), _PUMP_DONE))
        except BaseException as exc:  # noqa: BLE001 - re-raised by the consumer
            out_q.put((time.monotonic(), _PumpFailure(exc)))

    threading.Thread(target=pump, name="suuclaw-entry-stream", daemon=True).start()
    current = ""
    while True:
        try:
            arrived, item = out_q.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise TimeoutError(f"entry route exceeded {timeout:g}s") from exc
        if arrived > deadline:
            raise TimeoutError(f"entry route exceeded {timeout:g}s")
        if item is _PUMP_DONE:
            return
        if isinstance(item, _PumpFailure):
            raise item.exc
        mode, chunk = item
        if not isinstance(chunk, Mapping):
            continue
        if mode == "custom":
            node = str(chunk.get("node") or "")
            if node and node != current:
                current = node
                yield {"type": "node_start", "node": node}
            kind = str(chunk.get("type") or "")
            if kind and kind != "node_start":
                yield dict(chunk)
        elif mode == "updates":
            for raw_node, data in chunk.items():
                node = str(raw_node)
                if node != current:
                    current = node
                    yield {"type": "node_start", "node": node}
                yield {
                    "type": "node_output",
                    "node": node,
                    "data": dict(data) if isinstance(data, Mapping) else {},
                }


def _pump_stream(
    app: Any,
    inputs: Mapping[str, Any],
    recursion_limit: int,
    out_q: queue.Queue[Any],
) -> None:
    """Drain the compiled workflow's stream into ``out_q`` on a worker thread.

    LangGraph's sync stream blocks its consumer on internal futures between
    events; running that consumption here frees the caller to timeout, heartbeat
    and stay responsive. Never raises: a failure is queued as :class:`_PumpFailure`
    (or the ``_PUMP_DONE`` sentinel on a clean end), so the consumer learns about
    it through the same channel as the events.
    """
    try:
        stream = app.stream(
            dict(inputs),
            config={"recursion_limit": recursion_limit},
            stream_mode=["updates", "custom"],
        )
        for item in stream:
            out_q.put(item)
        out_q.put(_PUMP_DONE)
    except BaseException as exc:  # noqa: BLE001 - the queue is the error channel
        out_q.put(_PumpFailure(exc))


_TRANSIENT_HINTS = (
    "connection",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "rate limit",
    "overloaded",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "internal server error",
    "429",
    "502",
    "503",
    "504",
)

# ``ACTOR_PROMPT`` is imported from :mod:`suuclaw.prompts.stage2` at the top of
# this module: the prompt text lives with the other stage-2 prompts so there is
# exactly one copy, and it is re-exported here because this module was its
# original home and ``suuclaw.core`` still publishes it.

#: Shorthands models reach for when they forget the ``file_`` / ``_tool`` prefix.
#: Canonical names and class names always take precedence over these.
_EXTRA_TOOL_ALIASES: dict[str, str] = {
    "read": "file_read",
    "cat": "file_read",
    "view": "file_read",
    "write": "file_write",
    "create": "file_write",
    "edit": "file_edit",
    "replace": "file_edit",
    "patch": "file_edit",
    "search": "grep",
    "rg": "grep",
    "shell": "bash",
    "run": "bash",
    "exec": "bash",
    "command": "bash",
}


def is_transient_error(exc: BaseException) -> bool:
    """True for transport-level failures worth retrying (dropouts, 5xx, 429).

    A wrong model name, a bad key or a malformed request never matches: those
    fail identically on every attempt, so retrying only wastes time and tokens.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _TRANSIENT_HINTS)


def message_text(content: Any) -> str:
    """Flatten message content (str, or a list of provider blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content or "")


def index_tools(tools: Iterable[Any] | Mapping[str, Any]) -> dict[str, Any]:
    """Map every acceptable spelling of a tool name onto the tool itself.

    ``ACTOR_PROMPT`` speaks about ``FileWriteTool`` while the args_schema exposes
    the snake_case name ``file_write``; models mix the two freely, so both
    spellings (plus their lowercase / underscore-free variants) resolve here.
    Canonical names always win over aliases.
    """
    items = list(tools.values()) if isinstance(tools, Mapping) else list(tools)

    index: dict[str, Any] = {}
    for tool in items:
        index.setdefault(str(tool.name), tool)

    for tool in items:
        class_name = _source_class_name(tool)
        for alias in (class_name, class_name.removesuffix("Tool")):
            for spelling in (alias, alias.lower(), alias.replace("_", "").lower()):
                if spelling:
                    index.setdefault(spelling, tool)

    by_name = {str(tool.name): tool for tool in items}
    for alias, target in _EXTRA_TOOL_ALIASES.items():
        tool = by_name.get(target)
        if tool is not None:
            index.setdefault(alias, tool)

    return index


def _source_class_name(tool: Any) -> str:
    """Original tool class name, even once wrapped in a ``StructuredTool``."""
    metadata = getattr(tool, "metadata", None)
    if isinstance(metadata, Mapping):
        recorded = metadata.get("suuclaw_class")
        if recorded:
            return str(recorded)
    return type(tool).__name__


def _lookup(tool_map: Mapping[str, Any], name: str) -> Any | None:
    if not name:
        return None
    return (
        tool_map.get(name)
        or tool_map.get(name.lower())
        or tool_map.get(name.replace("_", "").lower())
    )


def execute_tool(
    call: Mapping[str, Any],
    tools: Iterable[Any] | Mapping[str, Any],
) -> dict[str, Any]:
    """Run one tool call and return a JSON-serialisable result envelope.

    Failures are *data*, not exceptions: the envelope becomes the ``ToolMessage``
    body, so the model can read ``{"ok": false, "error": ...}`` and retry instead
    of the whole loop blowing up.
    """
    tool_map = tools if isinstance(tools, Mapping) and _looks_indexed(tools) else index_tools(tools)
    name = str(call.get("name") or "")
    raw_args = call.get("args") or {}

    if not name:
        return {"ok": False, "tool": name, "error": "tool call is missing a name"}
    tool = _lookup(tool_map, name)
    if tool is None:
        available = sorted({str(item.name) for item in tool_map.values()})
        return {
            "ok": False,
            "tool": name,
            "error": f"unknown tool {name!r}. Available: {', '.join(available)}",
        }
    if not isinstance(raw_args, Mapping):
        kind = type(raw_args).__name__
        return {"ok": False, "tool": name, "error": f"tool arguments must be an object, got {kind}"}

    try:
        output = tool.invoke(dict(raw_args))
    except Exception as exc:  # noqa: BLE001 - schema/validation errors are reported, not raised
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}

    return {"ok": True, "tool": str(tool.name), "output": "" if output is None else str(output)}


def _looks_indexed(tools: Mapping[str, Any]) -> bool:
    """True when the mapping already holds objects with a ``name`` attribute."""
    return all(hasattr(value, "invoke") for value in tools.values())


def _as_count(raw: Any, default: int) -> int:
    """A call budget read off loose input: the caller's number, else the default."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


# --------------------------------------------------------------------------- #
# public entry point: drive the workflow
# --------------------------------------------------------------------------- #
#: Which driver events make it write a checkpoint, and under what status. A node
#: finishing *is* the node switch light mode promises to snapshot; the final
#: answer closes the run; an ``error`` is when the next hard stop is likeliest.
#: Inner-loop traffic (``ai_message`` / ``tool_call`` / ...) changes no graph
#: state, and every save walks the workspace and makes a git commit — saving per
#: tool call would be a disk-shaped denial of service against the run.
_CHECKPOINT_STATUS: dict[str, str] = {
    "node_output": "running",
    "error": "error",
    "final_answer": "finished",
}


def _fold_update(state: dict[str, Any], update: Any) -> None:
    """Mirror one node's returned update into the driver's view of graph state.

    The driver never sees the graph's real state object — LangGraph keeps it — so
    checkpoints are written from this fold instead. Plain channels are copied
    as-is (they are last-write-wins in the graph too); ``messages`` is appended
    to, because ``add_messages`` appends. A faithful reducer is out of scope: the
    graph never edits an existing message id today, so append is exact.
    """
    if not isinstance(update, Mapping):
        return
    for key, value in update.items():
        if key == "task":
            # The task is the run's identity — set once from the caller (or the
            # checkpoint on resume). A node echoing it back must not overwrite
            # that, or the next checkpoint would file the run under whatever a
            # node happened to put in its update.
            continue
        if key == "messages" and isinstance(value, list):
            existing = state.get("messages")
            if not isinstance(existing, list):
                existing = []
                state["messages"] = existing
            existing.extend(value)
        else:
            state[key] = value


def stream_agent_events(
    task: str,
    *,
    workspace: str | os.PathLike[str] | None = None,
    max_attempts: int | None = None,
    max_loops: int | None = None,
    model: Any | None = None,
    state: RuntimeState | None = None,
    tools: Sequence[Any] | None = None,
    temperature: float = 0.0,
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    shell: str | None = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    summarize: bool = True,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str = "light",
    resume_workspace: str | os.PathLike[str] | None = None,
    trace_mode: str = "on",
    state_overrides: Mapping[str, Any] | None = None,
) -> Iterator[Event]:
    """Run ``task`` through the plan → act → verify workflow, yielding one event per step.

    The event stream is the catalogue below, unchanged — ``final_answer`` still
    last. What the driver adds around it: a
    :class:`~suuclaw.core.checkpoint.CheckpointManager` snapshots the run on
    every node switch, and a :class:`~suuclaw.core.trace.TraceRecorder` writes
    ``.suuclaw/traces/<trace_id>/`` for the whole run. Checkpoint and trace
    bookkeeping (``checkpoint_saved``, ``checkpoint_resumed``, ``run_start``) is
    recorded into the trace but never yielded, so a front end that predates both
    features renders this stream exactly as before.

    Args:
        task: Natural-language instruction for the workflow.
        workspace: Sandbox root; created when missing. Optional when
            ``resume_workspace`` is given — resuming continues the run *in the
            workspace its checkpoint lives in*, so the resume path supplies it.
            Ignored if ``state`` is given.
        max_attempts: How many act → verify rounds to allow before the run reports a
            failure. ``None`` takes the graph's own default. A resume that already
            spent more always gets one attempt beyond what it spent.
        max_loops: ReAct loops each delegated code agent gets per call — not a
            run-wide cap. ``None`` takes the graph's own default.
        model: Pre-built chat model, shared by the workflow's nodes (the
            supervisor passes it down to the specialists). Built via
            ``create_model()`` when omitted (which is when ``OPENAI_API_KEY`` is
            required).
        state: Pre-built :class:`RuntimeState`; the CLI passes its own so the
            workspace is resolved exactly once per process. When given, the
            approval / checkpoint / trace policy kwargs are ignored — the state
            carries those.
        tools: Accepted for backwards compatibility and otherwise unused. The graph
            assigns each node its own tool set — the supervisor gets
            ``todo_write`` + the two delegation tools, the delegated code agent
            the workspace tools plus ``todo_update`` and the notepad pair, the
            verifier read-only tools — so a caller-supplied list has nowhere to go.
        temperature: Used only when this function builds the model itself.
        command_timeout: Default ``bash`` timeout when this function builds the state.
        shell: Explicit shell prefix when this function builds the state.
        retries: Extra attempts per model call for *transient* failures (connection
            drop, 5xx, 429), applied inside the supervisor's calls and the
            specialists' and verifier's loops. Non-transient errors — bad model
            name, bad key, malformed request — abort immediately, because they
            cannot succeed on retry.
        retry_backoff: Base delay in seconds; doubles after each retry.
        summarize: Let the final node word its report with the model. ``False`` keeps
            it purely deterministic. Either way, a failing or empty summary call
            falls back to the deterministic text.
        approval_mode: Policy for risky ``bash`` commands — ``inline`` (default,
            ask ``approval_handler``), ``auto`` (run flagged), ``deny`` (refuse).
            Only used when this function builds the state.
        approval_handler: Called with an
            :class:`~suuclaw.core.approval.ApprovalRequest` in ``inline`` mode.
            Only used when this function builds the state.
        checkpoint_mode: How much of the run to snapshot under
            ``.suuclaw/checkpoints`` — ``light`` (default), ``strict``, ``off``.
            Only used when this function builds the state.
        resume_workspace: Continue a checkpointed run from this workspace instead
            of starting over: files come back from the checkpoint's git commit,
            and a strict checkpoint also restores the plan, todos, attempts and
            messages. Mutually exclusive with ``state``.
        trace_mode: How much to trace under ``.suuclaw/traces`` — ``on`` (the
            sketch's spelling of the default ``light``), ``light``, ``strict``,
            ``off``. Only used when this function builds the state.
        state_overrides: Extra channels merged into the seed state after the
            resume merge — the session driver seeds the session context this
            way. ``task``, ``runtime`` and ``max_attempts`` are filtered out
            (they are the run's identity and budget, not seedable data).
    """
    if not task or not task.strip():
        if resume_workspace is None:
            yield {"type": "error", "node": "", "loop": 0, "message": "task must not be empty"}
            yield {
                "type": "final_answer",
                "node": "final",
                "content": "",
                "passed": False,
                "attempts": 0,
                "max_attempts": 0,
                "loops": 0,
                "exhausted": False,
                "error": "task must not be empty",
                "interrupted": True,
            }
            return
        # Resuming: the checkpoint carries the task. ``suuclaw --resume <ws>`` —
        # the recovery command RECOVERY.md hands out — names no task, so demanding
        # one here would make every documented resume unusable.
    else:
        task = task.strip()

    if workspace is None and resume_workspace is None and state is None:
        raise ValueError("stream_agent_events needs a workspace (or resume_workspace)")

    if state is not None and resume_workspace is not None:
        raise ValueError(
            "pass either state or resume_workspace, not both: resuming continues a "
            "run in the workspace its checkpoint lives in"
        )

    if state is None:
        # ``on`` is accepted as the sketch's spelling of "the default" and mapped
        # here, so a caller never has to know what the default actually is.
        if str(trace_mode or "").strip().lower() == "on":
            trace_mode = DEFAULT_TRACE_MODE
        run_workspace = resume_workspace if resume_workspace is not None else workspace
        state = RuntimeState.create(
            run_workspace,
            command_timeout=command_timeout,
            shell=shell,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
        )

    # Lazy on purpose: the graph's nodes import this module to run their inner
    # loops, so a module-scope import of the graph here would close the cycle and
    # neither package would import.
    from ..graph.nodes import DEFAULT_CODE_AGENT_LOOPS, DEFAULT_MAX_ATTEMPTS
    from ..graph.workflow import build_complex_workflow, initial_state

    attempts_budget = _as_count(max_attempts, DEFAULT_MAX_ATTEMPTS)
    code_agent_loops = _as_count(max_loops, DEFAULT_CODE_AGENT_LOOPS)

    resumed = False
    resume_event: dict[str, Any] | None = None
    if resume_workspace is not None:
        try:
            inputs, resume_event = CheckpointManager.load_resume_inputs(
                state, task, max_attempts=attempts_budget
            )
            resumed = True
            # The checkpoint's task wins when the caller re-stated none; a
            # checkpoint without either is not a resume but a dead end.
            task = str(inputs.get("task") or task or "").strip()
            if not task:
                raise CheckpointError("the checkpoint records no task to resume")
        except CheckpointError as exc:
            # A missing or unreadable checkpoint is an input error, not a crash:
            # report it as events, the way the rest of the driver reports trouble.
            message = str(exc)
            yield {"type": "error", "node": "", "loop": 0, "message": message}
            yield {
                "type": "final_answer",
                "node": "final",
                "content": "",
                "passed": False,
                "attempts": 0,
                "max_attempts": attempts_budget,
                "loops": 0,
                "exhausted": False,
                "error": message,
                "interrupted": True,
            }
            return
    else:
        inputs = initial_state(task, state, max_attempts=attempts_budget)

    if state_overrides:
        # Applied after the resume merge, so an explicit override also wins over
        # restored channels. The three identity/budget keys are refuse-to-seed.
        _refuse = {"task", "runtime", "max_attempts"}
        inputs.update({key: value for key, value in state_overrides.items() if key not in _refuse})

    if model is None:
        from ..providers.openai_provider import create_model

        model = create_model(temperature=temperature)

    app = build_complex_workflow(
        model,
        summarize=summarize,
        code_agent_loops=code_agent_loops,
        retries=retries,
        retry_backoff=retry_backoff,
    )

    # Worst case rather than the common case: the monitor brackets both working
    # nodes and each compression detour costs a superstep, so the floor is
    # ``SUPERSTEPS_PER_ATTEMPT * attempts`` plus the final node and room for an
    # interrupted run to report itself instead of tripping the guard. The resume
    # path may have raised the budget above what the caller asked for, so the
    # limit follows the *effective* budget, not the caller's.
    effective_attempts = _as_count(inputs.get("max_attempts"), attempts_budget)
    recursion_limit = SUPERSTEPS_PER_ATTEMPT * effective_attempts + RECURSION_HEADROOM

    manager = CheckpointManager(state, task=task)
    recorder = TraceRecorder(state, task=task)
    recorder.start(inputs, resumed=resumed, resume_event=resume_event)
    current_state: dict[str, Any] = dict(inputs)
    saved = manager.save(current_state, status="started", latest_node="start")
    if saved is not None:
        # Recorded, not yielded, like every other checkpoint event: the trace's
        # checkpoint_count has to include the save the run opened with.
        recorder.record(saved)

    latest_node = "start"

    def _checkpoint_interrupted() -> None:
        """Bookkeeping for a run that did not reach its own end. Never raises."""
        saved = manager.save(current_state, status="interrupted", latest_node=latest_node)
        if saved is not None:
            recorder.record(saved)
        recorder.end(status="interrupted", latest_node=latest_node, final_state=current_state)

    interrupted = False
    try:
        for event in iter_workflow_events(app, inputs, recursion_limit=recursion_limit):
            recorder.record(event)
            kind = str(event.get("type") or "")
            if kind == "node_output":
                _fold_update(current_state, event.get("data"))
            node = str(event.get("node") or "")
            if node:
                latest_node = node
            status = _CHECKPOINT_STATUS.get(kind)
            if status is not None:
                saved = manager.save(
                    current_state, status=status, latest_node=latest_node, event=event
                )
                if saved is not None:
                    recorder.record(saved)
            yield event
    except KeyboardInterrupt:
        # Ctrl+C landed inside this frame (a consumer driving us with ``next()``).
        interrupted = True
    except GeneratorExit:
        # The usual Ctrl+C shape: the exception lands in the *consumer's* frame
        # and this generator is closed a moment later. Nothing may be yielded
        # from this handler — but the checkpoint and the trace are plain writes,
        # so a run killed this way still leaves a resumable snapshot behind.
        _checkpoint_interrupted()
        raise

    if interrupted:
        _checkpoint_interrupted()
        yield {
            "type": "final_answer",
            "node": "final",
            "content": "",
            "passed": False,
            "attempts": int(current_state.get("attempts") or 0),
            "max_attempts": effective_attempts,
            "loops": 0,
            "exhausted": False,
            "error": "interrupted by user (Ctrl+C); the latest checkpoint is on disk",
            "interrupted": True,
        }
        return

    # The final_answer event already wrote the "finished" checkpoint inside the
    # loop; what is left is closing the trace over the whole run.
    recorder.end(status="finished", latest_node=latest_node, final_state=current_state)


def _brief(text: str, limit: int = 300) -> str:
    """A session turn's ``summary``: the head of the content, capped."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} more character(s) withheld]"


def stream_session_events(
    task: str,
    *,
    session_workspace: str | os.PathLike[str] | None = None,
    workspace: str | os.PathLike[str] | None = None,
    max_attempts: int | None = None,
    max_loops: int | None = None,
    model: Any | None = None,
    state: RuntimeState | None = None,
    temperature: float = 0.0,
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    shell: str | None = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    entry_timeout: float = ENTRY_TIMEOUT_SECONDS,
    summarize: bool = True,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
) -> Iterator[Event]:
    """Run one conversational turn through the entry graph and the workflow.

    The session-aware front door. Per call — one user input — it does::

        task ──▶ session  load · append user turn · save (the input is on file
                │          before any model call, so a crash loses nothing)
                │
                ├─▶ session context (header + workspace file list + last turns)
                │   seeded into the graph state as ``history_summary``
                │
                ├─▶ entry graph   intent_router
                │       ├── "chat"      ──▶ chat_responder's reply is the answer;
                │       │                   assistant turn recorded, stream ends
                │       └── "workflow"  ──▶ stream_agent_events (checkpoint, trace,
                │                           approval and the main graph unchanged),
                │                           its final answer recorded as the turn
                └─▶ session saved again with the assistant turn

    Because :func:`~suuclaw.core.session.build_session_context` re-reads the
    workspace's file listing and the recent turns *on every call*, each turn's
    context reflects the workspace as it stands now — the model sees what the
    previous turns created, not a stale snapshot.

    Event contract: the catalogue is unchanged. ``final_answer`` — the stream's
    last event on every path — carries three additive keys (``route``,
    ``session_id``, ``session_turn``); the entry graph's two nodes appear as
    ordinary ``node_start`` / ``node_output`` events. A renderer that knows
    nothing about sessions renders this stream unchanged.

    Args:
        task: The user's input for this turn, verbatim.
        session_workspace: Where ``.suuclaw/session/`` lives. Defaults to the
            sandbox workspace, so a plain call keeps everything under one root;
            set it to share one conversation across several sandboxes.
        workspace: Sandbox root, created when missing. Defaults to
            ``session_workspace``. Ignored if ``state`` is given.
        max_attempts / max_loops / retries / retry_backoff / summarize:
            Passed through to :func:`stream_agent_events` for the workflow path.
        entry_timeout: Hard deadline for the side-effect-free entry graph. On
            timeout, routing fails open to the workflow instead of hanging the
            whole conversation before trace and heartbeat reporting begin.
        model: Shared chat model for the entry graph and the workflow. Built
            via ``create_model()`` when omitted.
        state: Pre-built :class:`RuntimeState`; the CLI passes its own. When
            given, ``workspace`` and the policy kwargs are ignored — the state
            carries those.
        temperature / command_timeout / shell / approval_mode / approval_handler
            / checkpoint_mode / trace_mode: Only used when this function builds
            the state or the model.
    """
    task = str(task or "").strip()
    if not task:
        yield {"type": "error", "node": "", "loop": 0, "message": "task must not be empty"}
        yield {
            "type": "final_answer",
            "node": "final",
            "content": "",
            "passed": False,
            "attempts": 0,
            "max_attempts": 0,
            "loops": 0,
            "exhausted": False,
            "error": "task must not be empty",
            "interrupted": True,
        }
        return

    if state is None:
        if str(trace_mode or "").strip().lower() == "on":
            trace_mode = DEFAULT_TRACE_MODE
        if workspace is None and session_workspace is None:
            raise ValueError("stream_session_events needs a workspace (or session_workspace)")
        state = RuntimeState.create(
            workspace if workspace is not None else session_workspace,
            command_timeout=command_timeout,
            shell=shell,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
        )

    run_ws = Path(state.workspace)
    session_ws = Path(session_workspace) if session_workspace is not None else run_ws

    session = load_or_create_session(session_ws)
    turn = append_user_turn(session, task)
    save_session(session_ws, session)
    session_context = build_session_context(run_ws, session)
    session_id = str(session.get("session_id") or "")

    if model is None:
        from ..providers.openai_provider import create_model

        model = create_model(temperature=temperature)

    from ..graph.nodes import DEFAULT_MAX_ATTEMPTS
    from ..graph.workflow import build_entry_workflow, initial_state

    budget = _as_count(max_attempts, DEFAULT_MAX_ATTEMPTS)

    entry_inputs = initial_state(task, state, max_attempts=budget)
    # The channel both entry nodes read as "session context"; the main graph's
    # memory snapshot prefers it too, so the planner sees the prior turns.
    entry_inputs["history_summary"] = session_context

    def _session_keys(route: str) -> dict[str, Any]:
        return {"route": route, "session_id": session_id, "session_turn": turn}

    entry_state: dict[str, Any] | None = {}
    entry_node = "intent_router"
    route, reason = "workflow", ""
    try:
        for event in _iter_entry_events(
            lambda: build_entry_workflow(model), entry_inputs, entry_timeout
        ):
            entry_node = str(event.get("node") or entry_node)
            if event["type"] == "node_output":
                entry_state.update(event["data"])
            yield event
    except Exception as exc:  # noqa: BLE001 - the gate fails open toward workflow
        # Timeout or graph failure must discard any partial chat route. Late
        # events remain in this invocation's private queue, never in the run.
        entry_state = None
        reason = f"entry graph failed: {type(exc).__name__}: {exc}"
        yield {"type": "error", "node": entry_node, "loop": 0, "message": reason}
    if entry_state is not None:
        route = str(entry_state.get("intent_route") or "workflow")
        reason = str(entry_state.get("intent_reason") or reason)

    if route == "chat":
        reply = str(entry_state.get("final_answer") or entry_state.get("chat_response") or "")
        append_assistant_turn(
            session, turn=turn, route="chat", content=reply, summary=_brief(reply)
        )
        save_session(session_ws, session)
        yield {
            "type": "final_answer",
            "node": "chat_responder",
            "content": reply,
            "passed": True,
            "attempts": turn,
            "max_attempts": turn,
            "loops": 0,
            "exhausted": False,
            "error": None,
            "interrupted": False,
            **_session_keys("chat"),
        }
        return

    final_event: dict[str, Any] | None = None
    try:
        for event in stream_agent_events(
            task,
            state=state,
            model=model,
            max_attempts=budget,
            max_loops=max_loops,
            retries=retries,
            retry_backoff=retry_backoff,
            summarize=summarize,
            state_overrides={"history_summary": session_context},
        ):
            kind = str(event.get("type") or "")
            if kind == "final_answer":
                event.update(_session_keys("workflow"))
                final_event = event
            yield event
    except GeneratorExit:
        # The consumer closed mid-run (Ctrl+C lands there). Nothing may be
        # yielded from this handler — but the session write is plain disk I/O,
        # so the turn still records that the run was cut short.
        append_assistant_turn(
            session,
            turn=turn,
            route="workflow",
            content="",
            summary="interrupted before the workflow reported an answer",
        )
        save_session(session_ws, session)
        raise

    content = str((final_event or {}).get("content") or "")
    append_assistant_turn(
        session, turn=turn, route="workflow", content=content, summary=_brief(content)
    )
    save_session(session_ws, session)


def iter_workflow_events(
    app: Any,
    inputs: Mapping[str, Any],
    *,
    recursion_limit: int = 25,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> Iterator[Event]:
    """Run one pass of a compiled workflow and yield the unified event stream.

    Args:
        app: A compiled :mod:`suuclaw.graph.workflow` graph.
        inputs: Seed state, e.g. from
            :func:`~suuclaw.graph.workflow.initial_state`.
        recursion_limit: Superstep budget for this run.
        heartbeat_seconds: Quiet period after which a ``heartbeat`` event is
            yielded so a silent stretch stays visible (and timed) in the UI and
            the trace.

    Both halves of LangGraph's stream are needed, and they answer different
    questions. ``custom`` carries what each node's inner ReAct loop is doing right
    now — a tool call, its output, a retry — which is what makes progress visible
    while it happens. ``updates`` carries each node's returned state once it
    finishes, which is the only place the plan, the verdict and the final answer
    can be read from.

    A node is announced with ``node_start`` when its activity is first observed,
    so the header appears once per visit — including a second visit to the planner
    after a failed attempt. Chunks on ``custom`` carrying no ``type`` are ignored
    rather than guessed at.

    The stream is consumed on a pump thread: LangGraph's sync stream blocks its
    consumer on internal futures, and that consumption must not freeze the caller
    — a heartbeat after every ``heartbeat_seconds`` of silence is what keeps a
    long tool call (or a stalled wakeup chain) distinguishable from a dead run.
    The pump is a daemon: abandoning this generator early leaves it to the
    process's exit, exactly like the LangGraph worker threads it wraps.
    """
    current: str | None = None
    last_verifier: dict[str, Any] = {}
    final_text = ""
    error: str | None = None
    loops = 0
    max_attempts = int(inputs.get("max_attempts") or 0)

    out_q: queue.Queue[Any] = queue.Queue()
    pump = threading.Thread(
        target=_pump_stream,
        args=(app, inputs, recursion_limit, out_q),
        name="suuclaw-stream-reader",
        daemon=True,
    )
    quiet_since: float | None = None

    try:
        pump.start()
        while True:
            try:
                item = out_q.get(timeout=max(0.5, heartbeat_seconds))
            except queue.Empty:
                now = time.monotonic()
                if quiet_since is None:
                    quiet_since = now
                yield {
                    "type": "heartbeat",
                    "node": current or "",
                    "loop": loops,
                    "quiet_for": round(now - quiet_since, 3),
                    "interval": heartbeat_seconds,
                }
                continue
            quiet_since = None
            if item is _PUMP_DONE:
                break
            if isinstance(item, _PumpFailure):
                raise item.exc
            mode, chunk = item
            if mode == "custom":
                if not isinstance(chunk, Mapping):
                    continue
                node = str(chunk.get("node") or "")
                if node and node != current:
                    current = node
                    yield {"type": "node_start", "node": node}
                raw_loop = chunk.get("loop")
                if _is_number(raw_loop):
                    loops = max(loops, int(raw_loop))

                kind = str(chunk.get("type") or "")
                if not kind or kind == "node_start":
                    # An explicit start is already represented by the switch
                    # above. Forwarding it too would count one visit twice.
                    continue
                if kind == "final_answer":
                    # One of the inner loops closing its own turn. The run's answer
                    # is what the final node writes, so this is relabelled rather
                    # than forwarded: two events both claiming to be the final
                    # answer is exactly the confusion this wrapper exists to stop.
                    raw_inner = chunk.get("loops")
                    yield {
                        "type": "node_summary",
                        "node": node,
                        "content": str(chunk.get("content") or ""),
                        "loops": int(raw_inner) if _is_number(raw_inner) else 0,
                        "exhausted": bool(chunk.get("exhausted")),
                        "error": chunk.get("error"),
                        "interrupted": bool(chunk.get("interrupted")),
                    }
                    continue

                event = dict(chunk)
                if node:
                    event["node"] = node
                yield event
                continue

            if mode != "updates" or not isinstance(chunk, Mapping):
                continue

            for raw_node, data in chunk.items():
                node = str(raw_node)
                update = dict(data) if isinstance(data, Mapping) else {}
                if node != current:
                    current = node
                    yield {"type": "node_start", "node": node}
                if node == "verifier":
                    last_verifier = update
                if node == "final":
                    final_text = str(update.get("final_answer") or "")
                yield {"type": "node_output", "node": node, "data": update}
    except Exception as exc:  # noqa: BLE001 - surfaced as an event, not a traceback
        error = f"{type(exc).__name__}: {exc}"
        if "Recursion" in type(exc).__name__:
            error += f" (recursion_limit was {recursion_limit}; raise it with --max-attempts)"
        yield {"type": "error", "node": current or "", "loop": loops, "message": error}

    passed = bool(last_verifier.get("passed"))
    attempts = int(last_verifier.get("attempts") or 0)
    if not final_text and error:
        final_text = f"The run stopped before it could report an outcome.\n{error}"

    yield {
        "type": "final_answer",
        "node": "final",
        "content": final_text,
        "passed": passed,
        "attempts": attempts,
        "max_attempts": max_attempts or attempts,
        "loops": loops,
        "exhausted": not passed,
        "error": error,
        "interrupted": error is not None,
    }


def _is_number(raw: Any) -> bool:
    """True for values ``int()`` will accept, so counters can be read safely."""
    try:
        int(raw)
    except (TypeError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- #
# inner loop: what the graph's verifier node and delegated specialists run
# --------------------------------------------------------------------------- #
def stream_react_events(
    task: str,
    *,
    workspace: str | os.PathLike[str],
    max_loops: int = 10,
    model: Any | None = None,
    state: RuntimeState | None = None,
    tools: Sequence[Any] | None = None,
    temperature: float = 0.0,
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    shell: str | None = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    system_prompt: str = ACTOR_PROMPT,
    collect_messages: list[Any] | None = None,
) -> Iterator[Event]:
    """Run one ReAct loop for ``task`` and yield one event dict per step.

    This is the engine the graph's verifier node and delegated specialists drive
    *inside* their own node. It talks to exactly one model and one tool list,
    which is what makes
    it usable as a node body; the graph itself is driven by
    :func:`stream_agent_events`.

    Args:
        task: The brief handed to this loop (for a node, its assembled request).
        workspace: Sandbox root; created when missing. Ignored if ``state`` is given.
        max_loops: Maximum model⇄tool iterations before giving up.
        model: Pre-built chat model. Built via ``create_model()`` when omitted
            (which is when ``OPENAI_API_KEY`` is required).
        state: Pre-built :class:`RuntimeState`.
        tools: Pre-built tools for this loop.
        temperature: Used only when this function builds the model itself.
        command_timeout: Default ``bash`` timeout when this function builds the state.
        shell: Explicit shell prefix when this function builds the state.
        retries: Extra attempts for *transient* model failures (connection drop,
            5xx, 429). Non-transient errors — bad model name, bad key, malformed
            request — abort immediately, because they cannot succeed on retry.
        retry_backoff: Base delay in seconds; doubles after each retry.
        system_prompt: The role this loop plays. A node passes its own so the loop,
            retry policy and tool plumbing stay in one place instead of being
            re-implemented per node.
        collect_messages: When given, the messages the *loop itself* produced
            (assistant turns and tool results — not the system/human scaffolding,
            which the caller can rebuild) are appended to it. That is what lets a
            graph node hand them to the ``messages`` channel's reducer.
    """
    if not task or not task.strip():
        yield {"type": "error", "loop": 0, "message": "task must not be empty"}
        yield {
            "type": "final_answer",
            "content": "",
            "loops": 0,
            "exhausted": False,
            "error": "task must not be empty",
            "interrupted": True,
        }
        return

    if state is None:
        state = RuntimeState.create(
            workspace,
            command_timeout=command_timeout,
            shell=shell,
        )

    if tools is None:
        # lazy: `tools.registry` imports `core.state`, and `core/__init__` imports
        # this module — a module-scope import here closes that cycle.
        from ..tools.registry import build_tools

        tools = build_tools(state)
    tool_list = list(tools)

    if model is None:
        from ..providers.openai_provider import create_model

        model = create_model(temperature=temperature)

    agent = model.bind_tools(tool_list)
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task),
    ]

    last_content = ""
    last_error: str | None = None
    loops_used = 0
    completed = False  # the model answered without asking for more tools

    for loop in range(1, max_loops + 1):
        loops_used = loop

        response: Any = None
        for attempt in range(retries + 1):
            try:
                response = agent.invoke(messages)
                break
            except Exception as exc:  # noqa: BLE001 - surfaced as an event, not a traceback
                if attempt < retries and is_transient_error(exc):
                    delay = retry_backoff * (2**attempt)
                    yield {
                        "type": "retry",
                        "loop": loop,
                        "attempt": attempt + 1,
                        "max_retries": retries,
                        "delay": delay,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                    time.sleep(delay)
                    continue
                last_error = f"{type(exc).__name__}: {exc}"
                yield {"type": "error", "loop": loop, "message": last_error}
                break

        if response is None:
            break

        messages.append(response)
        content = message_text(getattr(response, "content", "")).strip()
        if content:
            last_content = content

        calls = list(getattr(response, "tool_calls", None) or [])
        invalid = list(getattr(response, "invalid_tool_calls", None) or [])

        yield {
            "type": "ai_message",
            "loop": loop,
            "content": content,
            "tool_calls": [str(call.get("name") or "") for call in calls],
            "invalid_tool_calls": [str(item.get("name") or "") for item in invalid],
        }

        if not calls:
            if invalid:
                preview = [str(item)[:200] for item in invalid]
                last_error = f"model produced malformed tool call(s): {preview}"
                yield {"type": "error", "loop": loop, "message": last_error}
            else:
                completed = True
            break

        for call in calls:
            name = str(call.get("name") or "")
            args = call.get("args") or {}
            call_id = str(call.get("id") or name)

            yield {"type": "tool_call", "loop": loop, "id": call_id, "name": name, "args": args}

            result = execute_tool(call, tool_list)

            yield {
                "type": "tool_result",
                "loop": loop,
                "id": call_id,
                "name": name,
                "ok": bool(result.get("ok")),
                "result": result,
            }

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=call_id,
                )
            )

    if collect_messages is not None:
        # Skip the [system, human] scaffolding: the caller rebuilds that from
        # graph state on every attempt, so replaying it would splice several
        # half-conversations together.
        collect_messages.extend(messages[2:])

    yield {
        "type": "final_answer",
        "content": last_content,
        "loops": loops_used,
        "exhausted": not completed and last_error is None,
        "error": last_error,
        "interrupted": last_error is not None,
    }
