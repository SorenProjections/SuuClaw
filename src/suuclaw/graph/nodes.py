"""The supervisor → verify nodes, plus the router that closes the loop.

Shape of the workflow
---------------------
::

    START ──▶ planner ──▶ verifier ──┬─ "final" ──▶ final ──▶ END
                 ▲                   │
                 └──── "planner" ────┘

Stage 3 has no separate actor: the **planner is the supervisor**, and the work
happens inside its own tool calls. It cannot touch the workspace or the web —
its tool list is the sandbox — so anything real has to be delegated:

* ``todo_write`` publishes or revises the plan (data, never prose);
* ``call_search_agent`` hands one research instruction to ``searchAgent``
  (:func:`~suuclaw.agents.search_agent.run_search_agent`);
* ``call_code_agent`` hands one implementation instruction to ``codeAgent``
  (:func:`~suuclaw.agents.code_agent.run_code_agent`).

The planner therefore runs a small ReAct loop of its own — plan, delegate, read
what came back, delegate again, summarise — instead of the stage-2 single
structured call. Forcing ``tool_choice`` is no longer possible (a forced call
could only ever be ``todo_write``, which would forbid delegation), which also
retires the DeepSeek "thinking mode rejects tool_choice" workaround entirely.

``verifier_route`` sends a failed run back to the **planner**: a failed attempt
means the *plan* was wrong (or under-specified), so replanning is what gives the
retry a chance of being different.

``final`` owns the ``final_answer`` channel and is the only writer of it. It
needs no model, so a run that exhausted its retry budget still ends with a
report rather than a traceback.

Two mechanisms worth calling out, because they are where the design lives:

**Specialist results are data, folded by the node.** A tool cannot write graph
state, so the delegation tools return *fragments* — the exact channels they
propose to update — into a list the node merges when the loop ends:
``research_notes`` blocks are concatenated, ``sources`` deduplicated by URL,
``agent_handoffs`` appended, ``todos`` and ``code_agent_summary`` last-write-wins,
``messages`` extended through the channel's reducer. Each delegation also emits
a ``handoff`` event on the custom stream, so a front end can show the hand-off
as it happens.

**Verification is judged twice, on purpose.** The verifier node runs
``verification_commands`` itself and stores the real exit codes, *and* asks the
model to judge the acceptance criteria from that evidence. A command's exit code
and the question "is this actually done?" are different questions; when they
disagree, the deterministic one wins (``passed`` is vetoed), because a model
saying "looks fine" is exactly the failure mode this whole node exists to catch.

Model injection: every node has a bare ``x_node(state)`` form for a compiled
graph, and a ``make_x_node(model)`` form that closes over one model instance.
The supervisor passes *its* model down to the specialists, so one model instance
serves the whole graph.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from ..agents.code_agent import DEFAULT_CODE_LOOPS, run_code_agent
from ..agents.search_agent import run_search_agent
from ..core.agent import (
    DEFAULT_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    is_transient_error,
    message_text,
    stream_react_events,
)
from ..core.state import RuntimeState
from ..prompts.stage3 import PLANNER_PROMPT
from ..prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from ..prompts.stage5 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT
from ..tools.bash_tool import run_command
from ..tools.registry import SuuClawTool, as_langchain_tools, build_read_only_tools
from ..tools.todo_tools import TodoStore, TodoWriteTool
from .memory import (
    CODE_AGENT_SUMMARY_CHARS,
    CONTEXT_SUMMARY_CHARS,
    HISTORY_SUMMARY_FILENAME,
    RESEARCH_NOTES_CHARS,
    CompressionEvent,
    _short_text,
    _trim_handoffs,
    append_memory_block,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from .prompts import (
    FINAL_PROMPT,
    PLANNER_REVISION_ADDENDUM,
    VERIFIER_PROMPT,
    final_request,
    initial_plan_request,
    revision_plan_request,
    verifier_request,
)
from .state import AgentHandoff, SuuClawGraphState, VerificationCheck, VerificationResult

__all__ = [
    "CONTEXT_TOKEN_LIMIT_ENVS",
    "DEFAULT_CODE_AGENT_LOOPS",
    "DEFAULT_CONTEXT_TOKEN_LIMIT",
    "DEFAULT_INTENT_CONFIDENCE",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_PLANNER_LOOPS",
    "DEFAULT_VERIFIER_LOOPS",
    "CallCodeAgentTool",
    "CallSearchAgentTool",
    "GraphStateError",
    "chat_responder_node",
    "context_compressor_node",
    "context_compressor_route",
    "context_monitor_node",
    "context_monitor_route",
    "extract_json_object",
    "final_answer_text",
    "final_node",
    "intent_route_fn",
    "intent_router_node",
    "make_chat_responder_node",
    "make_code_agent_tool",
    "make_context_compressor_node",
    "make_context_monitor_node",
    "make_final_node",
    "make_intent_router_node",
    "make_planner_node",
    "make_search_agent_tool",
    "make_verifier_node",
    "planner_node",
    "verifier_node",
    "verifier_route",
]

#: ReAct rounds the supervisor gets per visit: publish the plan, a few
#: delegations, and the closing summary. Delegations are expensive (each runs a
#: specialist loop), so the budget counts *planner* rounds, not tool calls.
DEFAULT_PLANNER_LOOPS = 6

#: Loops a delegated code agent gets, unless the caller overrides it.
DEFAULT_CODE_AGENT_LOOPS = DEFAULT_CODE_LOOPS

#: Loops the verifier's read-only agent gets. Inspection is cheaper than
#: implementation, but a verifier that never finishes reading is also a failure.
DEFAULT_VERIFIER_LOOPS = 10

#: plan → verify rounds before giving up, when the state does not say otherwise.
DEFAULT_MAX_ATTEMPTS = 3

#: How much of one command's output is stored per stream. Enough to diagnose,
#: small enough that three commands still fit in a prompt.
VERIFICATION_OUTPUT_CHARS = 4000

#: How much of a specialist's report travels back to the planner as the tool
#: result. The full summary is in ``code_agent_summary`` / ``research_notes``;
#: this is the model-facing receipt.
HANDOFF_REPORT_CHARS = 3000

#: Token budget the context monitor enforces when the state does not carry one.
#: The raw transcript plus the layered-memory snapshot must stay under this, or
#: the monitor flags a compression pass for the router to act on.
DEFAULT_CONTEXT_TOKEN_LIMIT = 400_000

#: Environment override for the context token budget.
CONTEXT_TOKEN_LIMIT_ENVS: tuple[str, ...] = (
    "SUUCLAW_CONTEXT_TOKEN_LIMIT",
)

#: Characters per token in the monitor's fallback estimate. Rough by design —
#: English prose averages ~4 chars/token and the estimate only has to decide
#: "comfortably under" vs "time to compress", not bill anyone.
CHARS_PER_TOKEN = 4

#: Per-message cap inside the compression brief. Long stdout/stderr is exactly
#: what compression is *for*, so feeding it back whole would defeat the point.
TRANSCRIPT_MESSAGE_CHARS = 2000

#: Cap on the digest that becomes the new context window (the ``AIMessage``).
#: Generous on purpose — this is the one place compression must not lose the
#: plot; the channels downstream apply their own, tighter budgets.
DIGEST_CHARS = 6000

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class GraphStateError(ValueError):
    """Raised when the graph state is missing something a node cannot invent."""


# --------------------------------------------------------------------------- #
# state access helpers
# --------------------------------------------------------------------------- #
def _runtime(state: Mapping[str, Any]) -> RuntimeState:
    """The ``RuntimeState`` every tool needs, or a clear complaint about it."""
    runtime = state.get("runtime")
    if runtime is None:
        raise GraphStateError(
            "graph state has no 'runtime' — seed the run with "
            "{'task': ..., 'runtime': RuntimeState.create(workspace)}"
        )
    return runtime


def _max_attempts(state: Mapping[str, Any]) -> int:
    raw = state.get("max_attempts")
    try:
        value = int(raw) if raw is not None else DEFAULT_MAX_ATTEMPTS
    except (TypeError, ValueError):
        value = DEFAULT_MAX_ATTEMPTS
    return max(value, 1)


def _verdict_destination(
    state: Mapping[str, Any], *, passed: bool, attempts: int
) -> Literal["final", "planner"]:
    """Where a run goes once the verifier has spoken: finish, or replan.

    The one place this rule lives. Two topologies call it — ``verifier_route``
    (the simple graph, where the router asks) and the verifier node itself (the
    context-aware graph, which routes through the monitor instead) — so the two
    cannot drift apart on the single decision that ends a run. ``attempts``
    counts *completed* rounds and is incremented by the verifier, so the value
    passed here already includes the round being judged.
    """
    if passed:
        return "final"
    if attempts >= _max_attempts(state):
        return "final"
    return "planner"


def _default_model(state: Mapping[str, Any]) -> Any:
    """Build a model from the runtime's metadata (the CLI records the model id there).

    Imported lazily so that merely importing this module never requires an API key.
    """
    from ..providers.openai_provider import create_model

    metadata = getattr(_runtime(state), "metadata", None) or {}
    return create_model(metadata.get("model"), temperature=0.0)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull one JSON object out of model output.

    Tolerates the three ways models wrap JSON in practice: bare, inside a
    ```` ```json ```` fence, or with prose on either side. Returns ``None`` rather
    than raising, because "the model did not produce JSON" is a verification
    outcome, not a crash.
    """
    if not text:
        return None

    candidates = [match.group(1) for match in _FENCE_RE.finditer(text)]
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start == -1 or end <= start:
                continue
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _plan_json(response: Any) -> dict[str, Any] | None:
    """The plan when a model answered in prose instead of calling ``todo_write``.

    The supervisor prompt no longer promises a bare-JSON reply, but a model that
    ignores its tools and answers with the plan object anyway is still worth
    reading — only an object that looks like a plan is accepted, so an unrelated
    JSON blob quoted in the model's prose cannot be mistaken for one.
    """
    parsed = extract_json_object(message_text(getattr(response, "content", "")))
    if parsed is None:
        return None
    plan_keys = ("todos", "acceptance_criteria", "verification_commands")
    if not any(key in parsed for key in plan_keys):
        return None
    return parsed


def _normalise_checks(raw: Any) -> list[VerificationCheck]:
    """Coerce the model's ``checks`` into :class:`VerificationCheck` rows."""
    checks: list[VerificationCheck] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return checks
    for item in raw:
        row = item if isinstance(item, Mapping) else {}
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        checks.append(
            {
                "name": name,
                "passed": bool(row.get("passed")),
                "detail": str(row.get("detail") or "").strip(),
            }
        )
    return checks


def _stream_writer() -> Any | None:
    """LangGraph's custom-stream writer, or ``None`` outside a running graph.

    Returning ``None`` rather than raising is what lets the same node function be
    called directly in a unit test and driven by a compiled graph in production.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001 - no run, no stream, no problem
        return None


def _emit(writer: Any | None, event: Mapping[str, Any], node: str) -> None:
    """Forward one agent event to the custom stream, tagged with its node.

    A broken consumer must not abort real work, hence the swallow.
    """
    if writer is None:
        return
    payload = dict(event)
    payload.setdefault("node", node)
    try:
        writer(payload)
    except Exception:  # noqa: BLE001 - a dead stream is not a reason to stop
        pass


def _with_retries(
    call: Callable[[], Any],
    *,
    node: str,
    retries: int,
    retry_backoff: float,
) -> Any:
    """Run ``call`` once, retrying only *transient* failures.

    The same policy the specialists' loops already apply, extended to the
    supervisor's own model calls. A retry is announced on the custom stream, so
    a stalled run looks stalled instead of hung.
    """
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - non-transient errors are re-raised
            if attempt >= retries or not is_transient_error(exc):
                raise
            delay = retry_backoff * (2**attempt)
            _emit(
                _stream_writer(),
                {
                    "type": "retry",
                    "loop": 0,
                    "attempt": attempt + 1,
                    "max_retries": retries,
                    "delay": delay,
                    "message": f"{type(exc).__name__}: {exc}",
                },
                node,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [text for item in raw if (text := str(item).strip())]


# --------------------------------------------------------------------------- #
# 1. delegation tools
# --------------------------------------------------------------------------- #
def _needs_revision(state: Mapping[str, Any]) -> bool:
    """A plan exists *and* a verification round has already failed against it.

    Keyed on ``attempts`` rather than on ``last_error`` being set, because
    ``last_error`` stays populated until a passing verdict clears it, so
    "is there a reason recorded?" would answer yes on every pass. The count of
    finished verify rounds is the honest version of "have we been here before".
    """
    if not state.get("todos"):
        return False
    return int(state.get("attempts") or 0) > 0 and not state.get("passed")


def _merge_sources(existing: Any, incoming: Any) -> tuple[list[dict[str, Any]], bool]:
    """Merge source rows by URL, first seen wins; returns ``(rows, changed)``."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in existing or []:
        if isinstance(source, Mapping):
            row = dict(source)
            url = str(row.get("url") or "")
            if url and url not in seen:
                seen.add(url)
                rows.append(row)
    changed = False
    for source in incoming or []:
        if not isinstance(source, Mapping):
            continue
        url = str(source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        rows.append(
            {
                "title": str(source.get("title") or url).strip(),
                "url": url,
                "score": source.get("score"),
            }
        )
        changed = True
    return rows, changed


def _call_search_agent_tool(
    state: Mapping[str, Any],
    writer: Any | None,
    instruction: str,
    *,
    model: Any | None = None,
) -> dict[str, Any]:
    """Hand one research instruction to the search agent; return a state fragment.

    Emits the ``handoff`` event first — a front end should see the delegation
    *before* the specialist's own events start arriving — then runs the
    specialist and shapes what came back as the channels it proposes to update:
    ``research_notes`` (this call's block; the node concatenates blocks),
    ``sources`` (this call's rows; the node deduplicates by URL), and the
    ``agent_handoffs`` record. The specialist's ``ok``/``error`` travel in the
    fragment's ``handoff.result`` being empty plus the report text, not as an
    exception: a failed search is an outcome the planner reads, not a crash.
    """
    instruction = str(instruction or "").strip()
    _emit(
        writer,
        {"type": "handoff", "from": "planner", "to": "searchAgent", "instruction": instruction},
        "planner",
    )
    result = run_search_agent(state, instruction, writer=writer, model=model)
    notes = str(result.get("summary") or "").strip()
    handoff: AgentHandoff = {
        "from_agent": "planner",
        "to_agent": "searchAgent",
        "instruction": instruction,
        "result": notes,
    }
    return {
        "ok": bool(result.get("ok")),
        "error": str(result.get("error") or ""),
        "research_notes": notes,
        "sources": [row for row in (result.get("sources") or []) if isinstance(row, Mapping)],
        "agent_handoffs": [dict(handoff)],
    }


def _call_code_agent_tool(
    state: Mapping[str, Any],
    writer: Any | None,
    instruction: str,
    *,
    model: Any | None = None,
    max_loops: int = DEFAULT_CODE_AGENT_LOOPS,
) -> dict[str, Any]:
    """Hand one implementation instruction to the code agent; return a state fragment.

    Same shape as :func:`_call_search_agent_tool`: the ``handoff`` event first,
    then the specialist run, then the proposed channel updates — ``todos`` (the
    plan as the specialist left it), ``code_agent_summary``, the
    ``agent_handoffs`` record, and the specialist's own turns for the
    ``messages`` channel (whose reducer appends them to the transcript).
    """
    instruction = str(instruction or "").strip()
    _emit(
        writer,
        {"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": instruction},
        "planner",
    )
    result = run_code_agent(
        state,
        instruction,
        writer=writer,
        model=model,
        max_loops=max_loops,
    )
    summary = str(result.get("summary") or "").strip()
    handoff: AgentHandoff = {
        "from_agent": "planner",
        "to_agent": "codeAgent",
        "instruction": instruction,
        "result": summary,
    }
    return {
        "ok": bool(result.get("ok")),
        "error": str(result.get("error") or ""),
        "todos": [row for row in (result.get("todos") or []) if isinstance(row, Mapping)],
        "code_agent_summary": summary,
        "agent_handoffs": [dict(handoff)],
        "messages": list(result.get("messages") or []),
    }


class CallSearchAgentInput(BaseModel):
    """Arguments for :class:`CallSearchAgentTool`."""

    instruction: str = Field(
        description=(
            "One focused research request for searchAgent, e.g. 'which maintained "
            "Python libraries parse ISO 8601 durations, and under what licences?'."
        )
    )


class CallSearchAgentTool(SuuClawTool):
    """The planner's delegation to ``searchAgent``.

    Unlike the workspace tools, this one operates on the *graph* state — it
    reads the task and prior research from it — so it takes the whole state
    mapping plus the stream writer, and appends its state-update fragment to
    ``updates`` for the supervisor node to fold in when the loop ends.
    """

    name = "call_search_agent"
    description = (
        "Delegate web/document research to searchAgent. It searches the public web "
        "and returns a research summary with source URLs; use it for current facts "
        "the workspace cannot answer. It cannot write files."
    )
    args_schema = CallSearchAgentInput

    def __init__(
        self,
        graph_state: Mapping[str, Any],
        writer: Any | None,
        *,
        model: Any | None = None,
        updates: list[dict[str, Any]],
    ) -> None:
        super().__init__(graph_state["runtime"])  # type: ignore[arg-type]
        self.graph_state = graph_state
        self.writer = writer
        self.model = model
        self.updates = updates

    def run(  # type: ignore[override]
        self,
        instruction: str,
    ) -> str:
        fragment = _call_search_agent_tool(
            self.graph_state, self.writer, instruction, model=self.model
        )
        self.updates.append(fragment)
        return _handoff_report("searchAgent", fragment)


class CallCodeAgentInput(BaseModel):
    """Arguments for :class:`CallCodeAgentTool`."""

    instruction: str = Field(
        description=(
            "One focused implementation request for codeAgent, naming the files to "
            "create or change and the checks to run, e.g. 'write greet.py with a "
            "greet(name) function, then run it once'."
        )
    )


class CallCodeAgentTool(SuuClawTool):
    """The planner's delegation to ``codeAgent``. See :class:`CallSearchAgentTool`."""

    name = "call_code_agent"
    description = (
        "Delegate file/code implementation to codeAgent. It edits the workspace "
        "(file and shell tools), reports todo progress, and returns a summary of "
        "files changed and checks run. It cannot search the web."
    )
    args_schema = CallCodeAgentInput

    def __init__(
        self,
        graph_state: Mapping[str, Any],
        writer: Any | None,
        *,
        model: Any | None = None,
        max_loops: int = DEFAULT_CODE_AGENT_LOOPS,
        updates: list[dict[str, Any]],
    ) -> None:
        super().__init__(graph_state["runtime"])  # type: ignore[arg-type]
        self.graph_state = graph_state
        self.writer = writer
        self.model = model
        self.max_loops = max_loops
        self.updates = updates

    def run(  # type: ignore[override]
        self,
        instruction: str,
    ) -> str:
        fragment = _call_code_agent_tool(
            self.graph_state,
            self.writer,
            instruction,
            model=self.model,
            max_loops=self.max_loops,
        )
        self.updates.append(fragment)
        return _handoff_report("codeAgent", fragment)


def _handoff_report(to_agent: str, fragment: Mapping[str, Any]) -> str:
    """The model-facing receipt for one delegation: outcome, summary, sources."""
    ok = bool(fragment.get("ok"))
    error = str(fragment.get("error") or "").strip()
    summary = str(
        fragment.get("code_agent_summary") or fragment.get("research_notes") or ""
    ).strip()

    lines = [f"{to_agent} {'finished' if ok else 'FAILED'}" + (f": {error}" if error else ".")]
    if summary:
        lines.append(summary)
    sources = fragment.get("sources") or []
    if sources:
        lines.append("Sources:")
        for row in sources[:8]:
            lines.append(f"- {row.get('title', '')} — {row.get('url', '')}")
        if len(sources) > 8:
            lines.append(f"(+{len(sources) - 8} more)")
    if not ok and not error:
        lines.append("(the specialist reported failure without a reason)")
    return _clip_report("\n".join(lines))


def _clip_report(text: str, limit: int = HANDOFF_REPORT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (+{len(text) - limit} more character(s) withheld)"


def make_search_agent_tool(
    graph_state: Mapping[str, Any],
    writer: Any | None,
    *,
    model: Any | None = None,
    updates: list[dict[str, Any]],
) -> CallSearchAgentTool:
    """A :class:`CallSearchAgentTool` bound to one graph state (test/工厂 convenience)."""
    return CallSearchAgentTool(graph_state, writer, model=model, updates=updates)


def make_code_agent_tool(
    graph_state: Mapping[str, Any],
    writer: Any | None,
    *,
    model: Any | None = None,
    max_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    updates: list[dict[str, Any]],
) -> CallCodeAgentTool:
    """A :class:`CallCodeAgentTool` bound to one graph state (test/工厂 convenience)."""
    return CallCodeAgentTool(
        graph_state,
        writer,
        model=model,
        max_loops=max_loops,
        updates=updates,
    )


# --------------------------------------------------------------------------- #
# 2. planner / supervisor
# --------------------------------------------------------------------------- #
def _merge_fragments(
    state: Mapping[str, Any],
    fragments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fold the delegation fragments into one partial state update.

    The tools cannot write graph state, so they propose; this disposes. The
    merge rules follow each channel's meaning: research blocks concatenate,
    sources deduplicate by URL, handoffs append, ``todos`` and
    ``code_agent_summary`` go to the latest delegation, ``messages`` extends
    (the channel's reducer does the appending once the node returns).
    """
    merged: dict[str, Any] = {}
    notes_blocks: list[str] = []
    sources, sources_changed = _merge_sources(state.get("sources"), [])
    handoffs: list[dict[str, Any]] = [dict(row) for row in (state.get("agent_handoffs") or [])]
    handoffs_changed = False

    for fragment in fragments:
        notes = str(fragment.get("research_notes") or "").strip()
        if notes:
            notes_blocks.append(notes)
        rows, changed = _merge_sources([], fragment.get("sources"))
        if rows:
            sources, _ = _merge_sources(sources, rows)
            sources_changed = sources_changed or changed or bool(rows)
        for row in fragment.get("agent_handoffs") or []:
            handoffs.append(dict(row))
            handoffs_changed = True
        if "todos" in fragment:
            merged["todos"] = fragment["todos"]
        if "code_agent_summary" in fragment:
            merged["code_agent_summary"] = fragment["code_agent_summary"]
        if fragment.get("messages"):
            merged.setdefault("messages", []).extend(fragment["messages"])

    if notes_blocks:
        merged["research_notes"] = "\n\n---\n\n".join(notes_blocks)
    if sources_changed:
        merged["sources"] = sources
    if handoffs_changed:
        merged["agent_handoffs"] = handoffs
    return merged


def _run_todo_write(
    tool: TodoWriteTool,
    args: Mapping[str, Any],
) -> str:
    """Execute one ``todo_write`` call; returns the model-facing text."""
    try:
        return tool.run(
            plan_summary=str(args.get("plan_summary") or ""),
            todos=list(args.get("todos") or []),
            acceptance_criteria=list(args.get("acceptance_criteria") or []),
            verification_commands=list(args.get("verification_commands") or []),
        )
    except Exception as exc:  # noqa: BLE001 - a bad plan payload is model-fixable data
        return f"{type(exc).__name__}: {exc}"


def _planner_input(state: Mapping[str, Any], memory: Mapping[str, Any]) -> str:
    """The supervisor's brief with the layered-memory snapshot appended.

    First pass and revision keep their own block shapes (the revision's
    evidence-first layout is the point of having two builders); memory rides
    last on both, so the model reads the situation before the rules about it.
    """
    base = revision_plan_request(state) if _needs_revision(state) else initial_plan_request(state)
    return append_memory_block(base, memory)


def _supervise(
    state: Mapping[str, Any],
    *,
    model: Any,
    planner_loops: int = DEFAULT_PLANNER_LOOPS,
    code_agent_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> dict[str, Any]:
    """The supervisor loop itself. See :func:`planner_node` for the contract."""
    runtime = _runtime(state)
    store = TodoStore.from_items(state.get("todos") or [])
    writer = _stream_writer()

    updates: list[dict[str, Any]] = []
    todo_tool = TodoWriteTool(runtime, store)
    search_tool = CallSearchAgentTool(state, writer, model=model, updates=updates)
    code_tool = CallCodeAgentTool(
        state,
        writer,
        model=model,
        max_loops=code_agent_loops,
        updates=updates,
    )
    bound = as_langchain_tools([todo_tool, search_tool, code_tool])
    by_name = {tool.name: tool for tool in bound}
    agent = model.bind_tools(bound)

    revising = _needs_revision(state)
    system = PLANNER_PROMPT + ("\n" + PLANNER_REVISION_ADDENDUM if revising else "")
    memory = build_layered_memory(state, node="planner")
    _emit(writer, memory_event(memory, node="planner"), "planner")
    request = _planner_input(state, memory)
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=request)]

    plan_args: dict[str, Any] | None = None
    produced: list[Any] = []
    problem = ""
    for loop in range(1, max(1, planner_loops) + 1):
        response = _with_retries(
            lambda: agent.invoke(messages),  # noqa: B023 - messages is read at call time
            node="planner",
            retries=retries,
            retry_backoff=retry_backoff,
        )
        messages.append(response)
        produced.append(response)

        calls = list(getattr(response, "tool_calls", None) or [])
        invalid = list(getattr(response, "invalid_tool_calls", None) or [])
        if not calls:
            if invalid:
                preview = [str(i)[:200] for i in invalid]
                problem = f"model produced malformed tool call(s): {preview}"
            else:
                # A clean close. If the model answered with the plan object as
                # prose anyway, take it — same fallback the stage-2 planner had.
                prose = _plan_json(response)
                if prose:
                    plan_args = prose
            break

        for position, call in enumerate(calls, start=1):
            name = str(call.get("name") or "")
            raw_args = call.get("args")
            call_id = str(call.get("id") or f"planner-{loop}-{position}")
            args = dict(raw_args) if isinstance(raw_args, Mapping) else {}

            tool = by_name.get(name)
            if tool is None:
                known = ", ".join(sorted(by_name)) or "(none)"
                output = f"Error: unknown tool {name!r}. Available tools: {known}"
            else:
                if name == "todo_write":
                    plan_args = args  # captured even if the store write below fails
                try:
                    output = str(tool.invoke(args))
                except Exception as exc:  # noqa: BLE001 - a bad payload is fixable data
                    output = f"{type(exc).__name__}: {exc}"

            reply = ToolMessage(content=output, tool_call_id=call_id)
            messages.append(reply)
            produced.append(reply)

    # Fold the plan fields captured off the todo_write call(s) — last write wins,
    # so a revision replaces the whole plan rather than merging with it.
    update: dict[str, Any] = {}
    if plan_args is not None:
        if not store.items:
            store.replace(plan_args.get("todos") or [])
        update.update(
            {
                "plan_summary": str(plan_args.get("plan_summary") or "").strip(),
                "acceptance_criteria": _string_list(plan_args.get("acceptance_criteria")),
                "verification_commands": _string_list(plan_args.get("verification_commands")),
            }
        )

    update.update(_merge_fragments(state, updates))

    # The planner always hands off to the verifier. Written on the degraded path
    # too (it merges ``update``), so a supervisor that produced nothing usable
    # still leaves the monitor a destination to route on rather than a guess.
    update["context_next_node"] = "verifier"

    if not store.items:
        reason = problem or (
            "the supervisor never published a plan"
            if not plan_args
            else "the supervisor's plan had no usable todos (every step was empty)"
        )
        degraded = _degraded_plan(state, reason)
        degraded.update(update)
        if produced:
            degraded["messages"] = produced
        return degraded

    if problem:
        update["last_error"] = problem
    update["todos"] = store.snapshot()
    if produced:
        update["messages"] = produced
    return update


def _degraded_plan(state: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """A one-step plan standing in for a supervisor that produced nothing usable.

    Deliberately not an exception: a run that proceeds with a poor plan the user
    can watch fail is more useful than a traceback, and the reason is preserved
    in ``last_error`` where the next planner pass will find it.
    """
    task = str(state.get("task") or "").strip() or "complete the requested task"
    rows = TodoStore().replace([{"id": "1", "content": task}])
    return {
        "plan_summary": f"(unstructured) {reason}",
        "todos": rows,
        "acceptance_criteria": [],
        "verification_commands": [],
        "last_error": reason,
    }


def planner_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Coordinate the specialists: plan, delegate, fold, report — from runtime metadata."""
    return _supervise(state, model=_default_model(state))


def make_planner_node(
    model: Any,
    *,
    planner_loops: int = DEFAULT_PLANNER_LOOPS,
    code_agent_loops: int = DEFAULT_CODE_AGENT_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Any:
    """Supervisor bound to one pre-built model (the form a compiled graph wants).

    The same model instance is handed to every delegated specialist, so the
    graph talks to exactly one chat model end to end.
    """

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _supervise(
            state,
            model=model,
            planner_loops=planner_loops,
            code_agent_loops=code_agent_loops,
            retries=retries,
            retry_backoff=retry_backoff,
        )

    node.__name__ = "planner_node"
    return node


# --------------------------------------------------------------------------- #
# 3. verifier
# --------------------------------------------------------------------------- #
def _run_verification_commands(
    state: Mapping[str, Any],
    runtime: RuntimeState,
) -> list[VerificationResult]:
    """Run every ``verification_commands`` entry and record the real outcome.

    Deterministic and model-independent by construction: this is the part of
    verification that cannot be talked out of its answer.
    """
    results: list[VerificationResult] = []
    for raw in state.get("verification_commands") or []:
        command = str(raw).strip()
        if not command:
            continue
        outcome = run_command(
            command,
            workspace=runtime.workspace,
            shell=runtime.shell,
            env=runtime.env,
            timeout=runtime.command_timeout,
        )
        results.append(
            {
                "command": command,
                "ok": outcome.ok,
                "exit_code": outcome.exit_code,
                # Truncated on the way in, not on the way out: this value is
                # stored in graph state and replayed into prompts.
                "stdout": runtime.truncate(
                    outcome.stdout, limit=VERIFICATION_OUTPUT_CHARS, note="stdout truncated"
                ),
                "stderr": runtime.truncate(
                    outcome.stderr, limit=VERIFICATION_OUTPUT_CHARS, note="stderr truncated"
                ),
            }
        )
    return results


def _verifier_input(
    state: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
) -> str:
    """The verifier's brief with the layered-memory snapshot appended.

    Facts first, memory last: the command evidence and the worker's claim are
    what the verdict turns on, so the snapshot may not push them down the
    message.
    """
    return append_memory_block(verifier_request(state, results), memory)


def _judge(
    state: Mapping[str, Any],
    *,
    model: Any,
    results: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    max_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> tuple[dict[str, Any], str]:
    """Ask the read-only verifier agent to judge; returns ``(report, problem)``."""
    runtime = _runtime(state)
    tools = build_read_only_tools(runtime)
    writer = _stream_writer()
    summary = ""

    for event in stream_react_events(
        _verifier_input(state, results, memory),
        workspace=runtime.workspace,
        max_loops=max_loops,
        model=model,
        state=runtime,
        tools=tools,
        retries=retries,
        retry_backoff=retry_backoff,
        system_prompt=VERIFIER_PROMPT,
    ):
        _emit(writer, event, "verifier")
        if event.get("type") == "final_answer":
            summary = str(event.get("content") or "").strip()

    report = extract_json_object(summary)
    if report is None:
        preview = summary[:300] or "(the verifier produced no output)"
        return {}, f"the verifier did not return JSON: {preview}"
    return report, ""


def _failing_commands(results: Sequence[Mapping[str, Any]]) -> str:
    """Names of the commands that did not exit 0, for the failure reason."""
    failing = [str(row.get("command") or "") for row in results if not row.get("ok")]
    return "; ".join(name for name in failing if name) or "(unnamed)"


def _verify(
    state: Mapping[str, Any],
    *,
    model: Any,
    verifier_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> dict[str, Any]:
    runtime = _runtime(state)
    attempts = int(state.get("attempts") or 0) + 1

    # Same brief the workers get: the assembled three-layer snapshot, announced
    # on the stream so a front end can show what the judge is judging against.
    memory = build_layered_memory(state, node="verifier")
    _emit(_stream_writer(), memory_event(memory, node="verifier"), "verifier")

    # Evidence before judgement: the model is handed what the commands actually
    # did, so it cannot conclude "fine" from the worker's prose alone.
    results = _run_verification_commands(state, runtime)
    report, problem = _judge(
        state,
        model=model,
        results=results,
        memory=memory,
        max_loops=verifier_loops,
        retries=retries,
        retry_backoff=retry_backoff,
    )

    checks = _normalise_checks(report.get("checks"))
    model_passed = bool(report.get("passed"))
    commands_ok = all(bool(row.get("ok")) for row in results)
    passed = model_passed and commands_ok

    reason = str(report.get("reason") or "").strip()
    if problem:
        reason = reason or problem
    if model_passed and not commands_ok:
        # The deterministic half of verification outranks the model's opinion.
        failing = _failing_commands(results)
        reason = f"acceptance criteria looked met, but verification command(s) failed: {failing}"
    if not reason:
        reason = "the verifier gave no reason"

    store = TodoStore.from_items(state.get("todos") or [])
    if passed:
        store.mark_all("completed")

    # The verifier's advice for the next delegation, cleared on a pass so a
    # stale instruction can never leak into a later revision brief.
    next_fix = "" if passed else str(report.get("recommended_next_instruction") or "").strip()

    # ``final_answer`` is deliberately NOT written here: the final node owns that
    # channel, and two writers for one user-facing string is how the two drift
    # apart. This node's job ends at the verdict — but the verdict *is* a routing
    # decision, and the context-aware graph routes through the monitor rather
    # than through ``verifier_route``, so the destination is handed over here
    # instead of being re-derived downstream.
    return {
        "passed": passed,
        "attempts": attempts,
        "verification_results": results,
        "verification_checks": checks,
        "last_error": "" if passed else reason,
        "recommended_next_instruction": next_fix,
        "context_next_node": _verdict_destination(state, passed=passed, attempts=attempts),
        "todos": store.snapshot(),
    }


def verifier_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Run the commands, judge the criteria, and decide whether to retry."""
    return _verify(state, model=_default_model(state))


def make_verifier_node(
    model: Any,
    *,
    max_loops: int = DEFAULT_VERIFIER_LOOPS,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Any:
    """Verifier bound to one pre-built model."""

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _verify(
            state,
            model=model,
            verifier_loops=max_loops,
            retries=retries,
            retry_backoff=retry_backoff,
        )

    node.__name__ = "verifier_node"
    return node


# --------------------------------------------------------------------------- #
# 4. final
# --------------------------------------------------------------------------- #
def _command_lines(results: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """One bullet per verification command: what ran, and what it exited with."""
    rows = [row for row in (results or []) if isinstance(row, Mapping)]
    if not rows:
        return ["- (no verification commands were run)"]
    lines = []
    for row in rows:
        exit_code = row.get("exit_code")
        if exit_code is None:
            shown = "no exit code (timeout or spawn failure)"
        else:
            shown = f"exit {exit_code}"
        verdict = "ok" if row.get("ok") else "FAILED"
        lines.append(f"- {verdict}: `{row.get('command', '')}` → {shown}")
    return lines


def _unfinished_lines(todos: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """The plan rows that are not ``completed`` — what a failed run left behind."""
    lines = []
    for row in todos or []:
        status = str(row.get("status") or "pending")
        if status == "completed":
            continue
        line = f"- [{status}] {row.get('id', '?')}: {row.get('content', '')}"
        note = str(row.get("note") or "").strip()
        if note:
            line += f"  (note: {note})"
        lines.append(line)
    return lines


def _worker_summary(state: Mapping[str, Any]) -> str:
    """The implementer's latest report: the stage-3 field, or the stage-2 one."""
    return (
        str(state.get("code_agent_summary") or "").strip()
        or str(state.get("last_actor_summary") or "").strip()
    )


def final_answer_text(state: Mapping[str, Any]) -> str:
    """Format the run's outcome into the user-facing ``final_answer``.

    Deterministic on purpose. A run that ended on its retry budget still owes the
    user a truthful report, and that report must not depend on one more model call
    succeeding — so this function only ever reads state.
    """
    attempts = int(state.get("attempts") or 0)
    noun = "attempt" if attempts == 1 else "attempts"
    results = state.get("verification_results")

    if state.get("passed"):
        body = (
            _worker_summary(state)
            or str(state.get("plan_summary") or "").strip()
            or "The task is complete."
        )
        blocks = [body, f"Verified after {attempts} {noun}."]
        checks = state.get("verification_checks")
        if checks:
            met = sum(1 for row in checks if row.get("passed"))
            blocks.append(f"Acceptance criteria met: {met}/{len(checks)}.")
        return "\n".join([*blocks, *_command_lines(results)])

    reason = str(state.get("last_error") or "").strip() or "the verifier recorded no reason"
    blocks = [
        f"Stopped after {attempts} {noun} without meeting the acceptance criteria.",
        f"Last failure: {reason}",
    ]
    unfinished = _unfinished_lines(state.get("todos"))
    if unfinished:
        blocks += ["", "Still unfinished:", *unfinished]
    return "\n".join([*blocks, "", "Verification:", *_command_lines(results)])


def _model_summary(state: Mapping[str, Any], model: Any) -> str:
    """A written report from ``model``, or ``""`` when it cannot produce one."""
    try:
        response = model.invoke(
            [SystemMessage(content=FINAL_PROMPT), HumanMessage(content=final_request(state))]
        )
    except Exception:  # noqa: BLE001 - the deterministic text is the fallback
        return ""
    return message_text(getattr(response, "content", "")).strip()


def _finalise(state: Mapping[str, Any], *, model: Any | None = None) -> dict[str, Any]:
    _emit(_stream_writer(), {"type": "node_start"}, "final")
    text = _model_summary(state, model) if model is not None else ""
    return {"final_answer": text or final_answer_text(state)}


def final_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Turn the verifier's verdict into ``final_answer``, without a model.

    This is the node the compiled graph uses, and it is why a run needs an API key
    only for the three working nodes — finishing the run is pure formatting.
    """
    return _finalise(state)


def make_final_node(model: Any | None = None) -> Any:
    """Final node that words its report with ``model``, formatted text otherwise.

    The deterministic formatter is not a degraded mode: it is what runs whenever
    the summary call is unavailable, comes back empty, or raises. Passing a model
    can therefore only improve the wording, never the truthfulness of the outcome.
    """

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _finalise(state, model=model)

    node.__name__ = "final_node"
    return node


# --------------------------------------------------------------------------- #
# 5. context monitor
# --------------------------------------------------------------------------- #
def _memory_payload(state: Mapping[str, Any]) -> str:
    """The assembled layered-memory snapshot as one prompt-shaped text block.

    The monitor counts what the *next* prompt would actually carry, and the
    next prompt always includes the snapshot — so it rides into the estimate
    as one more message rather than being forgotten until it blows the budget.
    """
    memory = build_layered_memory(state, node="context_monitor")
    return f"# Layered memory\n{format_layered_memory_for_prompt(memory)}"


def _estimate_token_count(state: Mapping[str, Any], *, model: Any) -> int:
    """The context's token cost: the model's own counter, else ``chars // 4``.

    The model tokenizer is the honest number, but it is the one part of this
    node that can fail — ``get_num_tokens_from_messages`` needs a tokenizer for
    the model id in play, and unknown or third-party ids raise. The fallback is
    deliberately crude: an estimate only has to separate "comfortably under
    the limit" from "time to compress", and being wrong by a few percent is
    invisible at any budget this node is given. Counted text is the raw
    transcript plus the memory payload — the two things every node's prompt is
    rebuilt from.
    """
    messages = list(state.get("messages") or [])
    payload = _memory_payload(state)
    if model is not None:
        try:
            counted = model.get_num_tokens_from_messages([*messages, HumanMessage(content=payload)])
            return max(int(counted), 0)
        except Exception:  # noqa: BLE001 - no tokenizer, no network, no problem
            pass
    texts = [message_text(getattr(message, "content", "")) for message in messages]
    texts.append(payload)
    return len("\n".join(texts)) // CHARS_PER_TOKEN


def _env_context_token_limit() -> int | None:
    """The budget the environment names, or ``None`` when it names nothing.

    Read on every visit rather than cached at import time, so a value exported
    in the shell — or written into ``.env``, which is loaded before any node
    runs — takes effect without rebuilding anything.

    A value that is not a *positive* integer is skipped in favour of the next
    name, and ultimately the default. That is the safe failure: a typo should
    cost a tester one confusing run, not mean "compress on every visit", which
    is what a stray ``0`` or ``-1`` would quietly mean.
    """
    for name in CONTEXT_TOKEN_LIMIT_ENVS:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _resolve_context_token_limit(state: Mapping[str, Any]) -> int:
    """This visit's budget: the state's channel, else the environment, else 400k.

    Three sources because they answer different questions. The state channel is
    a single run's deliberate setting (a caller seeding ``initial_state``); the
    environment is the machine's (a tester lowering it to see compression fire
    without having to build a huge transcript); the default is the product's.
    Most specific wins, and the resolved value is written back into the state
    channel by :func:`_monitor`, so a checkpoint records the budget the run
    actually used.
    """
    raw_limit = state.get("context_token_limit")
    if raw_limit is not None:
        try:
            value = int(raw_limit)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return _env_context_token_limit() or DEFAULT_CONTEXT_TOKEN_LIMIT


def _monitor(state: Mapping[str, Any], *, model: Any) -> dict[str, Any]:
    """The monitor itself. See :func:`context_monitor_node` for the contract."""
    _emit(_stream_writer(), {"type": "node_start"}, "context_monitor")
    token_count = _estimate_token_count(state, model=model)
    limit = _resolve_context_token_limit(state)

    return {
        "context_token_count": token_count,
        # Echoed rather than left implicit: it is the number the flag below was
        # measured against, and without it a reader of the final state cannot
        # tell an unarmed budget from a run that simply never got close.
        "context_token_limit": limit,
        "context_should_compress": token_count > limit,
        # Set by upstream (planner → "verifier", a failed verifier → "planner");
        # the monitor only carries it forward so the router reads one channel.
        "context_next_node": state.get("context_next_node") or "verifier",
    }


def context_monitor_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Estimate the context's token cost and flag a compression pass, if due.

    Deliberately writes only the ``context_*`` channels: the monitor is a
    measurement step, not a policy step. Whether compression actually happens
    is the router's decision (:func:`context_monitor_route`), and *what* the
    compressor keeps is the compressor's — keeping them apart is what lets the
    threshold be tuned without touching the node that measures it.

    Token counting tries the model's own
    ``get_num_tokens_from_messages(messages + [memory_payload])`` first and
    falls back to ``len(text) // 4`` over the same content when the tokenizer
    is unavailable (unknown model id, no network, a test double without the
    method).

    The budget comes from the first of these that is set (see
    :func:`_resolve_context_token_limit`):

    1. the ``context_token_limit`` channel — one run's deliberate setting;
    2. ``SUUCLAW_CONTEXT_TOKEN_LIMIT`` — the machine's, e.g.
       ``export SUUCLAW_CONTEXT_TOKEN_LIMIT=50000`` to watch
       compression fire on a modest transcript;
    3. :data:`DEFAULT_CONTEXT_TOKEN_LIMIT` — 400,000 tokens.

    The resolved value is written back to the ``context_token_limit`` channel,
    so the state says which budget the flag was measured against.
    """
    return _monitor(state, model=_default_model(state))


def make_context_monitor_node(model: Any) -> Any:
    """Context monitor bound to one pre-built model (for token counting only).

    A model without ``get_num_tokens_from_messages`` — or one whose tokenizer
    raises — is fine: the node degrades to the character-based estimate rather
    than failing a run over a measurement.
    """

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _monitor(state, model=model)

    node.__name__ = "context_monitor_node"
    return node


# --------------------------------------------------------------------------- #
# 6. context compressor
# --------------------------------------------------------------------------- #
def _transcript_for_compression(state: Mapping[str, Any]) -> str:
    """The raw transcript as numbered ``role: text`` lines, each capped.

    Numbering matters: it lets the model refer to an earlier message ("see
    [7]") in its digest, and lets a reader of ``HISTORY_SUMMARY.md`` tell how
    much was actually folded away. Per-message capping keeps one runaway
    command output from eating the brief.
    """
    lines = []
    for position, message in enumerate(state.get("messages") or [], start=1):
        role = type(message).__name__
        text = _short_text(message_text(getattr(message, "content", "")), TRANSCRIPT_MESSAGE_CHARS)
        lines.append(f"[{position}] {role}: {text}")
    return "\n".join(lines) or "(the transcript is empty)"


def _compression_request(state: Mapping[str, Any]) -> str:
    """The compressor's brief: the transcript, then the layered-memory snapshot.

    The snapshot rides along because the model should not re-derive what the
    runtime already assembled — the plan, the todos, the handoffs are all
    there, and the digest is supposed to *carry them forward*, not rediscover
    them from the transcript.
    """
    memory = build_layered_memory(state, node="context_compressor")
    return "\n\n".join(
        [
            "# Current transcript\n" + _transcript_for_compression(state),
            "# Layered memory snapshot\n" + format_layered_memory_for_prompt(memory),
        ]
    )


_DIGEST_LABELS: tuple[tuple[str, str], ...] = (
    ("summary", "Summary"),
    ("active_goal", "Active goal"),
    ("completed_work", "Completed work"),
    ("open_todos", "Open todos"),
    ("important_files", "Important files"),
    ("tool_findings", "Tool findings"),
    ("sources", "Sources"),
    ("next_steps", "Next steps"),
    ("risks", "Risks"),
)


def _digest_from_report(report: Mapping[str, Any]) -> str:
    """Render the model's JSON verdict into the digest text, field by field.

    Lists become bullets (capped per item and in count), scalars are truncated
    once. Empty keys are skipped rather than printed as empty sections — the
    digest is what the next prompt opens with, and noise has a real token cost.
    """
    lines: list[str] = []
    for key, label in _DIGEST_LABELS:
        value = report.get(key)
        if isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
            if not items:
                continue
            body = "\n".join(f"- {_short_text(item, 300)}" for item in items[:20])
        else:
            body = _short_text(value, 1500)
            if not body.strip():
                continue
        lines.append(f"## {label}\n{body}")
    return "\n\n".join(lines).strip()


def _deterministic_digest(state: Mapping[str, Any]) -> str:
    """The digest built from state alone, for when the model could not help.

    Same contract as the planner's degraded plan: compression must not be able
    to crash a run. This keeps the fields that live in channels anyway — so it
    loses nothing the state would not still carry, it just makes a thinner
    summary than the model would have.
    """
    sections: list[tuple[str, Any]] = [
        ("Summary", f"Task: {state.get('task', '')}"),
        ("Active goal", state.get("plan_summary", "")),
        ("Open todos", [row for row in (state.get("todos") or []) if isinstance(row, Mapping)]),
        ("Completed work", state.get("code_agent_summary", "")),
        ("Tool findings", state.get("research_notes", "")),
        ("Risks", state.get("last_error", "")),
    ]
    lines: list[str] = []
    for label, value in sections:
        if isinstance(value, list):
            rows = []
            for row in value:
                if str(row.get("status", "")) == "completed":
                    continue
                text = f"{row.get('id', '?')}: {row.get('content', '')}"
                rows.append(f"- {_short_text(text, 300)}")
            body = "\n".join(rows[:20])
        else:
            body = _short_text(value, 1500)
        if body.strip():
            lines.append(f"## {label}\n{body}")
    return "\n\n".join(lines).strip() or "(nothing to carry forward)"


def _compress(
    state: Mapping[str, Any],
    *,
    model: Any,
) -> dict[str, Any]:
    """The compressor itself. See :func:`context_compressor_node` for the contract."""
    _emit(_stream_writer(), {"type": "node_start"}, "context_compressor")
    runtime = _runtime(state)
    tokens_before = int(state.get("context_token_count") or 0)

    report: Mapping[str, Any] | None = None
    problem = ""
    try:
        response = _with_retries(
            lambda: model.invoke(
                [
                    SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
                    HumanMessage(content=_compression_request(state)),
                ]
            ),
            node="context_compressor",
            retries=DEFAULT_RETRIES,
            retry_backoff=DEFAULT_RETRY_BACKOFF,
        )
        report = extract_json_object(message_text(getattr(response, "content", "")))
        if report is None:
            problem = "the compressor did not return JSON"
    except Exception as exc:  # noqa: BLE001 - compression degrades, never crashes
        problem = f"{type(exc).__name__}: {exc}"

    digest = _digest_from_report(report) if report is not None else _deterministic_digest(state)
    summary = _short_text(digest, DIGEST_CHARS)

    tokens_after = _estimate_token_count(
        {**state, "messages": [AIMessage(content=summary)]}, model=model
    )

    event: CompressionEvent = {
        "at": datetime.now(timezone.utc).isoformat(),
        "trigger": "token_limit",
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "summary_chars": len(summary),
        "note": problem,
    }
    events = [dict(row) for row in (state.get("compression_events") or [])]
    events.append(dict(event))

    # Best-effort on purpose: the messages have already been replaced, so a
    # file-write failure is recorded in the event rather than allowed to
    # un-wind a state update the graph has half-applied.
    try:
        runtime.resolve(HISTORY_SUMMARY_FILENAME).write_text(f"{summary}\n", encoding="utf-8")
    except OSError as exc:
        event["note"] = event["note"] or f"HISTORY_SUMMARY.md write failed: {exc}"
        events[-1] = dict(event)

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage(content=summary),
        ],
        "context_summary": _short_text(summary, CONTEXT_SUMMARY_CHARS),
        "context_token_count": tokens_after,
        "context_should_compress": False,
        "research_notes": _short_text(state.get("research_notes", ""), RESEARCH_NOTES_CHARS),
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""), CODE_AGENT_SUMMARY_CHARS
        ),
        "history_summary": summary,
        "compression_events": events,
    }


def context_compressor_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Fold the transcript into a digest, replace the window, persist the history.

    Three things happen in one superstep, in this order:

    1. **Digest** — the model reads the transcript plus the layered-memory
       snapshot under :data:`~suuclaw.prompts.stage4.CONTEXT_COMPRESSION_PROMPT`
       and answers with a nine-key JSON object, which is rendered into a
       sectioned digest. A model failure or a non-JSON answer degrades to a
       deterministic digest built from the state channels (same philosophy as
       the planner's degraded plan: compression must not crash a run).
    2. **Replace** — ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` clears the
       transcript through the ``messages`` reducer, then one ``AIMessage`` with
       the digest becomes the new context origin.
    3. **Persist** — the digest is written to ``HISTORY_SUMMARY.md`` so the
       compressed history survives even a cold restart, and one
       :class:`~suuclaw.graph.memory.CompressionEvent` joins the audit trail.

    The truncated channels it returns (``research_notes``, ``agent_handoffs``,
    ``code_agent_summary``) are the same views the memory snapshot shows —
    compression tightens the *state* to match the tighter window, so a later
    snapshot cannot quietly re-inflate the context the compressor just shrank.
    """
    return _compress(state, model=_default_model(state))


def make_context_compressor_node(model: Any) -> Any:
    """Compressor bound to one pre-built model (the form a compiled graph wants)."""

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _compress(state, model=model)

    node.__name__ = "context_compressor_node"
    return node


# --------------------------------------------------------------------------- #
# 7. entry routing: intent router + chat responder
# --------------------------------------------------------------------------- #
#: Below this confidence a "chat" verdict is overruled back to ``workflow``.
#: The asymmetry is deliberate: sending real work down the chat branch loses it
#: silently, while sending small talk down the workflow branch only costs
#: tokens — so doubt resolves toward the heavier path.
DEFAULT_INTENT_CONFIDENCE = 0.55

#: How much of the session's compressed history rides into the intent/chat
#: prompts. Both nodes only need to know *what the conversation has been about*;
#: they are the cheap branches and must not re-read the workspace.
SESSION_CONTEXT_CHARS = 2000


def _session_context(state: Mapping[str, Any]) -> str:
    """The compressed conversation so far, or a placeholder on a cold start.

    ``history_summary`` is preferred because the compressor refreshes it on
    every pass; ``context_summary`` is the older stage-4 spelling, kept as a
    fallback so old state replays.
    """
    text = str(state.get("history_summary") or state.get("context_summary") or "").strip()
    return _short_text(text, SESSION_CONTEXT_CHARS) if text else "(no prior session context)"


def _entry_request(state: Mapping[str, Any]) -> str:
    """The brief both entry nodes read: session context first, input last."""
    return "\n\n".join(
        [
            "# Session context\n" + _session_context(state),
            "# Latest user input\n" + (str(state.get("task") or "").strip() or "(empty)"),
        ]
    )


def _classify_intent(state: Mapping[str, Any], *, model: Any) -> dict[str, Any]:
    """One LLM call, one JSON verdict, one guaranteed route.

    Returns the ``intent_*`` channels. Every failure shape — the call raised,
    the answer was not JSON, the route word was unrecognised, the confidence
    did not parse — lands on ``workflow`` with a reason that says which shape
    it was, so the entry graph degrades to the main workflow instead of
    guessing "chat" from noise.
    """
    _emit(_stream_writer(), {"type": "node_start"}, "intent_router")
    defaults = {
        "intent_route": "workflow",
        "intent_reason": "",
        "intent_confidence": 0.0,
    }
    try:
        response = _with_retries(
            lambda: model.invoke(
                [
                    SystemMessage(content=INTENT_ROUTER_PROMPT),
                    HumanMessage(content=_entry_request(state)),
                ]
            ),
            node="intent_router",
            # ChatOpenAI already owns transport retries. Repeating them here
            # multiplies a 120s request timeout into a many-minute silent gate.
            retries=0,
            retry_backoff=DEFAULT_RETRY_BACKOFF,
        )
        report = extract_json_object(message_text(getattr(response, "content", "")))
    except Exception as exc:  # noqa: BLE001 - a failed router defaults, never crashes
        defaults["intent_reason"] = f"intent router failed: {type(exc).__name__}: {exc}"
        return defaults

    if report is None:
        defaults["intent_reason"] = "the intent router did not return JSON"
        return defaults

    route = str(report.get("route") or "").strip().lower()
    try:
        confidence = min(max(float(report.get("confidence")), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(report.get("reason") or "").strip()

    if route == "chat" and confidence >= DEFAULT_INTENT_CONFIDENCE:
        return {"intent_route": "chat", "intent_reason": reason, "intent_confidence": confidence}
    if route == "chat":
        defaults["intent_reason"] = (
            reason or "chat verdict overruled"
        ) + f" — confidence {confidence:.2f} below {DEFAULT_INTENT_CONFIDENCE:.2f}"
    elif route == "workflow":
        defaults["intent_route"] = "workflow"
        defaults["intent_reason"] = reason
        defaults["intent_confidence"] = confidence
    else:
        defaults["intent_reason"] = f"unrecognised route {route!r}; defaulting to workflow"
    return defaults


def intent_router_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Decide whether the latest input is chat or workflow, from runtime metadata.

    Reads ``task`` plus the session's compressed history, writes the three
    ``intent_*`` channels, touches nothing else. The route itself is applied
    by :func:`intent_route_fn`; this node only records the decision.
    """
    return _classify_intent(state, model=_default_model(state))


def make_intent_router_node(model: Any) -> Any:
    """Intent router bound to one pre-built model (the form a compiled graph wants)."""

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _classify_intent(state, model=model)

    node.__name__ = "intent_router_node"
    return node


def _chat_respond(state: Mapping[str, Any], *, model: Any) -> dict[str, Any]:
    """The chat branch itself. See :func:`chat_responder_node` for the contract."""
    _emit(_stream_writer(), {"type": "node_start"}, "chat_responder")
    try:
        response = _with_retries(
            lambda: model.invoke(
                [
                    SystemMessage(content=CHAT_RESPONDER_PROMPT),
                    HumanMessage(content=_entry_request(state)),
                ]
            ),
            node="chat_responder",
            retries=0,
            retry_backoff=DEFAULT_RETRY_BACKOFF,
        )
        reply = message_text(getattr(response, "content", "")).strip()
        problem = ""
    except Exception as exc:  # noqa: BLE001 - chat degrades to a truthful fallback
        reply = ""
        problem = f"{type(exc).__name__}: {exc}"

    if not reply:
        reply = (
            "I could not answer that in chat mode"
            + (f" ({problem})" if problem else "")
            + ". If this needs workspace access or tooling, please send it as a "
            "task so it runs through the workflow route."
        )
    # The chat path never reaches the main workflow's ``final`` node, so this
    # node owns ``final_answer`` *for chat runs* — the two writers are mutually
    # exclusive by topology, which is what keeps the one-writer rule honest.
    return {"chat_response": reply, "final_answer": reply}


def chat_responder_node(state: SuuClawGraphState) -> dict[str, Any]:
    """Answer conversational input directly, without tools, from runtime metadata."""
    return _chat_respond(state, model=_default_model(state))


def make_chat_responder_node(model: Any) -> Any:
    """Chat responder bound to one pre-built model (the form a compiled graph wants)."""

    def node(state: SuuClawGraphState) -> dict[str, Any]:
        return _chat_respond(state, model=model)

    node.__name__ = "chat_responder_node"
    return node


def intent_route_fn(state: SuuClawGraphState) -> str:
    """The entry graph's router: chat answers here, workflow hands off.

    Note:
        ``"planner"`` is a *destination*, not a node of this graph — the
        compiled entry graph binds it to ``END``, and the caller (the driver,
        not the graph) picks the main workflow up from there.
    """
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def verifier_route(state: SuuClawGraphState) -> Literal["final", "planner"]:
    """Where to go after verification: finish, or replan and try again.

    A thin reader of :func:`_verdict_destination`, which the verifier node also
    calls to hand the same answer to the monitor. The simple graph asks the
    question here; the context-aware one gets it pre-answered and this function
    is not on its path at all.
    """
    return _verdict_destination(
        state,
        passed=bool(state.get("passed")),
        attempts=int(state.get("attempts") or 0),
    )


def context_monitor_route(state: SuuClawGraphState) -> str:
    """Where to go after the context check: finish, compress, or carry on.

    The order is the policy. ``passed`` wins first — a finished run must not
    detour through housekeeping on its way out. A spent attempt budget comes
    next: when upstream nominated ``"planner"`` and the budget is already gone,
    the run is over, and going around once more would spend a whole
    planner + verifier round to reach ``"final"`` anyway. Compression is checked
    third, because compressing *before* the next working node's prompt is built
    is the whole point of measuring early — and compressing a run that is about
    to end is tokens spent on a digest nobody will read. Failing all three,
    continue to the node upstream nominated in ``context_next_node``:
    ``"verifier"`` after the planner, ``"planner"`` after a failed verifier.

    The spent-budget check is belt and braces — the verifier already writes
    ``"final"`` into ``context_next_node`` when it burns the last attempt, so
    this only fires for a run seeded or resumed with a stale ``"planner"``. It
    is here because an unbounded loop is the one failure a budget check exists
    to prevent, and the router should not have to trust its callers for that.

    Note:
        The returned names are destinations, not nodes — the compiled graph
        has to bind them via ``add_conditional_edges``.
    """
    if state.get("passed"):
        return "final"
    destination = state.get("context_next_node") or "verifier"
    exhausted = int(state.get("attempts") or 0) >= _max_attempts(state)
    if destination == "planner" and exhausted:
        return "final"
    if state.get("context_should_compress"):
        return "context_compressor"
    return destination


def context_compressor_route(state: SuuClawGraphState) -> str:
    """Where to go after compression: back to the interrupted work.

    Compression is a detour, not a decision: the monitor already decided the run
    should continue (it only routes here when the run is neither finished nor
    done), so the compressor's only job is to hand control back to whatever node
    the upstream handoff pointed at — ``"verifier"`` after the planner,
    ``"planner"`` after a failed verifier.

    Note:
        The returned names are destinations, not nodes — the compiled graph has
        to bind them via ``add_conditional_edges``. The compressor never routes
        to itself: even if the fresh window were somehow still over budget, the
        next monitor visit is the one that measures and decides again.
    """
    return state.get("context_next_node", "verifier")
