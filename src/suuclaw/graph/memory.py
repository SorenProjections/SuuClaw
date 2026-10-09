"""The three-layer memory system: fixed rules, working memory, compressed history.

Why layers, not one transcript
------------------------------
The raw ``messages`` channel is an append-only audit trail — replaying it into
every prompt would splice several half-conversations together and blow the
budget besides. Nodes instead read *assembled* memory: a snapshot with exactly
three layers, each with a different lifetime:

* **Rules layer** (:data:`RULES_LAYER`) — fixed policy, identical on every
  call. It is never derived from state, so a prompt can treat it as ground
  truth about how the run is allowed to behave (workspace scope, path style,
  which file means what).
* **Working memory** (:func:`build_layered_memory`) — the current task's live
  state: plan, todos, criteria, the specialist handoffs, the last error. It is
  rebuilt from the graph state on every call and never persisted; it is the
  "right now" view.
* **History summary store** — the compressed past, in two forms: the durable
  ``HISTORY_SUMMARY.md`` file in the workspace (plus the notepad it sits next
  to) and the in-state ``context_summary`` / ``compression_events`` the
  compression step maintains. When the raw context is folded up, this is the
  layer that survives.

Everything is truncated per-field (see :func:`_short_text`) so no single
channel can blow the prompt on its own; the shape is what matters to the
reader, not every byte.

The snapshot is data, not prose
-------------------------------
:func:`build_layered_memory` returns a plain dict and
:func:`format_layered_memory_for_prompt` renders it as JSON. JSON, not
hand-rolled prose, because the shape is part of the contract: a node that
wants one field reads it structurally, and a prompt that shows the whole
snapshot shows something a model can parse reliably.

Note:
    The rules say memory is *assembled by the runtime* — so this module has no
    tools and no model calls. It reads the workspace files and the state and
    returns dicts; whoever calls it decides where the JSON goes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypedDict

from ..core.state import RuntimeState
from ..tools.notepad_tools import NOTEPAD_FILENAME, read_notepad

__all__ = [
    "HISTORY_SUMMARY_FILENAME",
    "RULES_LAYER",
    "CompressionEvent",
    "LayeredMemory",
    "append_memory_block",
    "build_layered_memory",
    "format_layered_memory_for_prompt",
    "memory_event",
    "read_history_summary",
]

#: The compressed-history file, sibling of the notepad in the workspace root.
HISTORY_SUMMARY_FILENAME = "HISTORY_SUMMARY.md"

#: Per-field character budgets. Each layer caps its own fields so one bloated
#: channel cannot crowd the others out of the prompt.
RESEARCH_NOTES_CHARS = 1600
CODE_AGENT_SUMMARY_CHARS = 1000
VERIFIER_SUMMARY_CHARS = 1000
LAST_ERROR_CHARS = 1400
HISTORY_SUMMARY_CHARS = 2200
NOTEPAD_CHARS = 1800
CONTEXT_SUMMARY_CHARS = 1600

#: How many recent specialist handoffs the snapshot carries. Older ones are
#: gone from the prompt, not from the state — the ``agent_handoffs`` channel
#: stays complete for audit.
HANDOFFS_KEEP = 6

#: How many recent compression passes the snapshot carries.
COMPRESSION_EVENTS_KEEP = 3


class CompressionEvent(TypedDict, total=False):
    """One context-compression pass, as recorded by the compression step.

    ``tokens_before`` / ``tokens_after`` are the estimates that triggered the
    pass and the post-compression estimate; ``summary_chars`` is how large the
    produced ``context_summary`` ended up. ``note`` is free text for anything
    the numbers cannot say ("messages replayed from a failed attempt").
    """

    at: str  # ISO-8601 timestamp of the pass
    trigger: str  # e.g. "token_limit"
    tokens_before: int
    tokens_after: int
    summary_chars: int
    note: str


class LayeredMemory(TypedDict, total=False):
    """The shape :func:`build_layered_memory` returns.

    Three keys, one per layer; each layer is its own plain dict (the shapes of
    ``working_memory`` and ``history_summary_store`` are fixed by
    :func:`build_layered_memory`, which is the single place that defines them).
    Declared as a ``TypedDict`` so the ``memory_snapshot`` state channel has a
    name — nodes still read it with ``state.get("memory_snapshot", {})``.
    """

    rules: dict[str, Any]
    working_memory: dict[str, Any]
    history_summary_store: dict[str, Any]


RULES_LAYER: dict[str, Any] = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        (
            "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and "
            "HISTORY_SUMMARY.md as compressed history."
        ),
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}


# --------------------------------------------------------------------------- #
# workspace-backed reads
# --------------------------------------------------------------------------- #
def read_history_summary(runtime: RuntimeState) -> dict[str, Any]:
    """The compressed-history file as ``{"exists", "content"}``.

    The history store is read-only from here: writing it is the compression
    step's job, this is the shared read path the snapshot (and any node that
    wants the same bytes) uses. Missing file reads as ``exists=False`` with an
    empty body — a first run has no history yet, and that is not an error.
    """
    path = runtime.resolve(HISTORY_SUMMARY_FILENAME)
    if not path.exists():
        return {"exists": False, "content": ""}
    return {"exists": True, "content": path.read_text(encoding="utf-8")}


def _notepad_record(runtime: RuntimeState) -> dict[str, Any]:
    """The notepad as ``{"exists", "content"}``, via the shared read path.

    ``exists`` is the file's real existence, so an empty-but-present notepad
    reads as ``exists=True, content=""`` — honest about the difference between
    "nothing noted yet" and "noted, and it is blank".
    """
    path = runtime.resolve(NOTEPAD_FILENAME)
    return {"exists": path.exists(), "content": read_notepad(runtime)}


# --------------------------------------------------------------------------- #
# shaping helpers
# --------------------------------------------------------------------------- #
def _short_text(text: Any, limit: int) -> str:
    """``text`` capped at ``limit`` characters, with ``...`` marking the cut.

    The one guard every variable-length field goes through: state channels are
    loose input (``None``, a list, a 40k-char report), and the snapshot's job
    is to hand the reader a bounded, predictable field, not to trust the
    writer. A cut always announces itself with the trailing ``...``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _trim_handoffs(handoffs: Any) -> list[dict[str, Any]]:
    """The most recent :data:`HANDOFFS_KEEP` handoff records, normalized to dicts.

    Order is preserved (oldest first) so the list still reads as a timeline;
    only the *front* is dropped. Non-mapping entries are skipped rather than
    coerced — a corrupt row must not crowd out real ones.
    """
    if not isinstance(handoffs, (list, tuple)):
        return []
    rows = [dict(item) for item in handoffs if isinstance(item, Mapping)]
    return rows[-HANDOFFS_KEEP:]


def _source_rows(sources: Any) -> list[dict[str, str]]:
    """Sources projected to ``{"title", "url"}`` — the citation, nothing else.

    Scores and ranking metadata helped the search agent pick them; a prompt
    reader only needs to know where a claim came from.
    """
    if not isinstance(sources, (list, tuple)):
        return []
    rows: list[dict[str, str]] = []
    for item in sources:
        if not isinstance(item, Mapping):
            continue
        rows.append({"title": str(item.get("title", "")), "url": str(item.get("url", ""))})
    return rows


def _session_id(runtime: RuntimeState) -> str:
    """The run's session id: explicit metadata when given, else the start time.

    ``RuntimeState`` has no session concept of its own, so the id comes from
    ``runtime.metadata["session_id"]`` when the caller (CLI, notebook, test)
    supplied one, and falls back to a timestamp derived from the runtime's
    creation time — stable for the whole run, unique across runs.
    """
    metadata = getattr(runtime, "metadata", None) or {}
    session_id = str(metadata.get("session_id") or "").strip()
    if session_id:
        return session_id
    return runtime.created_at.strftime("%Y%m%dT%H%M%SZ")


def _session_turn(state: Mapping[str, Any], runtime: RuntimeState) -> int:
    """One-based turn counter: how many plan→verify rounds have *begun*.

    ``attempts`` counts finished rounds, so ``attempts + 1`` is the round in
    flight; an explicit ``runtime.metadata["session_turn"]`` (set by a caller
    resuming a session) wins when present.
    """
    metadata = getattr(runtime, "metadata", None) or {}
    raw = metadata.get("session_turn")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    try:
        return int(state.get("attempts", 0) or 0) + 1
    except (TypeError, ValueError):
        return 1


# --------------------------------------------------------------------------- #
# the snapshot
# --------------------------------------------------------------------------- #
def build_layered_memory(state: Mapping[str, Any], *, node: str = "graph") -> LayeredMemory:
    """Assemble the three-layer snapshot for the given state.

    Args:
        state: The graph state. ``runtime`` (required — the workspace reads
            hang off it) plus any channels the layers need; everything is
            read with ``state.get``, so a fresh run with an empty state
            produces a valid, empty snapshot rather than a traceback.
        node: Which node is asking. Rides along in ``working_memory`` so a
            shared prompt builder can say whose memory this is.

    Returns:
        A :class:`LayeredMemory` dict — ``{"rules", "working_memory",
        "history_summary_store"}`` — ready for
        :func:`format_layered_memory_for_prompt` or direct field access. The
        rules layer is a copy, so callers can annotate their snapshot without
        mutating :data:`RULES_LAYER` for everyone else.
    """
    runtime = state.get("runtime")
    if runtime is None:
        raise ValueError(
            "build_layered_memory needs state['runtime'] — seed the run with "
            "{'task': ..., 'runtime': RuntimeState.create(workspace)}"
        )

    notepad = _notepad_record(runtime)
    history = read_history_summary(runtime)

    working_memory: dict[str, Any] = {
        "node": node,
        "task": str(state.get("task", "") or ""),
        "session_id": _session_id(runtime),
        "session_turn": _session_turn(state, runtime),
        "plan_summary": str(state.get("plan_summary", "") or ""),
        "todos": list(state.get("todos", []) or []),
        "acceptance_criteria": list(state.get("acceptance_criteria", []) or []),
        "verification_commands": list(state.get("verification_commands", []) or []),
        "research_notes": _short_text(state.get("research_notes", ""), RESEARCH_NOTES_CHARS),
        "sources": _source_rows(state.get("sources", [])),
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""), CODE_AGENT_SUMMARY_CHARS
        ),
        "verifier_summary": _short_text(state.get("verifier_summary", ""), VERIFIER_SUMMARY_CHARS),
        "last_error": _short_text(state.get("last_error", ""), LAST_ERROR_CHARS),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

    # The freshest history wins: an in-state summary the compression step has
    # produced but not yet flushed to disk beats the file's older contents.
    history_summary = str(state.get("history_summary") or "").strip() or history.get("content", "")

    history_summary_store: dict[str, Any] = {
        "history_path": HISTORY_SUMMARY_FILENAME,
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history_summary, HISTORY_SUMMARY_CHARS),
        "notepad_path": NOTEPAD_FILENAME,
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), NOTEPAD_CHARS),
        "context_summary": _short_text(state.get("context_summary", ""), CONTEXT_SUMMARY_CHARS),
        "compression_events": list(state.get("compression_events", []) or [])[
            -COMPRESSION_EVENTS_KEEP:
        ],
    }

    return {
        "rules": dict(RULES_LAYER),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def format_layered_memory_for_prompt(memory: Mapping[str, Any]) -> str:
    """Render a snapshot as the JSON block a prompt embeds.

    ``ensure_ascii=False`` keeps the text human-readable (the prompt is read
    by a model, but it is written in the run's language), and ``default=str``
    means a stray non-JSON value degrades to its string form instead of
    raising in the middle of a prompt build.
    """
    return json.dumps(dict(memory), ensure_ascii=False, indent=2, default=str)


# --------------------------------------------------------------------------- #
# delivery helpers
# --------------------------------------------------------------------------- #
def memory_event(memory: Mapping[str, Any], *, node: str) -> dict[str, Any]:
    """The custom-stream payload announcing that ``node`` received its snapshot.

    One event per prompt build, emitted *before* the model call so a front end
    can show what the agent is about to see. The snapshot rides in whole —
    consumers who want one field read it structurally, exactly like a prompt
    would; nobody needs a second, lossy projection of it. The payload is a
    true copy (a JSON roundtrip): a consumer mutating what it received must
    not be able to bend the caller's snapshot, and non-JSON values degrade to
    strings exactly as they would in the prompt.
    """
    safe = json.dumps(dict(memory), ensure_ascii=False, default=str)
    return {"type": "memory", "node": node, "memory": json.loads(safe)}


def append_memory_block(request: str, memory: Mapping[str, Any]) -> str:
    """Join a request brief and its snapshot under the shared section header.

    The header is a contract with the compressor's fake-model tests and any
    prompt reader that wants to split the brief from the memory: the snapshot
    always arrives as the last ``# Layered memory`` block of the human message.
    """
    blocks = [request.rstrip(), "# Layered memory", format_layered_memory_for_prompt(memory)]
    return "\n\n".join(blocks)
