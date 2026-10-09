"""Tracing: what a run did, written down while it happens.

A checkpoint answers "how do I get this run back"; a trace answers "what actually
happened". They read the same event stream and write different things, so both can
be on at once::

    <workspace>/.suuclaw/traces/<trace_id>/
        trace.json     the run's numbers, plus the head and tail of its timeline
        timeline.md    the same timeline as prose, for a human
        events.jsonl   one line per event, appended as it arrives   (strict only)

Three modes, one policy each:

* ``light`` — ``trace.json`` and ``timeline.md``, written in one go when the run
  ends. A few kilobytes, and what a post-mortem usually needs.
* ``strict`` — adds ``events.jsonl``, appended live. This is the mode a *crash*
  leaves evidence in: ``light`` only writes at the end, so a run that dies hard
  leaves nothing behind at all.
* ``off`` — nothing is written, and every method returns ``None``.

Two properties this module refuses to give up:

* **A trace never breaks the run it is recording.** Every writer swallows its own
  failure and records it in ``last_error``; a full disk must not kill a run that
  was otherwise fine.
* **A trace never grows without bound.** The in-memory timeline is capped, long
  strings are trimmed before they reach ``events.jsonl``, and ``trace.json``
  carries the head and the tail of the timeline rather than all of it.

``.suuclaw/`` ignores itself (the same ``.gitignore`` the checkpoint writer
maintains), so a trace never shows up in the user's ``git status``.

Layering: module scope imports the standard library and
:mod:`suuclaw.core.checkpoint` — the state-summary readers (todo counts, the
verification tally, ``_jsonable``) live there because ``RECOVERY.md`` needed them
first, and a second copy of them here would be a second thing to keep in sync.
If that list keeps growing they belong in a shared ``core/_summaries.py``.
Nothing from ``tools`` or ``graph`` is imported at module scope, because
:mod:`suuclaw.core.state` imports this module for its mode constants and a
module-scope import in the other direction would close that cycle.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .checkpoint import (
    _as_int,
    _context_summary,
    _jsonable,
    _open_todos,
    _shorten,
    _todo_counts,
    _verification_summary,
    _write_gitignore,
)
from .paths import runtime_path

if TYPE_CHECKING:  # pragma: no cover - typing only, see the layering note above
    from .state import RuntimeState

__all__ = [
    "DEFAULT_TRACE_MODE",
    "MAX_TIMELINE_ROWS",
    "TIMELINE_HEAD",
    "TIMELINE_TAIL",
    "TRACE_DIR_PARTS",
    "VALID_TRACE_MODES",
    "TraceRecorder",
    "build_timeline_markdown",
    "new_trace_id",
    "normalize_trace_mode",
]

#: The trace tree, relative to the workspace. Part zero is the directory whose
#: self-ignoring ``.gitignore`` this module shares with the checkpoint writer.
TRACE_DIR_PARTS = (".suuclaw", "traces")

#: ``light`` is the default for the same reason ``light`` is the checkpoint
#: default: it costs almost nothing and still answers the question. ``off`` exists
#: to say no outright, and it has to be said outright — a default of ``off`` would
#: make a request for a trace fail silently, which is the worst way to fail.
VALID_TRACE_MODES: set[str] = {"light", "strict", "off"}
DEFAULT_TRACE_MODE = "light"

#: How much of the timeline ``trace.json`` and ``timeline.md`` carry: the opening
#: moves and the closing ones are what a post-mortem reads, and a run can produce
#: thousands of rows.
TIMELINE_HEAD = 20
TIMELINE_TAIL = 80

#: Hard cap on the in-memory timeline. Never reached by a run this graph can
#: produce (the recursion limit bounds the passes), but "unbounded list fed by a
#: stream" is not a thing to leave lying around.
MAX_TIMELINE_ROWS = 20_000

MAX_SUMMARY_CHARS = 200
MAX_FINAL_ANSWER_CHARS = 2000

#: ``events.jsonl`` is meant to stay greppable, so a single 20 KB tool result is
#: trimmed before it is written. The full text is still in the tool's own result.
MAX_EVENT_STRING_CHARS = 4000

#: Workflow-level events, i.e. the ones the driver emits about the graph itself.
#: Everything else came off the ``custom`` stream, and is where the counters are
#: kept. Kept as a set so :meth:`TraceRecorder.record` can route for a caller that
#: does not want to split the stream itself.
WORKFLOW_EVENT_KINDS: frozenset[str] = frozenset(
    {"node_start", "node_output", "node_summary", "final_answer"}
)

#: The prose the ``bash`` tool puts in a result that went through the approval
#: gate. The gate's ``requires_approval`` flag is carried in the tool's own
#: ``CommandResult``, which the event stream flattens into text, so the marker is
#: the only thing on the wire today. Both paths are checked; see
#: :func:`_approval_required`.
_APPROVAL_MARKER = "approval: required"


def normalize_trace_mode(mode: str | None) -> str:
    """Coerce ``mode`` to one of :data:`VALID_TRACE_MODES`.

    An unknown value falls back to :data:`DEFAULT_TRACE_MODE` rather than raising:
    the value arrives from a CLI flag, and refusing to finish a run because a typo
    said ``verbose`` would be a poor trade.
    """
    if not mode:
        return DEFAULT_TRACE_MODE
    normalized = str(mode).strip().lower()
    return normalized if normalized in VALID_TRACE_MODES else DEFAULT_TRACE_MODE


def new_trace_id(at: datetime | None = None) -> str:
    """A sortable, unique trace id — ``trace-20260913T164355Z-a1b2c3``.

    Timestamp first so traces sort chronologically in a file listing, random tail
    so two runs in the same second cannot collide.
    """
    moment = at or _utcnow()
    return f"trace-{moment.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"


class TraceRecorder:
    """Accumulates one run's events and writes them to the workspace.

    One recorder records exactly one run. A caller drives it from the event stream
    it already renders::

        recorder = TraceRecorder(state, task)
        recorder.start(inputs, resumed=resumed, resume_event=event)
        for event in events:
            recorder.record(event)
        recorder.end(status="passed" if passed else "failed",
                     latest_node="final", final_state=final_state)

    :meth:`record` routes on the event's ``type`` and is the method to use when the
    caller has a single stream; :meth:`record_custom_event` and
    :meth:`record_graph_update` are the two halves of it, for a caller that already
    separates the inner-loop events from the graph-level ones.

    The counters live in :meth:`record_custom_event`. That is the split the event
    catalogue draws: ``tool_call`` / ``tool_result`` / ``handoff`` /
    ``checkpoint_saved`` are the inner loop's, and the node bookkeeping is the
    graph's.
    """

    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.workspace = Path(runtime.workspace)
        self.mode = normalize_trace_mode(getattr(runtime, "trace_mode", None))
        self.task = task or ""
        #: A run's identity, so events from several sources can be joined later.
        #: Taken from the runtime when it has one (a resume continues the same
        #: trace), generated otherwise. The runtime is *not* written back to: a
        #: constructor that mutates its argument is a surprise waiting to happen.
        self.trace_id = str(getattr(runtime, "trace_id", "") or "") or new_trace_id()
        #: Where every file below lives.
        self.root = runtime_path(self.workspace, "traces", self.trace_id)

        # -- the run's numbers, the reason this module exists ----------------
        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0

        # -- book-keeping ----------------------------------------------------
        self.created_at = _utcnow()
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.status = "pending"
        self.latest_node = ""
        self.resumed = False
        self.timeline_dropped = 0
        #: Why the last write failed, if it did. Every method reports the same
        #: thing through its return value; this is for callers that log instead.
        self.last_error = ""
        self.errors: list[str] = []

        self._timeline: list[dict[str, Any]] = []
        #: Nodes announced with ``node_start`` whose ``node_output`` has not
        #: arrived yet — see :meth:`_count_visit`.
        self._announced: set[str] = set()
        self._root_ready = False

    # -- layout -------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether this recorder writes anything at all."""
        return self.mode != "off"

    @property
    def trace_json(self) -> Path:
        return self.root / "trace.json"

    @property
    def timeline_md(self) -> Path:
        return self.root / "timeline.md"

    @property
    def events_jsonl(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def timeline_count(self) -> int:
        """Rows currently held in memory, dropped ones excluded."""
        return len(self._timeline)

    # -- the run ------------------------------------------------------------
    def start(
        self,
        inputs: Mapping[str, Any],
        *,
        resumed: bool = False,
        resume_event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Mark the run as begun and record its seed; ``None`` when disabled.

        Never raises: the directory is created here so a trace that exists at all
        means a run got as far as starting.
        """
        if not self.enabled:
            return None

        self.started_at = _utcnow()
        self.status = "running"
        self.resumed = bool(resumed)
        # The task can arrive here when the caller only had it in the seed state,
        # and every file this module writes names the task, so adopt it once.
        run_task = str(inputs.get("task") or self.task or "")
        if not self.task:
            self.task = run_task

        try:
            self._ensure_root()
        except OSError as exc:
            self._note_error(f"cannot create {self.root}: {exc}")

        event: dict[str, Any] = {
            "type": "run_start",
            "node": "",
            "loop": 0,
            "trace_id": self.trace_id,
            "mode": self.mode,
            "workspace": str(self.workspace),
            "task": run_task,
            "max_attempts": _as_int(inputs.get("max_attempts")),
            "resumed": self.resumed,
        }
        if resume_event is not None:
            event["resume"] = _jsonable(resume_event)

        row = self._row(event)
        row["summary"] = "run started" + (" (resumed)" if self.resumed else "")
        self._append(row, event)
        return event

    def end(
        self,
        *,
        status: str = "finished",
        latest_node: str = "",
        final_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Write ``trace.json`` and ``timeline.md``; ``None`` when disabled.

        Returns the payload it wrote — also when the *write* failed, in which case
        ``write_error`` says so. Never raises, for the same reason
        :meth:`~suuclaw.core.checkpoint.CheckpointManager.save` never does: the
        bookkeeping is not worth the run.
        """
        if not self.enabled:
            return None

        self.ended_at = _utcnow()
        self.status = str(status or "").strip() or self.status
        self.latest_node = str(latest_node or "")
        base = self.started_at or self.created_at
        duration_ms = max(0, int((self.ended_at - base).total_seconds() * 1000))

        head, tail, omitted = _split_timeline(self._timeline)
        # Rows dropped off the front are not on the timeline either, so they count
        # as omitted too: head + tail + omitted has to add up to timeline_total,
        # otherwise the three numbers are a puzzle instead of a summary.
        omitted += self.timeline_dropped
        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "task": self._task(),
            "status": self.status,
            "latest_node": self.latest_node,
            "workspace": str(self.workspace),
            "mode": self.mode,
            "resumed": self.resumed,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "duration_ms": duration_ms,
            "node_visits": dict(self.node_visits),
            "node_visits_total": sum(self.node_visits.values()),
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "timeline_head": head,
            "timeline_tail": tail,
            "timeline_omitted": omitted,
            "timeline_total": len(self._timeline) + self.timeline_dropped,
            "timeline_dropped": self.timeline_dropped,
            "outcome": _outcome(final_state, status=self.status, latest_node=self.latest_node),
            "paths": {
                "trace": str(self.trace_json),
                "timeline": str(self.timeline_md),
                "events": str(self.events_jsonl) if self.mode == "strict" else "",
            },
            "errors": list(self.errors),
            "write_error": "",
        }

        try:
            self._ensure_root()
            _write_json(self.trace_json, payload)
            self._write_markdown(payload)
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.errors.append(self.last_error)
            payload["errors"] = list(self.errors)
            payload["write_error"] = self.last_error
        return payload

    def _write_markdown(self, payload: Mapping[str, Any]) -> None:
        self.timeline_md.write_text(build_timeline_markdown(payload), encoding="utf-8")

    # -- events -------------------------------------------------------------
    def record(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        """Record one event, routed by its ``type``; the timeline row, or ``None``.

        Both halves of the driver's stream end up here unchanged, which is the
        point: a caller renders and records from the same loop.
        """
        if not isinstance(event, Mapping):
            return None
        kind = str(event.get("type") or "")
        if kind in WORKFLOW_EVENT_KINDS:
            return self.record_graph_update(event)
        return self.record_custom_event(event)

    def record_custom_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        """Record an inner-loop event and fold it into the counters.

        Which counters, exactly: ``tool_call`` bumps ``tool_calls``; a
        ``tool_result`` whose ``ok`` is falsy bumps ``failed_tool_calls`` and one
        that came out of the approval gate bumps ``approval_count`` (both can
        apply); ``handoff`` bumps ``handoff_count``; ``checkpoint_saved`` bumps
        ``checkpoint_count``.
        """
        if not self.enabled or not isinstance(event, Mapping):
            return None

        kind = str(event.get("type") or "")
        if kind == "tool_call":
            self.tool_calls += 1
        elif kind == "tool_result":
            if not _ok_of(event):
                self.failed_tool_calls += 1
            if _approval_required(event):
                self.approval_count += 1
        elif kind == "handoff":
            self.handoff_count += 1
        elif kind == "checkpoint_saved":
            self.checkpoint_count += 1

        row = self._row(event)
        self._append(row, event)
        return row

    def record_graph_update(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        """Record a graph-level event and count the node visit it implies.

        ``node_start`` / ``node_output`` / ``node_summary`` / ``final_answer`` all
        belong here; anything else is recorded without being counted, so a
        mis-routed event still shows up in the timeline.
        """
        if not self.enabled or not isinstance(event, Mapping):
            return None

        kind = str(event.get("type") or "")
        if kind in {"node_start", "node_output"}:
            self._count_visit(kind, str(event.get("node") or ""))

        row = self._row(event)
        self._append(row, event)
        return row

    def _count_visit(self, kind: str, node: str) -> None:
        """Count one visit to ``node``, from whichever half of the pair arrives.

        The driver announces a visit with ``node_start`` and closes it with
        ``node_output``, so counting both would double every number. Counting only
        ``node_start`` would report nothing for a caller that feeds the updates
        stream alone. So a visit is counted by the announcement, and the
        ``node_output`` that closes it is marked as already counted — falling back
        to counting the output when no announcement ever arrived.
        """
        if not node:
            return
        if kind == "node_start":
            if node not in self._announced:
                self._announced.add(node)
                self.node_visits[node] = self.node_visits.get(node, 0) + 1
            return
        if node in self._announced:
            self._announced.discard(node)
            return
        self.node_visits[node] = self.node_visits.get(node, 0) + 1

    def _row(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """The timeline row for ``event`` — compact, JSON-safe, never the whole event."""
        now = _utcnow()
        base = self.started_at or self.created_at
        row: dict[str, Any] = {
            "at": _iso(now),
            "t_ms": max(0, int((now - base).total_seconds() * 1000)),
            "kind": str(event.get("type") or "unknown"),
            "node": str(event.get("node") or ""),
            "summary": _summarize_event(event),
        }
        # ``loop 0`` is noise — the seed event carries it and no ReAct loop is
        # numbered 0 — so it is dropped rather than rendered as "run_start L0".
        if _is_int(event.get("loop")) and int(event["loop"]) > 0:
            row["loop"] = int(event["loop"])
        return row

    def _append(self, row: dict[str, Any], event: Mapping[str, Any]) -> None:
        """Keep ``row`` in memory, drop the oldest beyond the cap, log it if strict."""
        self._timeline.append(row)
        if len(self._timeline) > MAX_TIMELINE_ROWS:
            dropped = len(self._timeline) - MAX_TIMELINE_ROWS
            del self._timeline[:dropped]
            self.timeline_dropped += dropped
        if self.mode == "strict":
            self._append_event_line(row, event)

    def _append_event_line(self, row: Mapping[str, Any], event: Mapping[str, Any]) -> None:
        """Append one line to ``events.jsonl``. A failure is recorded, not raised."""
        record = {
            "at": row.get("at") or "",
            "t_ms": _as_int(row.get("t_ms")),
            "kind": row.get("kind") or "",
            "node": row.get("node") or "",
            "loop": _as_int(row.get("loop")),
            "event": _trim_strings(_jsonable(event)),
        }
        try:
            self._ensure_root()
            with self.events_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self._note_error(f"cannot append to {self.events_jsonl}: {exc}")

    # -- internals ----------------------------------------------------------
    def _ensure_root(self) -> None:
        """Create the trace directory once, and keep its metadata root out of git.

        The ignore file is the checkpoint writer's, and it belongs to the
        metadata root — *not* to ``traces/``: one self-ignoring file there
        covers both trees, and writing it from here means a trace stays out of the
        user's ``git status`` even when checkpoints are off.
        """
        if self._root_ready:
            return
        _write_gitignore(self.root.parent.parent)
        self.root.mkdir(parents=True, exist_ok=True)
        self._root_ready = True

    def _note_error(self, message: str) -> None:
        self.last_error = message
        if message not in self.errors:
            self.errors.append(message)

    def _task(self) -> str:
        return str(self.task or "")


# --------------------------------------------------------------------------- #
# event and payload helpers
# --------------------------------------------------------------------------- #
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str:
    return value.isoformat(timespec="seconds") if value else ""


def _is_int(value: Any) -> bool:
    """True for values ``int()`` accepts, so a loop number can be read safely."""
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _one_line(text: str) -> str:
    """Collapse whitespace, because a summary is one cell in a table."""
    return re.sub(r"\s+", " ", text).strip()


def _preview(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """A one-line, length-capped preview of ``text``."""
    flat = _one_line(text)
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _ok_of(event: Mapping[str, Any]) -> bool:
    """Whether a ``tool_result`` succeeded.

    The driver puts ``ok`` at the top level; a hand-assembled event may carry it
    inside ``result``. A missing flag reads as *success*: ``failed_tool_calls`` is
    a claim, and an unknown should not be counted as a failure.
    """
    if "ok" in event:
        return bool(event.get("ok"))
    result = event.get("result")
    if isinstance(result, Mapping) and "ok" in result:
        return bool(result.get("ok"))
    return True


def _approval_required(event: Mapping[str, Any]) -> bool:
    """Whether a ``tool_result`` went through the approval gate.

    Two shapes, because the flag has two carriers. A caller that keeps the
    ``CommandResult`` can put ``requires_approval`` in the event (anywhere in it —
    the result envelope is nested one level down in practice), and the ``bash``
    tool also writes a human-readable marker into its output, which is what the
    event stream actually carries today. The structured check runs first so this
    gets *more* precise, never less, once the flag is wired through.
    """
    if _deep_flag(event, "requires_approval"):
        return True
    return _deep_text(event.get("result") if "result" in event else event, _APPROVAL_MARKER)


def _deep_flag(value: Any, key: str, depth: int = 4) -> bool:
    """True when ``key`` appears anywhere below ``value`` with a truthy value."""
    if depth < 0:
        return False
    if isinstance(value, Mapping):
        if value.get(key):
            return True
        return any(_deep_flag(item, key, depth - 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_deep_flag(item, key, depth - 1) for item in value)
    return False


def _deep_text(value: Any, marker: str, depth: int = 4) -> bool:
    """True when ``marker`` appears in any string below ``value``."""
    if depth < 0:
        return False
    if isinstance(value, str):
        return marker in value
    if isinstance(value, Mapping):
        return any(_deep_text(item, marker, depth - 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_deep_text(item, marker, depth - 1) for item in value)
    return False


def _trim_strings(value: Any, depth: int = 6) -> Any:
    """Cap every string below ``value`` before it is written to ``events.jsonl``.

    Trimming the *values* rather than the serialised line keeps the line valid
    JSON: a truncated JSON object is not a record, it is a parse error.
    """
    if depth < 0:
        return _shorten(repr(value), MAX_EVENT_STRING_CHARS)
    if isinstance(value, str):
        return _shorten(value, MAX_EVENT_STRING_CHARS)
    if isinstance(value, Mapping):
        return {str(key): _trim_strings(item, depth - 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trim_strings(item, depth - 1) for item in value]
    return value


def _summarize_event(event: Mapping[str, Any]) -> str:
    """One line saying what an event did — the timeline's ``detail`` column."""
    kind = str(event.get("type") or "")
    if kind == "node_start":
        return "start"
    if kind == "node_output":
        data = event.get("data")
        if not isinstance(data, Mapping) or not data:
            return "update (empty)"
        bits = [f"update: {', '.join(sorted(str(key) for key in data))}"]
        if "passed" in data:
            bits.append(f"passed={bool(data.get('passed'))}")
        if "attempts" in data:
            bits.append(f"attempts={_as_int(data.get('attempts'))}")
        return " · ".join(bits)
    if kind == "node_summary":
        return _loop_summary(event, "summary")
    if kind == "final_answer":
        return (
            f"final: passed={bool(event.get('passed'))} "
            f"attempts={_as_int(event.get('attempts'))}/{_as_int(event.get('max_attempts'))}"
        )
    if kind == "ai_message":
        bits: list[str] = []
        content = _preview(str(event.get("content") or ""))
        if content:
            bits.append(content)
        planned = [str(item) for item in (event.get("tool_calls") or [])]
        if planned:
            bits.append("plans " + ", ".join(planned))
        malformed = [str(item) for item in (event.get("invalid_tool_calls") or [])]
        if malformed:
            bits.append("malformed " + ", ".join(malformed))
        return f"ai ({_as_int(event.get('loop'))}): " + (" · ".join(bits) or "(no content)")
    if kind == "tool_call":
        name = str(event.get("name") or "?")
        args = json.dumps(_jsonable(event.get("args")), ensure_ascii=False, default=str)
        return f"{name}({_preview(args, 120)})"
    if kind == "tool_result":
        name = str(event.get("name") or "?")
        tail = " · approval required" if _approval_required(event) else ""
        return f"{name} {'ok' if _ok_of(event) else 'failed'}{tail}"
    if kind == "error":
        return "error: " + _preview(str(event.get("message") or ""))
    if kind == "retry":
        return (
            f"retry {_as_int(event.get('attempt'))}/{_as_int(event.get('max_retries'))} "
            f"in {event.get('delay')}s: {_preview(str(event.get('message') or ''), 120)}"
        )
    if kind == "handoff":
        return f"{event.get('from') or '?'} → {event.get('to') or '?'}: " + _preview(
            str(event.get("instruction") or "")
        )
    if kind == "checkpoint_saved":
        return (
            f"{event.get('id') or '?'} ({event.get('mode') or '?'}, "
            f"{_as_int(event.get('files'))} file(s))"
        )
    if kind == "checkpoint_error":
        return "checkpoint failed: " + _preview(str(event.get("message") or ""))
    if kind == "run_start":
        return "run started" + (" (resumed)" if event.get("resumed") else "")
    return _preview(json.dumps(_jsonable(event), ensure_ascii=False, default=str))


def _loop_summary(event: Mapping[str, Any], label: str) -> str:
    """``summary (3 loops) [exhausted]: text`` — a node's closing turn, condensed."""
    flags = [name for name in ("exhausted", "interrupted", "error") if event.get(name)]
    loops = _as_int(event.get("loops") if event.get("loops") is not None else event.get("loop"))
    suffix = f" [{', '.join(flags)}]" if flags else ""
    return f"{label} ({loops} loop(s)){suffix}: {_preview(str(event.get('content') or ''))}"


def _split_timeline(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Head, tail, and how many rows neither covers.

    ``head`` and ``tail`` never overlap, and together they always cover the whole
    timeline when the run is short: with 40 rows that is rows 0-19 and 20-39, not
    the same 20 rows counted twice.
    """
    total = len(rows)
    if total <= TIMELINE_HEAD + TIMELINE_TAIL:
        head = [dict(row) for row in rows[:TIMELINE_HEAD]]
        tail = [dict(row) for row in rows[TIMELINE_HEAD:]]
        return head, tail, 0
    head = [dict(row) for row in rows[:TIMELINE_HEAD]]
    tail = [dict(row) for row in rows[-TIMELINE_TAIL:]]
    return head, tail, total - TIMELINE_HEAD - TIMELINE_TAIL


def _outcome(
    final_state: Mapping[str, Any] | None,
    *,
    status: str,
    latest_node: str,
) -> dict[str, Any]:
    """What the run ended up with, read off its final graph state.

    Defensive like ``build_recovery_markdown``: it is read after the run is over,
    so an unexpected channel is a dash rather than an exception.
    """
    outcome: dict[str, Any] = {"status": status, "node": latest_node, "channels": []}
    if not isinstance(final_state, Mapping):
        return outcome
    outcome.update(
        {
            "passed": bool(final_state.get("passed")),
            "attempts": _as_int(final_state.get("attempts")),
            "max_attempts": _as_int(final_state.get("max_attempts")),
            "todos": _todo_counts(final_state),
            "open_todos": _open_todos(final_state),
            "verification": _verification_summary(final_state),
            "context": _context_summary(final_state),
            "last_error": _shorten(str(final_state.get("last_error") or ""), 500),
            "final_answer": _shorten(
                str(final_state.get("final_answer") or ""), MAX_FINAL_ANSWER_CHARS
            ),
            "channels": sorted(str(key) for key in final_state),
        }
    )
    return outcome


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` atomically, so a reader never sees half a file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass


def _row_cell(value: Any) -> str:
    """One markdown table cell: a single line, with pipes escaped."""
    return _one_line(str(value if value is not None else "")).replace("|", "\\|")


def _timeline_table(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """The timeline as a markdown table — ``+t`` is the offset from the run's start."""
    lines = ["| +t | kind | node | detail |", "| --- | --- | --- | --- |"]
    for row in rows:
        kind = _row_cell(row.get("kind"))
        if _is_int(row.get("loop")):
            kind = f"{kind} L{int(row['loop'])}"
        lines.append(
            f"| +{_as_int(row.get('t_ms')) / 1000:.3f}s | {kind} "
            f"| {_row_cell(row.get('node'))} | {_row_cell(row.get('summary'))} |"
        )
    return lines


def build_timeline_markdown(payload: Mapping[str, Any]) -> str:
    """Render the human-facing ``timeline.md`` for a trace ``payload``.

    Defensive on purpose, for the same reason ``RECOVERY.md`` is: this file exists
    to answer "what happened", and a missing key must not be what stops it.
    """
    visits = payload.get("node_visits")
    if not isinstance(visits, Mapping):
        visits = {}
    paths = payload.get("paths")
    if not isinstance(paths, Mapping):
        paths = {}
    outcome = payload.get("outcome")
    if not isinstance(outcome, Mapping):
        outcome = {}
    head = [row for row in (payload.get("timeline_head") or []) if isinstance(row, Mapping)]
    tail = [row for row in (payload.get("timeline_tail") or []) if isinstance(row, Mapping)]
    omitted = _as_int(payload.get("timeline_omitted"))
    dropped = _as_int(payload.get("timeline_dropped"))
    dropped_note = f" (+{dropped} dropped)" if dropped else ""
    duration_ms = _as_int(payload.get("duration_ms"))
    strict = str(payload.get("mode") or "") == "strict"
    mode_note = (
        "every event is in `events.jsonl`" if strict else "summary only — no event log was written"
    )

    lines = [
        f"# SuuClaw trace `{payload.get('trace_id') or '-'}`",
        "",
        f"- task: {payload.get('task') or '(not recorded)'}",
        f"- status: {payload.get('status') or '-'}"
        + (f" · last node: `{payload.get('latest_node')}`" if payload.get("latest_node") else ""),
        f"- workspace: `{payload.get('workspace') or '-'}`",
        f"- mode: `{payload.get('mode') or '-'}` — {mode_note}",
        f"- started: {payload.get('started_at') or '-'}",
        f"- ended: {payload.get('ended_at') or '-'} (+{duration_ms / 1000:.3f}s)",
    ]
    if payload.get("resumed"):
        lines.append("- resumed: yes — this trace continues an earlier run")

    lines += [
        "",
        "## Counters",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| node visits | {_as_int(payload.get('node_visits_total'))} |",
        f"| tool calls | {_as_int(payload.get('tool_calls'))} |",
        f"| failed tool calls | {_as_int(payload.get('failed_tool_calls'))} |",
        f"| approvals required | {_as_int(payload.get('approval_count'))} |",
        f"| checkpoints saved | {_as_int(payload.get('checkpoint_count'))} |",
        f"| delegations | {_as_int(payload.get('handoff_count'))} |",
        f"| timeline events | {_as_int(payload.get('timeline_total'))}{dropped_note} |",
    ]
    if visits:
        per_node = " · ".join(f"{name} ×{count}" for name, count in sorted(visits.items()))
        lines.append(f"| visits per node | {_row_cell(per_node)} |")

    lines += ["", "## Outcome", ""]
    if outcome:
        todos = outcome.get("todos")
        todos = todos if isinstance(todos, Mapping) else {}
        open_todos = ", ".join(str(item) for item in (outcome.get("open_todos") or [])) or "-"
        lines += [
            f"- passed: {outcome.get('passed')}",
            f"- attempts: {_as_int(outcome.get('attempts'))} "
            f"/ {_as_int(outcome.get('max_attempts'))}",
            f"- todos: {_as_int(todos.get('completed'))} / {_as_int(todos.get('total'))} completed",
            f"- open todos: {open_todos}",
            f"- last error: {str(outcome.get('last_error') or '').strip() or '-'}",
        ]
        answer = str(outcome.get("final_answer") or "").strip()
        if answer:
            lines += ["", "### Final answer on file", "", answer]
    else:
        lines.append("- nothing recorded about the outcome")

    lines += ["", "## Timeline", ""]
    if head or tail:
        if omitted:
            lines += _timeline_table(head)
            note = f"… {omitted} event(s) omitted from this file."
            if dropped:
                note += f" The opening {dropped} were dropped before the head was kept."
            lines += ["", note, ""]
            lines += _timeline_table(tail)
        else:
            lines += _timeline_table([*head, *tail])
    else:
        lines.append("_(no events were recorded)_")

    lines += [
        "",
        "## Files",
        "",
        f"- trace: `{paths.get('trace') or '-'}`",
        f"- timeline: `{paths.get('timeline') or '-'}`",
    ]
    if strict:
        lines.append(f"- events: `{paths.get('events') or '-'}`")
    else:
        lines.append("- events: not recorded — use trace mode `strict` for the full log")
    for message in payload.get("errors") or []:
        lines.append(f"- write error: {_row_cell(message)}")
    if payload.get("write_error"):
        lines += ["", f"> **This trace could not be written:** {_row_cell(payload['write_error'])}"]

    lines += [
        "",
        "## What a trace is not",
        "",
        "- It is a record, not a resume point: the `RECOVERY.md` beside the",
        "  checkpoint is what gets a dead run going again.",
        "- `+t` is wall-clock time since the run started, not CPU time.",
        "",
    ]
    return "\n".join(lines)
