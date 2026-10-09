"""Checkpointing: snapshot a run to disk so a crash costs minutes, not the run.

A LangGraph checkpointer (``InMemorySaver``, sqlite) restores a run *inside one
process tree*. This module solves the neighbouring problem: the process died, the
terminal is gone, and the user comes back later. So a checkpoint here is a
directory a human can read::

    <workspace>/.suuclaw/checkpoints/
        checkpoint.json   metadata + what the run had done   (light + strict)
        RECOVERY.md       how to get the run back, in prose  (light + strict)
        state.json        the full graph state               (strict only)
        events.jsonl      one line per event, appended       (strict only)

Three modes, one policy each:

* ``light`` — ``checkpoint.json`` + ``RECOVERY.md`` + a git snapshot of the
  workspace, written on every node switch. Cheap enough to do all the time: the
  graph state itself is *not* stored, so a resume restores the workspace and
  restarts the workflow from the task.
* ``strict`` — adds ``state.json`` (the whole graph state) and ``events.jsonl``
  (one line per event), so a resume also picks the plan, the todos and the
  attempt count back up mid-run.
* ``off`` — nothing is written and :meth:`CheckpointManager.save` returns ``None``.

Two properties this module refuses to give up:

* **A checkpoint never breaks the run it is recording.** ``save()`` swallows its
  own failures and reports them as a ``checkpoint_error`` event; a full disk must
  not kill a run that was otherwise fine.
* **A snapshot never touches anything outside the workspace.** The git snapshot
  only commits when the workspace *is* the repository root — a workspace nested
  inside somebody's larger repo would otherwise stage the user's own work into a
  checkpoint commit.

Layering: module scope imports the standard library and ``langchain_core`` only.
Everything else — ``tools.grep_tool``'s skip list, ``graph.workflow``'s seed
dict, ``graph.state``'s channel schema — is imported inside the function that
needs it, because :mod:`suuclaw.core.state` imports this module for its mode
constants and a module-scope import in the other direction would close the cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .paths import RUNTIME_DIRECTORY_NAMES, runtime_path

if TYPE_CHECKING:  # pragma: no cover - typing only, see the layering note above
    from .state import RuntimeState

__all__ = [
    "CHECKPOINT_DIR_PARTS",
    "DEFAULT_CHECKPOINT_MODE",
    "GIT_TIMEOUT_SECONDS",
    "MAX_MANIFEST_FILES",
    "MAX_RECOVERY_TABLE_ROWS",
    "MAX_STATE_MESSAGES",
    "VALID_CHECKPOINT_MODES",
    "CheckpointError",
    "CheckpointManager",
    "GitRestore",
    "GitSnapshot",
    "build_recovery_markdown",
    "normalize_checkpoint_mode",
    "restore_workspace_git",
    "resume_command",
    "snapshot_workspace_git",
]

#: The checkpoint tree, relative to the workspace. Part zero is also the one
#: directory the manifest and the git snapshot both exclude, so a checkpoint can
#: never record itself.
CHECKPOINT_DIR_PARTS = (".suuclaw", "checkpoints")

#: ``light`` is the default because it is the mode that costs almost nothing and
#: still answers "the run died — now what?"; ``off`` exists to say no outright.
VALID_CHECKPOINT_MODES: set[str] = {"light", "strict", "off"}
DEFAULT_CHECKPOINT_MODE = "light"

#: A save happens on every node switch, so every loop below is bounded: the file
#: list, the bytes hashed, the messages kept in ``state.json``.
MAX_MANIFEST_FILES = 2000
MAX_HASH_BYTES = 1024 * 1024
MAX_HASH_TOTAL_BYTES = 64 * 1024 * 1024
MAX_STATE_MESSAGES = 400
MAX_RECOVERY_TABLE_ROWS = 20

GIT_TIMEOUT_SECONDS = 30.0

#: Identity for the snapshot commits, applied per call with ``git -c`` so the
#: repository's own config is never rewritten.
_GIT_IDENTITY = ("-c", "user.name=SuuClaw Checkpoint", "-c", "user.email=suuclaw@localhost")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be used — missing, unreadable, or stale.

    Distinct from a failed *save*, which is reported as an event instead: a caller
    that asked to resume has no run to fall back on and needs to hear about it.
    """


def normalize_checkpoint_mode(mode: str | None) -> str:
    """Coerce ``mode`` to one of :data:`VALID_CHECKPOINT_MODES`.

    An unknown value falls back to :data:`DEFAULT_CHECKPOINT_MODE` rather than
    raising: the value arrives from a CLI flag, and refusing to *stop* a run
    because a typo said ``lite`` would be a poor trade.
    """
    if not mode:
        return DEFAULT_CHECKPOINT_MODE
    normalized = str(mode).strip().lower()
    return normalized if normalized in VALID_CHECKPOINT_MODES else DEFAULT_CHECKPOINT_MODE


def resume_command(workspace: str | os.PathLike[str]) -> str:
    """The command that resumes the run checkpointed for ``workspace``.

    Quoted only when the path needs it, so the common case stays copy-pasteable
    and the Windows case (``C:\\Users\\未闻花名\\...``) still works in PowerShell.

    Note:
        ``--resume`` is the contract this string advertises; the CLI flag itself
        is wired separately, so a command printed by this function only works
        once that flag exists.
    """
    text = str(workspace)
    if any(ch.isspace() for ch in text):
        text = f'"{text}"'
    return f"suuclaw --resume {text}"


def build_recovery_markdown(payload: Mapping[str, Any]) -> str:
    """Render the human-facing ``RECOVERY.md`` for a checkpoint ``payload``.

    Defensive on purpose: this file is read *when something has already gone
    wrong*, so a missing or unexpected key renders as a dash instead of raising
    and taking the checkpoint down with it. Older payloads stay readable.
    """
    workspace = str(payload.get("workspace") or "")
    mode = str(payload.get("mode") or "")
    strict = bool(payload.get("strict"))
    git = payload.get("git") or {}
    manifest = payload.get("manifest") or {}
    todos = payload.get("todos") or {}
    context = payload.get("context") or {}
    resume = payload.get("resume") or {}
    command = str(resume.get("command") or (resume_command(workspace) if workspace else ""))

    lines = [
        "# SuuClaw recovery guide",
        "",
        f"- checkpoint: `{payload.get('id') or '-'}`",
        f"- saved: {payload.get('created_at') or '-'}",
        f"- mode: `{mode}` — "
        + (
            "full graph state is on disk, a resume continues mid-run"
            if strict
            else "workspace snapshot only, a resume restarts the workflow"
        ),
        f"- status: {payload.get('status') or '-'}"
        + (f" · node: `{payload.get('node')}`" if payload.get("node") else ""),
        f"- attempts: {payload.get('attempts', 0)} / {payload.get('max_attempts', 0)}",
        "",
        "## Task",
        "",
        str(payload.get("task") or "(not recorded)"),
        "",
        "## Where the run had got to",
        "",
        f"- passed: {payload.get('passed')}",
        f"- open todos: {', '.join(payload.get('open_todos') or []) or '-'}",
        f"- todos: {todos.get('completed', 0)} completed / {todos.get('total', 0)} total",
        f"- context: {context.get('tokens', 0)} / {context.get('limit', 0)} tokens",
        f"- last error: {str(payload.get('last_error') or '').strip() or '-'}",
    ]
    final = str(payload.get("final_answer") or "").strip()
    if final:
        lines += ["", "## Last answer on file", "", final]

    lines += [
        "",
        "## Resume",
        "",
        "```bash",
        command or "suuclaw --resume <workspace>",
        "```",
        "",
    ]
    lines += [
        "Run it from the project directory, or add `--workspace` yourself if you use",
        "the long form:",
        "",
        "```bash",
        f"suuclaw --workspace {workspace or '<workspace>'}",
        "```",
        "",
        "## Files",
        "",
        f"- workspace: `{workspace or '-'}`",
        f"- files recorded: {manifest.get('count', 0)}"
        + (" (list truncated)" if manifest.get("truncated") else ""),
        f"- git commit: `{str(git.get('commit') or '')[:12] or '-'}`"
        + (f" — {git['note']}" if git.get("note") else ""),
    ]
    if strict:
        lines.append(f"- full state: `{payload.get('state_path') or '-'}`")
        lines.append(f"- event log: `{payload.get('events_path') or '-'}`")
    else:
        lines.append("- full state: not saved in `light` mode")

    rows = list(manifest.get("files") or [])
    if rows:
        lines += [
            "",
            f"### File list (first {min(len(rows), MAX_RECOVERY_TABLE_ROWS)})",
            "",
            "| path | bytes | modified |",
            "| --- | --- | --- |",
        ]
        for row in rows[:MAX_RECOVERY_TABLE_ROWS]:
            lines.append(
                f"| `{row.get('path', '')}` | {row.get('size', 0)} | {row.get('modified', '')} |"
            )
        hidden = len(rows) - MAX_RECOVERY_TABLE_ROWS
        if hidden > 0:
            lines += ["", f"…and {hidden} more, listed in `checkpoint.json`."]

    lines += [
        "",
        "## What a resume does and does not do",
        "",
        "- It restores files from the git commit above into the workspace, and it",
        "  leaves files created *after* the checkpoint in place — a recovery that",
        "  deleted the user's work would be worse than no recovery.",
        "- Files already modified since the checkpoint are overwritten by the",
        "  checkpoint's version. Copy anything you edited by hand before resuming.",
        "- `light` mode has no graph state, so the run restarts from the task with",
        "  the restored workspace; `strict` mode also restores the plan, the todos",
        "  and the attempt count.",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class GitSnapshot:
    """Outcome of :func:`snapshot_workspace_git`.

    ``commit`` is empty whenever the snapshot did not happen, and ``note`` then
    says why — in prose, because this string ends up in ``RECOVERY.md`` next to a
    dash where a commit hash was expected.
    """

    commit: str = ""
    repo_root: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.commit)


@dataclass(frozen=True)
class GitRestore:
    """Outcome of :func:`restore_workspace_git`."""

    files: int = 0
    note: str = ""

    @property
    def ok(self) -> bool:
        return not self.note


def _run_git(
    workspace: Path,
    *args: str,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | None:
    """Run ``git -C workspace <args>``; ``None`` means git never ran.

    ``-C`` (and any ``-c`` in ``args``) must precede the subcommand, which is why
    callers pass flags through ``args`` rather than as a separate argument.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(workspace), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _same_path(left: Path, right: Path) -> bool:
    """True when two paths point at the same directory, case aside."""
    try:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))
    except OSError:  # pragma: no cover - unresolvable path
        return False


def _snapshot_target(workspace: Path) -> GitSnapshot | None:
    """The checks both git helpers share; ``None`` means "go ahead"."""
    probe = _run_git(workspace, "rev-parse", "--show-toplevel")
    if probe is None:
        return GitSnapshot(note="git is not available, so only the file manifest was recorded")
    if probe.returncode != 0:
        return GitSnapshot(
            note="the workspace is not a git repository, so only the file manifest was recorded"
        )
    top = probe.stdout.strip()
    if not _same_path(Path(top), workspace):
        return GitSnapshot(
            repo_root=top,
            note=(
                f"the workspace is inside the repository at {top}, which this refuses to "
                "commit to — a checkpoint must not stage files outside its own workspace"
            ),
        )
    return None


def snapshot_workspace_git(workspace: str | os.PathLike[str], *, message: str) -> GitSnapshot:
    """Commit the workspace so a checkpoint also captures file *contents*.

    Only commits when the workspace **is** the repository root: running
    ``git add -A`` for a workspace that sits inside a larger repo would stage the
    user's own unrelated work. The checkpoint directory keeps itself out of the
    snapshot with a ``.gitignore`` of its own, so a save never commits the
    bookkeeping of the previous save.
    """
    root = Path(workspace)
    blocked = _snapshot_target(root)
    if blocked is not None:
        return blocked

    _write_gitignore(root / CHECKPOINT_DIR_PARTS[0])
    for directory in RUNTIME_DIRECTORY_NAMES:
        if (root / directory).is_dir():
            _write_gitignore(root / directory)

    staged = _run_git(root, "add", "-A")
    if staged is None or staged.returncode != 0:
        detail = (staged.stderr or staged.stdout).strip() if staged else "git timed out"
        return GitSnapshot(note=f"git add failed: {detail or 'unknown error'}")

    status = _run_git(root, "status", "--porcelain")
    if status is not None and status.returncode == 0 and not status.stdout.strip():
        head = _run_git(root, "rev-parse", "HEAD")
        commit = head.stdout.strip() if head and head.returncode == 0 else ""
        return GitSnapshot(
            commit=commit, repo_root=str(root), note="no changes since the last snapshot"
        )

    committed = _run_git(root, *_GIT_IDENTITY, "commit", "-m", message)
    if committed is None or committed.returncode != 0:
        detail = (committed.stderr or committed.stdout).strip() if committed else "git timed out"
        return GitSnapshot(note=f"git commit failed: {detail or 'unknown error'}")

    head = _run_git(root, "rev-parse", "HEAD")
    commit = head.stdout.strip() if head and head.returncode == 0 else ""
    return GitSnapshot(commit=commit, repo_root=str(root))


def restore_workspace_git(
    workspace: str | os.PathLike[str],
    commit: str,
) -> GitRestore:
    """Bring back the files a checkpoint commit recorded.

    Uses ``git checkout <commit> -- .``, not a reset and definitely not a
    ``git clean``: overwriting files is recovering, deleting files is not. A run
    that crashed *after* creating something does not lose that something, and
    nothing outside the workspace is ever touched.
    """
    root = Path(workspace)
    blocked = _snapshot_target(root)
    if blocked is not None:
        return GitRestore(note=blocked.note)

    known = _run_git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if known is None or known.returncode != 0:
        return GitRestore(note=f"commit {commit[:12]} is not in this repository any more")

    listed = _run_git(root, "ls-tree", "-r", "--name-only", commit)
    raw_names = listed.stdout.splitlines() if listed else []
    names = [line.strip() for line in raw_names if line.strip()]
    # The checkpoint tree is excluded from the count the same way it is excluded
    # from the manifest: this number is shown to the user as "your files", and a
    # bookkeeping file of ours padding it by one is a number nobody can explain.
    prefixes = tuple(f"{directory}/" for directory in RUNTIME_DIRECTORY_NAMES)
    files = len([name for name in names if not name.startswith(prefixes)])

    restored = _run_git(root, "checkout", commit, "--", ".")
    if restored is None or restored.returncode != 0:
        detail = (restored.stderr or restored.stdout).strip() if restored else "git timed out"
        return GitRestore(note=f"git checkout failed: {detail or 'unknown error'}")

    return GitRestore(files=files)


def _write_gitignore(directory: Path) -> None:
    """Keep ``directory`` out of the workspace's git snapshots.

    Written on every save so the ignore survives a user who deletes it, and it
    also keeps the checkpoint tree out of the *user's* own ``git status``.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ".gitignore"
        wanted = "*\n!.gitignore\n"
        if not target.exists() or target.read_text(encoding="utf-8", errors="replace") != wanted:
            target.write_text(wanted, encoding="utf-8")
    except OSError:
        pass


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    """Serialise one LangChain message into plain JSON-safe data."""
    from .agent import message_text  # lazy: state -> checkpoint -> agent -> state

    payload: dict[str, Any] = {
        "type": type(message).__name__,
        "role": str(getattr(message, "type", "") or ""),
        "content": message_text(getattr(message, "content", "")),
    }
    tool_call_id = getattr(message, "tool_call_id", "")
    if tool_call_id:
        payload["tool_call_id"] = str(tool_call_id)
    calls = getattr(message, "tool_calls", None)
    if calls:
        payload["tool_calls"] = [
            {
                "name": str(call.get("name") or ""),
                "args": _jsonable(call.get("args")),
                "id": str(call.get("id") or ""),
            }
            for call in calls
            if isinstance(call, Mapping)
        ]
    return payload


def _message_from_payload(payload: Mapping[str, Any]) -> BaseMessage | None:
    """Rebuild a message from :func:`_message_payload`; ``None`` when unreadable.

    Only the fields the trace needs are revived. ``messages`` is a record the
    nodes append to and never read (see
    :class:`~suuclaw.graph.state.SuuClawGraphState`), so a message that comes
    back slightly thinner than it went out costs nothing.
    """
    kind = str(payload.get("type") or "")
    content = str(payload.get("content") or "")
    if kind == HumanMessage.__name__:
        return HumanMessage(content=content)
    if kind == AIMessage.__name__:
        calls = payload.get("tool_calls") or []
        return AIMessage(
            content=content,
            tool_calls=[
                {
                    "name": str(call.get("name") or ""),
                    "args": call.get("args") if isinstance(call.get("args"), dict) else {},
                    "id": str(call.get("id") or ""),
                    "type": "tool_call",
                }
                for call in calls
                if isinstance(call, Mapping)
            ],
        )
    if kind == ToolMessage.__name__:
        return ToolMessage(content=content, tool_call_id=str(payload.get("tool_call_id") or ""))
    if kind == SystemMessage.__name__:
        return SystemMessage(content=content)
    return None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a graph-state value into JSON-safe data.

    Never raises and never drops a key: an unserialisable object becomes its
    ``repr``, which is ugly but is still evidence. A checkpoint that raised here
    would be a checkpoint that broke the run it was recording.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseMessage):
        return _message_payload(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _iso_from_timestamp(value: float) -> str:
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):  # pragma: no cover - odd clocks
        return ""


def _shorten(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` characters with a visible marker."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} more character(s) withheld]"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 256), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


class CheckpointManager:
    """Writes and reads one workspace's checkpoint tree.

    The manager is intentionally dumb about the graph: it receives a state
    mapping and records what it can read from it, so a channel added to
    :class:`~suuclaw.graph.state.SuuClawGraphState` later shows up in
    ``state.json`` automatically (a strict save stores the whole thing, filtered
    by the channel schema on the way back in).
    """

    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.workspace = Path(runtime.workspace)
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        self.task = task or ""
        #: Where every file below lives. Part zero is also what the manifest and
        #: the git snapshot exclude, so a checkpoint never records itself.
        self.root = runtime_path(self.workspace, "checkpoints", marker="checkpoint.json")
        #: Why the last save failed, if it did. ``save()`` reports the same text
        #: through its event; this is for callers that log instead of render.
        self.last_error = ""

    # -- layout -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether this manager writes anything at all."""
        return self.mode != "off"

    @property
    def checkpoint_json(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def state_json(self) -> Path:
        return self.root / "state.json"

    @property
    def events_jsonl(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def recovery_md(self) -> Path:
        return self.root / "RECOVERY.md"

    # -- writing ----------------------------------------------------------
    def save(
        self,
        state: Mapping[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Record ``state``; return the event describing it, or ``None`` if disabled.

        Never raises. A failure comes back as a ``checkpoint_error`` event rather
        than an exception, because the alternative is losing a run that was
        working fine except for its bookkeeping.

        Args:
            state: The graph state as of this moment (a mapping of channels).
            status: Free-form run status — ``running`` while nodes work, whatever
                the driver calls the end state otherwise.
            latest_node: The node this save is attached to, if any.
            event: The event being processed, appended to ``events.jsonl`` in
                strict mode.
        """
        if not self.enabled:
            return None
        try:
            return self._save(state, status=status, latest_node=latest_node, event=event)
        except Exception as exc:  # deliberate: bookkeeping must not kill the run
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "type": "checkpoint_error",
                "node": latest_node or "",
                "loop": 0,
                "mode": self.mode,
                "workspace": str(self.workspace),
                "message": self.last_error,
            }

    def _save(
        self,
        state: Mapping[str, Any],
        *,
        status: str,
        latest_node: str | None,
        event: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        checkpoint_id = f"ckpt-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:4]}"
        strict = self.mode == "strict"
        self.root.mkdir(parents=True, exist_ok=True)

        if strict:
            # Both of these are append-or-overwrite on purpose: the event log is
            # the *sequence*, state.json is only ever the latest.
            self._append_event(
                event, status=status, node=latest_node, at=now, checkpoint_id=checkpoint_id
            )
            self._write_json(self.state_json, self._state_payload(state, at=now))

        manifest = self._manifest()
        git = snapshot_workspace_git(
            self.workspace, message=f"suuclaw checkpoint {checkpoint_id}"
        )

        payload: dict[str, Any] = {
            "id": checkpoint_id,
            "created_at": now.isoformat(timespec="seconds"),
            "mode": self.mode,
            "strict": strict,
            "status": status,
            "node": latest_node or "",
            "task": str(state.get("task") or self.task or ""),
            "workspace": str(self.workspace),
            "checkpoint_path": str(self.checkpoint_json),
            "recovery_path": str(self.recovery_md),
            "state_path": str(self.state_json) if strict else "",
            "events_path": str(self.events_jsonl) if strict else "",
            "attempts": _as_int(state.get("attempts")),
            "max_attempts": _as_int(state.get("max_attempts")),
            "passed": bool(state.get("passed")),
            "last_error": str(state.get("last_error") or ""),
            "open_todos": _open_todos(state),
            "todos": _todo_counts(state),
            "verification": _verification_summary(state),
            "context": _context_summary(state),
            "messages": len(state.get("messages") or []),
            "final_answer": _shorten(str(state.get("final_answer") or ""), 2000),
            "next_node": str(state.get("context_next_node") or ""),
            "manifest": manifest,
            "git": {"commit": git.commit, "repo_root": git.repo_root, "note": git.note},
            "resume": {"command": resume_command(self.workspace), "workspace": str(self.workspace)},
        }
        self._write_json(self.checkpoint_json, payload)
        self.recovery_md.write_text(build_recovery_markdown(payload), encoding="utf-8")
        self.last_error = ""

        return {
            "type": "checkpoint_saved",
            "node": latest_node or "",
            "loop": 0,
            "id": checkpoint_id,
            "mode": self.mode,
            "status": status,
            "path": str(self.checkpoint_json),
            "recovery_path": str(self.recovery_md),
            "attempts": payload["attempts"],
            "files": manifest["count"],
            "git_commit": git.commit,
            "git_note": git.note,
            "strict": strict,
        }

    def _state_payload(self, state: Mapping[str, Any], *, at: datetime) -> dict[str, Any]:
        """The full state, minus what cannot be serialised and cannot be replayed.

        ``runtime`` is dropped and replaced by a descriptor: it is a live object
        holding a workspace path and policy values, and the resume path takes a
        fresh one from the caller. ``messages`` are capped by recency — a long run
        produces thousands, and ``state.json`` is meant to be openable.
        """
        messages = [item for item in (state.get("messages") or []) if isinstance(item, BaseMessage)]
        kept = messages[-MAX_STATE_MESSAGES:]
        payload = {
            str(key): _jsonable(value)
            for key, value in state.items()
            if key not in {"runtime", "messages"}
        }
        payload["messages"] = [_message_payload(message) for message in kept]
        payload["messages_dropped"] = len(messages) - len(kept)
        payload["saved_at"] = at.isoformat(timespec="seconds")
        payload["runtime"] = _runtime_descriptor(state.get("runtime"), workspace=self.workspace)
        return payload

    def _manifest(self) -> dict[str, Any]:
        """List the workspace's files, excluding the checkpoint tree and caches."""
        from ..tools.grep_tool import SKIP_DIRS  # lazy: core must not import tools at import time

        skip = SKIP_DIRS | RUNTIME_DIRECTORY_NAMES
        files: list[dict[str, Any]] = []
        scanned = 0
        total_bytes = 0
        hashed_bytes = 0

        for dirpath, dirnames, filenames in os.walk(self.workspace):
            dirnames[:] = sorted(name for name in dirnames if name not in skip)
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                try:
                    stat = path.stat()
                except OSError:
                    continue
                scanned += 1
                total_bytes += stat.st_size
                if len(files) >= MAX_MANIFEST_FILES:
                    continue
                entry: dict[str, Any] = {
                    "path": path.relative_to(self.workspace).as_posix(),
                    "size": stat.st_size,
                    "modified": _iso_from_timestamp(stat.st_mtime),
                    "sha256": "",
                }
                fits = stat.st_size <= MAX_HASH_BYTES
                if fits and hashed_bytes + stat.st_size <= MAX_HASH_TOTAL_BYTES:
                    entry["sha256"] = _sha256(path)
                    hashed_bytes += stat.st_size
                files.append(entry)

        return {
            "count": len(files),
            "scanned": scanned,
            "truncated": len(files) < scanned,
            "total_bytes": total_bytes,
            "files": files,
        }

    def _append_event(
        self,
        event: Mapping[str, Any] | None,
        *,
        status: str,
        node: str | None,
        at: datetime,
        checkpoint_id: str,
    ) -> None:
        """Append one line to ``events.jsonl`` — the event, or a save marker."""
        record: dict[str, Any] = {
            "at": at.isoformat(timespec="seconds"),
            "checkpoint": checkpoint_id,
            "status": status,
            "node": node or "",
        }
        if event is None:
            record["type"] = "checkpoint"
        else:
            record["event"] = _jsonable(event)
        try:
            with self.events_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            raise CheckpointError(f"cannot append to {self.events_jsonl}: {exc}") from exc

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Write ``payload`` atomically, so a crash cannot leave half a file.

        ``os.replace`` is what makes the checkpoint readable *while* it is being
        rewritten — the point of the feature is surviving a crash, so the file
        must never be observed truncated.
        """
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except OSError as exc:
            raise CheckpointError(f"cannot write {path}: {exc}") from exc
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass

    # -- reading ----------------------------------------------------------
    def read_payload(self) -> dict[str, Any]:
        """Read ``checkpoint.json``, or raise :class:`CheckpointError`."""
        if not self.checkpoint_json.is_file():
            raise CheckpointError(
                f"no checkpoint at {self.checkpoint_json}; "
                "run the task without --resume to create one"
            )
        try:
            payload = json.loads(self.checkpoint_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"cannot read {self.checkpoint_json}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointError(f"{self.checkpoint_json} does not hold a checkpoint object")
        # Present current locations after directory/workspace migration. This
        # updates the in-memory view only; the saved evidence is left untouched.
        payload["workspace"] = str(self.workspace)
        payload["checkpoint_path"] = str(self.checkpoint_json)
        payload["recovery_path"] = str(self.recovery_md)
        if payload.get("strict"):
            payload["state_path"] = str(self.state_json)
            payload["events_path"] = str(self.events_jsonl)
        payload["resume"] = {
            "command": resume_command(self.workspace), "workspace": str(self.workspace)
        }
        return payload

    @classmethod
    def load_resume_inputs(
        cls,
        runtime: RuntimeState,
        task: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Rebuild everything a driver needs to continue the checkpointed run.

        Returns:
            ``(inputs, resume_event)`` — ``inputs`` is what
            :func:`~suuclaw.core.agent.iter_workflow_events` takes (the same
            seed shape as :func:`~suuclaw.graph.workflow.initial_state`), and
            ``resume_event`` is the event to show the user.

        Raises:
            CheckpointError: no checkpoint file, or one that is not a checkpoint
                object. A checkpoint that is merely *stale* — a commit git no
                longer knows, a workspace that has moved — is not an error: the
                event's ``note`` says what could not be restored, so the run still
                continues with whatever is on disk.

        The budget rule: a resume always allows at least one more attempt than the
        checkpoint had already spent, so ``max_attempts`` smaller than the spent
        count cannot turn "resume" into "immediately report budget exhausted".
        """
        manager = cls(runtime, task or "")
        payload = manager.read_payload()

        resume_task = str(task or payload.get("task") or "")
        git = payload.get("git") or {}
        commit = str(git.get("commit") or "")
        if commit:
            restore = restore_workspace_git(manager.workspace, commit)
        else:
            restore = GitRestore(
                note="the checkpoint recorded no commit, so no files were restored"
            )

        from ..graph.workflow import initial_state  # lazy: graph imports core.state

        restored: dict[str, Any] = {}
        messages_restored = 0
        if bool(payload.get("strict")):
            # The state belongs beside this checkpoint. Recorded absolute paths
            # can become stale after workspace or metadata directory migration.
            restored = manager._restored_channels(manager.state_json)
            messages = restored.get("messages") or []
            messages_restored = len(messages) if isinstance(messages, list) else 0

        spent = _as_int(restored.get("attempts")) if restored else 0
        budget = max(_as_int(max_attempts, default=3), spent + 1)
        inputs = initial_state(resume_task, runtime, max_attempts=budget)
        inputs.update(restored)
        inputs["task"] = resume_task
        inputs["runtime"] = runtime
        inputs["max_attempts"] = budget

        event = {
            "type": "checkpoint_resumed",
            "node": str(payload.get("node") or ""),
            "loop": 0,
            "id": str(payload.get("id") or ""),
            "mode": str(payload.get("mode") or ""),
            "saved_at": str(payload.get("created_at") or ""),
            "workspace": str(manager.workspace),
            "task": resume_task,
            "attempts": _as_int(inputs.get("attempts")),
            "max_attempts": budget,
            "state_restored": bool(restored),
            "messages_restored": messages_restored,
            "files_restored": restore.files,
            "git_commit": commit,
            "note": restore.note,
            "path": str(manager.checkpoint_json),
        }
        return inputs, event

    def _restored_channels(self, state_path: Path) -> dict[str, Any]:
        """Filter a saved state down to real channels and revive the messages.

        The filter is the channel schema itself, so a state file written by an
        older build cannot inject a key the graph does not know. Anything
        unreadable degrades to "restore the workspace only" instead of failing:
        in ``light`` terms, that is still a useful resume.
        """
        from ..graph.state import SuuClawGraphState  # lazy: graph imports core.state

        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, Mapping):
            return {}

        allowed = set(SuuClawGraphState.__annotations__) - {"runtime"}
        restored: dict[str, Any] = {key: value for key, value in raw.items() if key in allowed}
        restored.pop("messages_dropped", None)
        restored.pop("saved_at", None)

        messages: list[BaseMessage] = []
        for item in raw.get("messages") or []:
            if not isinstance(item, Mapping):
                continue
            message = _message_from_payload(item)
            if message is not None:
                messages.append(message)
        if messages:
            restored["messages"] = messages
        else:
            restored.pop("messages", None)
        return restored


# --------------------------------------------------------------------------- #
# payload helpers — small readers over the state mapping
# --------------------------------------------------------------------------- #
def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _todo_rows(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in (state.get("todos") or []) if isinstance(row, Mapping)]


def _todo_counts(state: Mapping[str, Any]) -> dict[str, int]:
    rows = _todo_rows(state)
    counts = {"total": len(rows), "pending": 0, "in_progress": 0, "completed": 0, "blocked": 0}
    for row in rows:
        status = str(row.get("status") or "pending")
        if status in counts:
            counts[status] += 1
    return counts


def _open_todos(state: Mapping[str, Any], *, limit: int = 5) -> list[str]:
    rows = [row for row in _todo_rows(state) if str(row.get("status") or "") != "completed"]
    return [_shorten(str(row.get("content") or ""), 200) for row in rows[:limit]]


def _verification_summary(state: Mapping[str, Any]) -> dict[str, int]:
    results = [row for row in (state.get("verification_results") or []) if isinstance(row, Mapping)]
    checks = [row for row in (state.get("verification_checks") or []) if isinstance(row, Mapping)]
    return {
        "commands": len(state.get("verification_commands") or []),
        "results": len(results),
        "failed": len([row for row in results if not row.get("ok")]),
        "checks": len(checks),
        "checks_failed": len([row for row in checks if not row.get("passed")]),
    }


def _context_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tokens": _as_int(state.get("context_token_count")),
        "limit": _as_int(state.get("context_token_limit")),
        "should_compress": bool(state.get("context_should_compress")),
        "compressions": len(state.get("compression_events") or []),
    }


def _runtime_descriptor(runtime: Any, *, workspace: Path) -> dict[str, Any]:
    """What a reader needs to know about the sandbox policy of the saved run.

    The live object is not serialised — the resume path builds a real one from the
    caller's runtime — but the *policy* it carried is worth recording, because a
    later run with different settings explains why the checkpoint resumes
    differently than it did.
    """
    if runtime is None:
        return {"workspace": str(workspace)}
    return {
        "workspace": str(getattr(runtime, "workspace", workspace)),
        "shell": getattr(runtime, "shell", None),
        "command_timeout": getattr(runtime, "command_timeout", None),
        "max_output_chars": getattr(runtime, "max_output_chars", None),
        "approval_mode": getattr(runtime, "approval_mode", None),
        "checkpoint_mode": getattr(runtime, "checkpoint_mode", None),
    }
