"""Session management: the cross-run conversation log under ``.suuclaw/``.

A checkpoint solves "the run died — what had it done?"; a session solves the
neighbouring question the entry graph (:func:`~suuclaw.graph.workflow.
build_entry_workflow`) needs answered: **has this user been here before, and
what were we talking about?** A short follow-up like "继续" or "修一下" only
routes correctly when the intent router can see the prior turns, so the session
is the *session context* those cheap nodes read::

    <workspace>/.suuclaw/session/
        session.json        the data: session_id, turn_index, recent_turns
        SESSION_SUMMARY.md  the same thing a human can open and read

The ``session.json`` shape::

    {
      "session_id": "uuid",
      "turn_index": 3,
      "recent_turns": [
        {"turn": 1, "role": "user", "content": "...", "timestamp": "..."},
        {"turn": 2, "role": "assistant", "route": "workflow",
         "content": "...", "summary": "..."},
        {"turn": 3, "role": "user", "content": "...", "timestamp": "..."}
      ],
      "created_at": "...", "updated_at": "..."
    }

Turn numbering: :func:`append_user_turn` mints the number (``turn_index``
increments and is returned); :func:`append_assistant_turn` *takes* it, so a
reply is always attached to the input it answers. An assistant turn carries
``route`` (``chat`` / ``workflow``) because "what route did the last few
inputs take" is exactly the signal a continuation needs.

Two properties inherited from :mod:`suuclaw.core.checkpoint`:

* **Bookkeeping never breaks the run.** :func:`save_session` reports failures
  in its receipt (``error``) instead of raising; :func:`load_or_create_session`
  starts fresh over a file it cannot read rather than crashing the front end.
* **Nothing leaves the workspace.** The session tree lives under
  ``<workspace>/.suuclaw/``, the file listing :func:`build_session_context`
  shows excludes that tree, and nothing here touches a path outside it.

Layering: module scope imports the standard library only. ``tools.grep_tool``'s
skip list — the same one the checkpoint manifest uses — is imported inside the
function that needs it, keeping ``core`` importable without the tool package.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import RUNTIME_DIRECTORY_NAMES, runtime_path

__all__ = [
    "MAX_CONTEXT_TURNS",
    "MAX_RECENT_FILES",
    "MAX_RECENT_TURNS",
    "MAX_SESSION_CONTEXT",
    "MAX_TURN_CONTENT",
    "SESSION_FILE",
    "SESSION_ROOT",
    "SESSION_SUMMARY_FILE",
    "VALID_ROUTES",
    "append_assistant_turn",
    "append_user_turn",
    "build_session_context",
    "load_or_create_session",
    "save_session",
    "session_dir",
    "session_path",
    "session_summary_path",
]

#: The session tree, relative to the workspace — the sibling of the checkpoint
#: and trace trees, so one ``.suuclaw/`` directory stays the whole bookkeeping.
SESSION_ROOT = ".suuclaw/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"

#: Budgets. The context is what two cheap nodes embed into *every* prompt, so it
#: is deliberately small; the per-turn cap keeps one pasted log from eating it.
MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000

#: How much history each view keeps. ``recent_turns`` in the file (50) is a
#: superset of what the context builder shows (10), so trimming the file never
#: silently changes what the prompts see.
MAX_RECENT_TURNS = 50
MAX_CONTEXT_TURNS = 10
MAX_RECENT_FILES = 30

#: The only routes an assistant turn may carry; anything else degrades to "".
VALID_ROUTES: set[str] = {"chat", "workflow"}


def session_dir(workspace: str | os.PathLike[str]) -> Path:
    """The canonical session tree, migrating legacy metadata when present."""
    return runtime_path(workspace, "session", marker=SESSION_FILE)


def session_path(workspace: str | os.PathLike[str]) -> Path:
    """Where ``session.json`` lives for ``workspace``."""
    return session_dir(workspace) / SESSION_FILE


def session_summary_path(workspace: str | os.PathLike[str]) -> Path:
    """Where ``SESSION_SUMMARY.md`` lives for ``workspace``."""
    return session_dir(workspace) / SESSION_SUMMARY_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _shorten(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` characters with a visible marker."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} more character(s) withheld]"


def _blank_session() -> dict[str, Any]:
    """A fresh, empty session dict — the shape every caller can rely on."""
    now = _now()
    return {
        "session_id": str(uuid4()),
        "turn_index": 0,
        "recent_turns": [],
        "created_at": now,
        "updated_at": now,
    }


def _read_session(workspace: Path) -> dict[str, Any] | None:
    """Load ``session.json``; ``None`` when missing or unreadable."""
    path = session_path(workspace)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


# --------------------------------------------------------------------------- #
# loading / appending
# --------------------------------------------------------------------------- #
def load_or_create_session(workspace: str | os.PathLike[str]) -> dict[str, Any]:
    """Load ``session.json`` for ``workspace``, creating it if there is none.

    A missing *or corrupt* session both start fresh: the file is a conversation
    log, and refusing to run because yesterday's log had a typo in it would put
    bookkeeping above the work. The workspace directory itself is created when
    missing, matching :func:`suuclaw.core.paths.ensure_workspace`.
    """
    root = Path(workspace)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot use workspace {root}: {exc}") from exc

    payload = _read_session(root)
    if payload is None:
        session = _blank_session()
        save_session(root, session)
        return session

    # Repair rather than trust: every key a caller may read is defaulted here,
    # so an older or hand-edited file still yields the documented shape.
    session = _blank_session()
    session["session_id"] = str(payload.get("session_id") or session["session_id"])
    try:
        session["turn_index"] = max(0, int(payload.get("turn_index")))
    except (TypeError, ValueError):
        pass
    turns = payload.get("recent_turns")
    session["recent_turns"] = (
        [turn for turn in turns if isinstance(turn, dict)] if isinstance(turns, list) else []
    )
    session["created_at"] = str(payload.get("created_at") or session["created_at"])
    session["updated_at"] = str(payload.get("updated_at") or session["updated_at"])
    return session


def append_user_turn(session: dict[str, Any], content: str) -> int:
    """Record one user input; return the turn number it was given.

    The number is minted here — ``turn_index`` increments and the same value
    goes into the new row — so a caller can hand it straight to
    :func:`append_assistant_turn` when the reply comes back.
    """
    session["turn_index"] = _as_int(session.get("turn_index")) + 1
    turn = session["turn_index"]
    session.setdefault("recent_turns", []).append(
        {
            "turn": turn,
            "role": "user",
            "content": _shorten(str(content or ""), MAX_TURN_CONTENT),
            "timestamp": _now(),
        }
    )
    return turn


def append_assistant_turn(
    session: dict[str, Any],
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    """Record one assistant reply against the ``turn`` it answers.

    ``route`` is ``"chat"`` or ``"workflow"``; anything else is stored as an
    empty string rather than guessing. ``summary`` is the short version the
    context builder prefers — the full ``content`` stays on file either way.
    """
    normalized = str(route or "").strip().lower()
    session.setdefault("recent_turns", []).append(
        {
            "turn": _as_int(turn),
            "role": "assistant",
            "route": normalized if normalized in VALID_ROUTES else "",
            "content": _shorten(str(content or ""), MAX_TURN_CONTENT),
            "summary": _shorten(str(summary or ""), MAX_TURN_CONTENT),
            "timestamp": _now(),
        }
    )


# --------------------------------------------------------------------------- #
# saving
# --------------------------------------------------------------------------- #
def save_session(workspace: str | os.PathLike[str], session: dict[str, Any]) -> dict[str, Any]:
    """Write ``session.json`` and regenerate ``SESSION_SUMMARY.md``.

    Returns a receipt — paths, turn counts and an ``error`` field that is empty
    on success. Never raises: the session is bookkeeping, and a read-only disk
    must not take down a run that was otherwise fine. ``updated_at`` is stamped
    here (the single writer) and ``recent_turns`` is trimmed to
    :data:`MAX_RECENT_TURNS` before the file is written.
    """
    root = Path(workspace)
    receipt: dict[str, Any] = {
        "path": str(session_path(root)),
        "summary_path": str(session_summary_path(root)),
        "session_id": str(session.get("session_id") or ""),
        "turn_index": _as_int(session.get("turn_index")),
        "turns_kept": 0,
        "error": "",
    }

    session["updated_at"] = _now()
    turns = [turn for turn in (session.get("recent_turns") or []) if isinstance(turn, dict)]
    session["recent_turns"] = turns[-MAX_RECENT_TURNS:]
    receipt["turns_kept"] = len(session["recent_turns"])

    try:
        directory = session_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(session_path(root), session)
        (session_summary_path(root)).write_text(build_summary_markdown(session), encoding="utf-8")
    except OSError as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    return receipt


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` atomically — the checkpoint's trick, same reason."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass


def build_summary_markdown(session: Mapping[str, Any]) -> str:
    """Render the human-readable ``SESSION_SUMMARY.md`` for ``session``.

    Defensive like ``RECOVERY.md``: a missing or odd field renders as a dash
    instead of taking the summary down, because this file is what a user opens
    when they want to know "what did we talk about?".
    """
    turns = [turn for turn in (session.get("recent_turns") or []) if isinstance(turn, Mapping)]
    lines = [
        "# SuuClaw session summary",
        "",
        f"- session: `{session.get('session_id') or '-'}`",
        f"- turns: {len(turns)} (index at {_as_int(session.get('turn_index'))})",
        f"- created: {session.get('created_at') or '-'}",
        f"- updated: {session.get('updated_at') or '-'}",
        "",
    ]
    if not turns:
        lines += ["No turns recorded yet.", ""]
        return "\n".join(lines)

    lines.append("## Conversation")
    for turn in turns[-MAX_RECENT_TURNS:]:
        role = str(turn.get("role") or "-")
        heading = f"### Turn {_as_int(turn.get('turn'))} · {role}"
        if role == "assistant":
            heading += f" · {turn.get('route') or '-'}"
        lines += [heading, f"- at: {turn.get('timestamp') or '-'}", ""]
        body = str(turn.get("summary") or "") if role == "assistant" else ""
        body = body or _shorten(str(turn.get("content") or ""), MAX_TURN_CONTENT)
        lines += [body or "(empty)", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the prompt-facing context
# --------------------------------------------------------------------------- #
def build_session_context(
    workspace: str | os.PathLike[str],
    session: Mapping[str, Any] | None = None,
) -> str:
    """Assemble the session context string the cheap entry nodes embed.

    Three sections, capped in that order:

    * header — session id and turn index (always present, tiny);
    * the workspace's 30 most recently modified files, newest first — the
      "is this a continuation of prior coding work" signal;
    * the last :data:`MAX_CONTEXT_TURNS` turns — user inputs verbatim (already
      capped per turn), assistant replies as their summary, content as fallback.

    Pass ``session`` explicitly when the caller already holds it; without one
    the file is *read only* — a missing session yields a placeholder context
    rather than quietly minting a new one, because building a prompt must not
    have write side effects. The whole string never exceeds
    :data:`MAX_SESSION_CONTEXT` characters.
    """
    root = Path(workspace)
    if session is None:
        session = _read_session(root)
    if not isinstance(session, Mapping):
        session = {}

    lines = [
        "# Session context",
        f"- session: {session.get('session_id') or '(none yet)'}",
        f"- turns so far: {_as_int(session.get('turn_index'))}",
        _workspace_file_lines(root),
        _conversation_lines(session),
    ]
    return _shorten("\n".join(lines), MAX_SESSION_CONTEXT)


def _workspace_file_lines(root: Path) -> str:
    """The recent-files section: 30 newest paths, checkpoint tree excluded."""
    from ..tools.grep_tool import SKIP_DIRS  # lazy: core must not import tools at import time

    skip = SKIP_DIRS | RUNTIME_DIRECTORY_NAMES
    entries: list[tuple[float, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in skip)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, path.relative_to(root).as_posix()))
    entries.sort(reverse=True)

    lines = ["", "## Workspace files (most recent first)"]
    if not entries:
        lines.append("(no files yet)")
        return "\n".join(lines)
    for _, rel in entries[:MAX_RECENT_FILES]:
        lines.append(f"- {rel}")
    hidden = len(entries) - MAX_RECENT_FILES
    if hidden > 0:
        lines.append(f"…and {hidden} more.")
    return "\n".join(lines)


def _conversation_lines(session: Mapping[str, Any]) -> str:
    """The recent-turns section: what was said, newest last (reading order)."""
    turns = [turn for turn in (session.get("recent_turns") or []) if isinstance(turn, Mapping)]
    lines = ["", "## Recent conversation (oldest first)"]
    if not turns:
        lines.append("(no prior turns)")
        return "\n".join(lines)

    for turn in turns[-MAX_CONTEXT_TURNS:]:
        number = _as_int(turn.get("turn"))
        role = str(turn.get("role") or "-")
        if role == "assistant":
            label = f"assistant ({turn.get('route') or 'unknown route'})"
            body = str(turn.get("summary") or "").strip() or _shorten(
                str(turn.get("content") or ""), 600
            )
        else:
            label = "user"
            body = str(turn.get("content") or "").strip()
        lines.append(f"[turn {number}] {label}: {body or '(empty)'}")
    return "\n".join(lines)


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
