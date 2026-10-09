"""Per-conversation workspace preparation for the Feishu gateway."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from ..core.paths import RUNTIME_DIRECTORY_NAMES, runtime_path
from ..core.session import session_dir

__all__ = [
    "activate_conversation_generation",
    "active_conversation_generation",
    "active_conversation_root",
    "new_conversation_generation",
    "prepare_conversation_workspace",
]

_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
} | RUNTIME_DIRECTORY_NAMES
_SKIP_FILES = {
    ".env",
    "credentials.json",
    "notepad.md",
    "secrets.json",
}
_MARKER = "feishu-workspace-seeded"
_ACTIVE_FILE = "active-session.json"
_GENERATIONS_DIR = "sessions"


def active_conversation_generation(conversation_namespace: Path) -> str:
    """Return the selected generation id, or ``""`` for the legacy root."""
    state_path = conversation_namespace / _ACTIVE_FILE
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    generation = str(payload.get("generation") or "") if isinstance(payload, dict) else ""
    if len(generation) != 32 or any(char not in "0123456789abcdef" for char in generation):
        return ""
    return generation


def active_conversation_root(conversation_namespace: Path) -> Path:
    """Resolve the currently selected, persistent workspace generation."""
    generation = active_conversation_generation(conversation_namespace)
    if not generation:
        return conversation_namespace
    return conversation_namespace / _GENERATIONS_DIR / f"session-{generation}"


def new_conversation_generation(conversation_namespace: Path) -> tuple[str, Path]:
    """Reserve a fresh generation path without changing the active pointer."""
    generation = uuid4().hex
    root = conversation_namespace / _GENERATIONS_DIR / f"session-{generation}"
    return generation, root


def activate_conversation_generation(
    conversation_namespace: Path,
    generation: str,
) -> Path:
    """Atomically select an already prepared generation and return its root."""
    if len(generation) != 32 or any(char not in "0123456789abcdef" for char in generation):
        raise ValueError("invalid conversation generation")
    namespace = conversation_namespace.expanduser().resolve()
    root = namespace / _GENERATIONS_DIR / f"session-{generation}"
    if not root.is_dir():
        raise ValueError("conversation generation has not been prepared")
    namespace.mkdir(parents=True, exist_ok=True)
    state_path = namespace / _ACTIVE_FILE
    temporary = state_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps({"generation": generation}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, state_path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
    return root


def _copy_missing_tree(source: Path, target: Path) -> None:
    """Copy a safe project snapshot without following links or overwriting files."""
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in _SKIP_DIRS or item.name.lower() in _SKIP_FILES or item.is_symlink():
            continue
        destination = target / item.name
        if item.is_dir():
            _copy_missing_tree(item, destination)
        elif item.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def _migrate_legacy_session(legacy_workspace: Path, target: Path) -> None:
    legacy_session = session_dir(legacy_workspace)
    target_session = session_dir(target)
    if not legacy_session.is_dir() or target_session.exists():
        return
    _copy_missing_tree(legacy_session, target_session)


def prepare_conversation_workspace(
    template_workspace: Path,
    conversation_workspace: Path,
    *,
    legacy_workspace: Path | None = None,
) -> Path:
    """Create one isolated workdir, seed it once, and preserve legacy session data.

    The configured workspace is a template after isolation is enabled. Existing
    files are copied only on the first use of a conversation, never overwritten,
    and symlinks plus runtime/credential directories are deliberately skipped.
    """
    source = template_workspace.expanduser().resolve()
    target = conversation_workspace.expanduser().resolve()
    marker = runtime_path(target, _MARKER)
    source.mkdir(parents=True, exist_ok=True)

    if not marker.is_file():
        _copy_missing_tree(source, target)
        if legacy_workspace is not None:
            legacy = legacy_workspace.expanduser().resolve()
            if legacy != target:
                _migrate_legacy_session(legacy, target)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("seeded\n", encoding="utf-8")
    return target
