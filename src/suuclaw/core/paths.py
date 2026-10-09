"""Workspace-scoped path helpers.

Every filesystem-facing tool funnels through :func:`resolve_in_workspace`, so a
tool can never read, write, or execute outside ``RuntimeState.workspace``.

The check is done on the *resolved* path (``Path.resolve()`` follows symlinks and
normalises ``..``), which is what makes ``../`` tricks and symlink escapes fail
instead of silently succeeding.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "RUNTIME_DIRECTORY_NAMES",
    "WorkspacePathError",
    "ensure_workspace",
    "is_within",
    "relative_posix",
    "resolve_in_workspace",
    "runtime_path",
    "to_display_path",
]

RUNTIME_DIRECTORY_NAMES = frozenset({".suuclaw"})




def runtime_path(
    workspace: str | os.PathLike[str],
    *parts: str,
    marker: str | None = None,
) -> Path:
    """Return SuuClaw metadata paths without creating directories."""
    return Path(workspace).joinpath(".suuclaw", *parts)


class WorkspacePathError(ValueError):
    """Raised when a requested path would escape the workspace sandbox."""


def ensure_workspace(workspace: str | os.PathLike[str], *, create: bool = True) -> Path:
    """Return ``workspace`` as an absolute, resolved directory.

    ``create=True`` (the default) makes the CLI work out of the box: running
    ``suuclaw <task> --workspace ./sandbox`` just creates ``./sandbox``.
    """
    path = Path(os.fspath(workspace)).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    if not resolved.is_dir():
        raise WorkspacePathError(f"workspace is not an existing directory: {resolved}")
    return resolved


def is_within(path: str | os.PathLike[str], base: str | os.PathLike[str]) -> bool:
    """True when ``path`` resolves to ``base`` or something nested inside it."""
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
    except ValueError:
        return False
    return True


def resolve_in_workspace(
    workspace: str | os.PathLike[str],
    raw_path: str | os.PathLike[str] | None = None,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve ``raw_path`` against ``workspace`` and guard the sandbox boundary.

    Relative paths are interpreted from the workspace root; absolute paths are
    accepted only when they still point inside the workspace. ``""``/``None``/``.``
    resolve to the workspace itself.
    """
    root = Path(os.fspath(workspace)).resolve()
    text = "" if raw_path is None else str(raw_path).strip()

    if text in {"", ".", "./", ".\\"}:
        candidate = root
    else:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate

    resolved = candidate.resolve()
    if not is_within(resolved, root):
        raise WorkspacePathError(
            f"path {(text or '.')!r} resolves to {resolved}, outside the workspace {root}"
        )
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {resolved}")
    return resolved


def relative_posix(path: str | os.PathLike[str], base: str | os.PathLike[str]) -> str:
    """Return ``path`` relative to ``base`` using forward slashes (stable output)."""
    target = Path(path)
    try:
        rel = target.resolve().relative_to(Path(base).resolve())
    except ValueError:
        return target.as_posix()
    text = rel.as_posix()
    return text or "."


def to_display_path(workspace: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    """Alias of :func:`relative_posix`, kept for readability at call sites."""
    return relative_posix(path, workspace)
