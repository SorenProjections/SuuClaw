"""Safe per-turn artifact discovery for Feishu delivery."""

from __future__ import annotations

import io
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ArtifactSnapshot", "ArtifactTracker", "build_artifact_archive"]

_WRITE_TOOLS = {"file_write", "file_edit", "write", "edit"}
_EXCLUDED_PARTS = {".git", ".suuclaw", "__pycache__", ".pytest_cache", ".ruff_cache"}
_SENSITIVE_MARKERS = (
    "api_key",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_SENSITIVE_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Immutable hand-off from a completed Agent turn to the uploader."""

    root: Path
    paths: tuple[Path, ...]
    failures: tuple[str, ...] = ()
    omitted: int = 0


class ArtifactTracker:
    """Collect files successfully written by one Agent event stream.

    Tracking tool call/result pairs is deliberate: taking a global workspace
    snapshot would allow concurrent chats to claim one another's files.  Every
    candidate is resolved again at collection time so deleted files, symlink
    escapes, hidden runtime state, and likely credential files are excluded.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self._pending: dict[str, str] = {}
        self._written: list[str] = []

    def observe(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("type") or "")
        name = str(event.get("name") or "")
        call_id = str(event.get("id") or "")

        if kind == "tool_call" and name in _WRITE_TOOLS and call_id:
            args = event.get("args")
            if not isinstance(args, Mapping):
                return
            raw_path = str(args.get("file_path") or args.get("path") or "").strip()
            if raw_path:
                self._pending[call_id] = raw_path
            return

        if kind != "tool_result" or name not in _WRITE_TOOLS or not call_id:
            return
        raw_path = self._pending.pop(call_id, "")
        if bool(event.get("ok")) and raw_path and raw_path not in self._written:
            self._written.append(raw_path)

    def collect(self) -> tuple[Path, ...]:
        """Return existing, non-sensitive files that remain inside workspace."""
        artifacts: list[Path] = []
        seen: set[Path] = set()
        for raw_path in self._written:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            try:
                resolved = candidate.resolve(strict=True)
                relative = resolved.relative_to(self.workspace)
            except (FileNotFoundError, OSError, ValueError):
                continue
            if resolved in seen or not resolved.is_file():
                continue
            if self._is_excluded(relative):
                continue
            seen.add(resolved)
            artifacts.append(resolved)
        return tuple(artifacts)

    def snapshot(
        self,
        destination: Path,
        *,
        max_files: int,
        max_bytes: int,
    ) -> ArtifactSnapshot:
        """Copy this turn's bounded artifacts outside the Agent workdir.

        Uploading from a private snapshot closes the race where another turn
        overwrites a shared filename after this turn completed but before the
        Feishu upload read it.
        """
        sources = self.collect()
        selected = sources[: max(0, max_files)]
        omitted = max(0, len(sources) - len(selected))
        root = destination.expanduser().resolve()
        paths: list[Path] = []
        failures: list[str] = []

        for source in selected:
            try:
                size = source.stat().st_size
                relative = source.relative_to(self.workspace)
                if size <= 0 or size > max_bytes:
                    failures.append(source.name)
                    continue
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            except (OSError, ValueError):
                failures.append(source.name)
                continue
            paths.append(target)
        return ArtifactSnapshot(
            root=root,
            paths=tuple(paths),
            failures=tuple(failures),
            omitted=omitted,
        )

    @staticmethod
    def _is_excluded(relative: Path) -> bool:
        lowered_parts = tuple(part.lower() for part in relative.parts)
        if any(part.startswith(".") or part in _EXCLUDED_PARTS for part in lowered_parts):
            return True
        name = relative.name.lower()
        stem = relative.stem.lower()
        if name in _SENSITIVE_NAMES:
            return True
        return any(marker in stem for marker in _SENSITIVE_MARKERS)


def build_artifact_archive(
    paths: tuple[Path, ...],
    *,
    workspace: Path,
    max_bytes: int,
) -> bytes | None:
    """Build a bounded in-memory ZIP while preserving workspace-relative paths."""
    if len(paths) < 2:
        return None
    root = workspace.expanduser().resolve()
    buffer = io.BytesIO()
    try:
        if sum(path.stat().st_size for path in paths) > max_bytes:
            return None
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                resolved = path.resolve(strict=True)
                relative = resolved.relative_to(root)
                archive.write(resolved, relative.as_posix())
                if buffer.tell() > max_bytes:
                    return None
    except (FileNotFoundError, OSError, ValueError, zipfile.BadZipFile):
        return None
    data = buffer.getvalue()
    return data if 0 < len(data) <= max_bytes else None
