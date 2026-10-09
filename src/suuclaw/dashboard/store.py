"""Durable dashboard sessions and bounded, workspace-scoped artifact reads.

Dashboard metadata lives outside the Agent's working directory. A session owns
one initially empty working directory, reused by its later conversational turns.
Nothing in the selected source workspace is copied, renamed, or overwritten.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})
MAX_PREVIEW_BYTES = 256 * 1024
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_FILES = 1000
_ID = re.compile(r"^[a-f0-9]{32}$")
_SENSITIVE_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
_SENSITIVE_MARKERS = ("credential", "secret", "private_key", "api_key", "password")
_EXCLUDED_DIRS = {"__pycache__", "node_modules", "venv"}


class DashboardError(Exception):
    """An expected API error with a safe, actionable public description."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hidden_or_sensitive(parts: tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in parts)
    if any(part.startswith(".") or part in _EXCLUDED_DIRS for part in lowered):
        return True
    name = lowered[-1] if lowered else ""
    return (
        name in _SENSITIVE_NAMES
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
        or any(marker in name for marker in _SENSITIVE_MARKERS)
    )


class DashboardStore:
    """Atomic JSON metadata and append-only event logs guarded by one lock."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.root = self.workspace / ".suuclaw" / "dashboard"
        self.sessions_root = self.root / "sessions"
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        for directory in (self.workspace / ".suuclaw", self.root, self.sessions_root):
            if directory.resolve() != directory:
                raise DashboardError("unsafe_storage", "Invalid dashboard storage path.")
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        if not self.sessions_root.resolve().is_relative_to(self.workspace):
            raise DashboardError("unsafe_storage", "Dashboard storage escapes the workspace.")
        self._load()

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _ID.fullmatch(value):
            raise DashboardError("not_found", "Session or run does not exist.", 404)

    def _session_dir(self, session_id: str) -> Path:
        self._validate_id(session_id)
        path = self.sessions_root / session_id
        if path.is_symlink() or not path.resolve().is_relative_to(self.sessions_root.resolve()):
            raise DashboardError("unsafe_storage", "Invalid session storage path.", 403)
        return path

    def session_workspace(self, session_id: str) -> Path:
        self.get_session(session_id)
        root = self._session_dir(session_id) / "workspace"
        if root.is_symlink() or root.resolve() != root:
            raise DashboardError("unsafe_storage", "Invalid session workspace path.", 403)
        return root

    def _run_dir(self, run: dict[str, Any]) -> Path:
        path = self._session_dir(run["session_id"]) / "runs" / run["id"]
        if path.resolve() != path:
            raise DashboardError("unsafe_storage", "Invalid run storage path.", 403)
        return path

    def _load(self) -> None:
        for folder in self.sessions_root.iterdir():
            if not _ID.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            try:
                session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
                if session.get("id") != folder.name:
                    continue
                self._sessions[session["id"]] = session
                for run_path in (folder / "runs").glob("*/run.json"):
                    if run_path.resolve() != run_path:
                        continue
                    run = json.loads(run_path.read_text(encoding="utf-8"))
                    if (
                        _ID.fullmatch(str(run.get("id", "")))
                        and run.get("id") == run_path.parent.name
                        and run.get("session_id") == session["id"]
                    ):
                        self._runs[run["id"]] = run
                        # JSONL is authoritative if a crash landed between the
                        # event append and the metadata replace.
                        events = self.read_events(run["id"])
                        run["last_seq"] = max((e["seq"] for e in events), default=0)
            except (OSError, ValueError, TypeError, KeyError):
                # A damaged record must not make healthy sessions inaccessible.
                continue

    def create_session(self, title: str = "新会话") -> dict[str, Any]:
        now = utcnow()
        session_id = uuid4().hex
        session = {
            "id": session_id,
            "title": str(title or "新会话").strip()[:100] or "新会话",
            "created_at": now,
            "updated_at": now,
            "active_run_id": None,
            "last_run_status": None,
            "run_count": 0,
            "messages": [],
        }
        with self._lock:
            folder = self._session_dir(session_id)
            (folder / "workspace").mkdir(parents=True)
            (folder / "runs").mkdir()
            _write_json(folder / "session.json", session)
            self._sessions[session_id] = session
            return self.get_session(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {key: _copy(value) for key, value in item.items() if key != "messages"}
                for item in sorted(
                    self._sessions.values(), key=lambda item: item["updated_at"], reverse=True
                )
            ]

    def get_session(self, session_id: str) -> dict[str, Any]:
        self._validate_id(session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise DashboardError("not_found", "Session does not exist.", 404)
            result = _copy(session)
            result["workspace"] = str(self._session_dir(session_id) / "workspace")
            result["runs"] = sorted(
                [_copy(run) for run in self._runs.values() if run["session_id"] == session_id],
                key=lambda run: run["created_at"],
            )
            return result

    def create_run(self, session_id: str, message: str) -> dict[str, Any]:
        with self._lock:
            self.get_session(session_id)
            session = self._sessions[session_id]
            if session.get("active_run_id"):
                raise DashboardError("session_busy", "This session already has an active run.", 409)
            run_id = uuid4().hex
            now = utcnow()
            run = {
                "id": run_id,
                "run_id": run_id,
                "session_id": session_id,
                "status": "running",
                "message": message,
                "created_at": now,
                "finished_at": None,
                "elapsed_ms": 0,
                "last_seq": 0,
                "answer": "",
                "error": None,
                "events_url": f"/api/runs/{run_id}/events",
            }
            folder = self._run_dir(run)
            folder.mkdir(parents=True)
            (folder / "events.jsonl").touch()
            _write_json(folder / "run.json", run)
            self._runs[run_id] = run
            session["active_run_id"] = run_id
            session["last_run_status"] = "running"
            session["updated_at"] = now
            session["run_count"] += 1
            if session["run_count"] == 1 and session["title"] == "新会话":
                session["title"] = " ".join(message.split())[:48]
            session["messages"].append(
                {"role": "user", "content": message, "at": now, "run_id": run_id}
            )
            _write_json(self._session_dir(session_id) / "session.json", session)
            return _copy(run)

    def get_run(self, run_id: str) -> dict[str, Any]:
        self._validate_id(run_id)
        with self._lock:
            if run_id not in self._runs:
                raise DashboardError("not_found", "Run does not exist.", 404)
            return _copy(self._runs[run_id])

    def active_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_copy(run) for run in self._runs.values() if run["status"] == "running"]

    def append_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self.get_run(run_id)
            run = self._runs[run_id]
            if run["status"] in TERMINAL_STATUSES:
                return None
            record = {
                **event,
                "seq": run["last_seq"] + 1,
                "run_id": run_id,
                "session_id": run["session_id"],
                "at": utcnow(),
            }
            folder = self._run_dir(run)
            event_path = folder / "events.jsonl"
            # Preserve a torn tail for diagnosis, but keep the next JSON record
            # on its own line so a crash cannot hide the recovery event.
            if event_path.stat().st_size:
                with event_path.open("rb") as stream:
                    stream.seek(-1, os.SEEK_END)
                    needs_newline = stream.read(1) != b"\n"
            else:
                needs_newline = False
            with event_path.open("a", encoding="utf-8") as stream:
                if needs_newline:
                    stream.write("\n")
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            run["last_seq"] = record["seq"]
            run["elapsed_ms"] = record.get("elapsed_ms", run["elapsed_ms"])
            _write_json(folder / "run.json", run)
            return _copy(record)

    def read_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            run = self.get_run(run_id)
            events: list[dict[str, Any]] = []
            path = self._run_dir(run) / "events.jsonl"
            if not path.exists():
                return events
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                        if isinstance(event.get("seq"), int) and event["seq"] > after:
                            events.append(event)
                    except (ValueError, TypeError, AttributeError):
                        # Only a torn final line is expected after abrupt exit.
                        continue
            return events

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        answer: str = "",
        error: str | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("finish_run needs a terminal status")
        with self._lock:
            self.get_run(run_id)
            run = self._runs[run_id]
            if run["status"] in TERMINAL_STATUSES:
                return
            run.update(status=status, answer=answer, error=error, finished_at=utcnow())
            _write_json(self._run_dir(run) / "run.json", run)
            session = self._sessions[run["session_id"]]
            if session.get("active_run_id") == run_id:
                session["active_run_id"] = None
            session["last_run_status"] = status
            session["updated_at"] = run["finished_at"]
            if answer or error:
                session["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer or error,
                        "at": run["finished_at"],
                        "run_id": run_id,
                        "status": status,
                    }
                )
            _write_json(self._session_dir(session["id"]) / "session.json", session)

    def recover_interrupted(self) -> None:
        """Report abandoned runs honestly instead of leaving a permanent spinner."""
        with self._lock:
            for run in self.active_runs():
                message = (
                    "The dashboard stopped before this run finished. Start a new turn to retry."
                )
                self.append_event(
                    run["id"],
                    {
                        "type": "run_end",
                        "status": "interrupted",
                        "elapsed_ms": run["elapsed_ms"],
                        "error": message,
                    },
                )
                self.finish_run(run["id"], "interrupted", error=message)
            # Also repair a torn session update after a run became terminal.
            for session in self._sessions.values():
                active_id = session.get("active_run_id")
                active = self._runs.get(active_id)
                if active_id and (active is None or active["status"] in TERMINAL_STATUSES):
                    session["active_run_id"] = None
                    session["last_run_status"] = active["status"] if active else "interrupted"
                    _write_json(self._session_dir(session["id"]) / "session.json", session)

    def resolve_file(self, session_id: str, raw_path: str) -> Path:
        root = self.session_workspace(session_id)
        text = raw_path.replace("\\", "/")
        parts = tuple(text.split("/"))
        if (
            not text
            or "\x00" in text
            or ":" in text
            or text.startswith("/")
            or PureWindowsPath(text).drive
            or any(part in {"", ".", ".."} for part in parts)
            or _hidden_or_sensitive(parts)
        ):
            raise DashboardError("unsafe_path", "This file is not available for preview.", 403)
        path = root.joinpath(*parts)
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except FileNotFoundError as exc:
            raise DashboardError("not_found", "File does not exist.", 404) from exc
        except (OSError, ValueError) as exc:
            raise DashboardError("unsafe_path", "File leaves the session workspace.", 403) from exc
        if (
            resolved != path
            or _hidden_or_sensitive(relative.parts)
            or not resolved.is_file()
            or resolved.stat().st_nlink > 1
        ):
            raise DashboardError("unsafe_path", "This file is not available for preview.", 403)
        return resolved

    def list_files(self, session_id: str) -> dict[str, Any]:
        root = self.session_workspace(session_id)
        files: list[dict[str, Any]] = []
        visited = 0
        for current, directories, names in os.walk(root, followlinks=False):
            base = Path(current)
            directories[:] = sorted(
                name
                for name in directories
                if not _hidden_or_sensitive((name,)) and (base / name).resolve() == base / name
            )
            for name in sorted(names):
                visited += 1
                if visited > MAX_FILES * 4 or len(files) >= MAX_FILES:
                    return {"files": files, "truncated": True}
                relative = (base / name).relative_to(root).as_posix()
                try:
                    path = self.resolve_file(session_id, relative)
                    stat = path.stat()
                except (DashboardError, OSError):
                    continue
                files.append(
                    {
                        "path": relative,
                        "name": name,
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(timespec="milliseconds"),
                    }
                )
        return {"files": files, "truncated": False}

    def read_file(self, session_id: str, raw_path: str, *, download: bool = False) -> Any:
        path = self.resolve_file(session_id, raw_path)
        size = path.stat().st_size
        if download and size > MAX_DOWNLOAD_BYTES:
            raise DashboardError("file_too_large", "Downloads are limited to 20 MiB.", 413)
        limit = MAX_DOWNLOAD_BYTES if download else MAX_PREVIEW_BYTES
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if download:
            if len(data) > limit:
                raise DashboardError("file_too_large", "Downloads are limited to 20 MiB.", 413)
            return path.name, data
        result = {"path": raw_path, "name": path.name, "size": size, "truncated": size > limit}
        try:
            decoder = codecs.getincrementaldecoder("utf-8-sig")()
            content = decoder.decode(data[:limit], final=size <= limit)
            if "\x00" in content:
                raise UnicodeError("binary content")
        except UnicodeError:
            return {**result, "binary": True, "message": "Binary file; download to inspect."}
        return {**result, "binary": False, "content": content}
