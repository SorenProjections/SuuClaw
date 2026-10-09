"""Persistence and artifact boundary checks, without a model or network service."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from suuclaw.dashboard.store import (
    MAX_DOWNLOAD_BYTES,
    MAX_PREVIEW_BYTES,
    DashboardError,
    DashboardStore,
)


def test_sessions_are_isolated_and_preserve_source(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    store = DashboardStore(tmp_path)
    first = store.create_session()
    second = store.create_session("Other")
    first_root = store.session_workspace(first["id"])
    second_root = store.session_workspace(second["id"])
    assert first_root != second_root
    assert first_root.is_relative_to(tmp_path / ".suuclaw" / "dashboard" / "sessions")
    assert list(first_root.iterdir()) == []
    (first_root / "report.md").write_text("one", encoding="utf-8")
    assert store.list_files(second["id"])["files"] == []
    assert source.read_text(encoding="utf-8") == "original"


def test_runs_persist_replay_and_conversation(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    run = store.create_run(session["id"], "Write a report")
    for event in (
        {"type": "run_start", "elapsed_ms": 0},
        {"type": "tool_call", "name": "file_write", "args": {"path": "report.md"}, "elapsed_ms": 5},
        {"type": "final_answer", "content": "Done", "elapsed_ms": 10},
        {"type": "run_end", "status": "completed", "elapsed_ms": 11},
    ):
        store.append_event(run["id"], event)
    store.finish_run(run["id"], "completed", answer="Done")
    restored = DashboardStore(tmp_path)
    assert restored.get_run(run["id"])["status"] == "completed"
    assert [event["seq"] for event in restored.read_events(run["id"], 2)] == [3, 4]
    detail = restored.get_session(session["id"])
    assert detail["active_run_id"] is None
    assert detail["title"] == "Write a report"
    assert [(message["role"], message["content"]) for message in detail["messages"]] == [
        ("user", "Write a report"),
        ("assistant", "Done"),
    ]
    assert detail["runs"][0]["elapsed_ms"] == 11
    # Terminal runs cannot be rewritten by a late worker.
    assert restored.append_event(run["id"], {"type": "late"}) is None
    restored.finish_run(run["id"], "failed", error="late failure")
    assert restored.get_run(run["id"])["status"] == "completed"


def test_busy_session_rejects_second_run(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    first = store.create_run(session["id"], "First")
    with pytest.raises(DashboardError, match="already has an active run") as exc:
        store.create_run(session["id"], "Second")
    assert exc.value.status == 409
    assert len(store.get_session(session["id"])["messages"]) == 1
    store.finish_run(first["id"], "failed", error="Stopped")
    assert store.create_run(session["id"], "Third")["id"] != first["id"]


def test_crash_recovery_repairs_torn_log_and_clears_active_run(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    run = store.create_run(session["id"], "Interrupted task")
    store.append_event(run["id"], {"type": "node_start", "node": "planner", "elapsed_ms": 1})
    path = store.sessions_root / session["id"] / "runs" / run["id"] / "events.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"seq":2,"type":"partial')
    restored = DashboardStore(tmp_path)
    restored.recover_interrupted()
    assert restored.get_run(run["id"])["status"] == "interrupted"
    assert restored.get_session(session["id"])["active_run_id"] is None
    events = restored.read_events(run["id"])
    assert [event["seq"] for event in events] == [1, 2]
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "interrupted"
    # Recovery is idempotent.
    restored.recover_interrupted()
    assert len(restored.read_events(run["id"])) == 2


def test_event_log_authoritative_after_metadata_write_crash(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    run = store.create_run(session["id"], "Task")
    store.append_event(run["id"], {"type": "node_start", "elapsed_ms": 0})
    path = store.sessions_root / session["id"] / "runs" / run["id"] / "run.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["last_seq"] = 0
    path.write_text(json.dumps(metadata), encoding="utf-8")
    restored = DashboardStore(tmp_path)
    restored.append_event(run["id"], {"type": "node_output", "elapsed_ms": 1})
    assert [event["seq"] for event in restored.read_events(run["id"])] == [1, 2]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "report.txt:stream",
        ".env",
        ".env.production",
        ".suuclaw/session/session.json",
        "secrets.json",
        "nested/.git/config",
        "id_rsa",
        "private.key",
        "nested/../report.txt",
        "",
    ],
)
def test_unsafe_files_rejected(tmp_path: Path, path: str) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    with pytest.raises(DashboardError) as exc:
        store.read_file(session["id"], path)
    assert exc.value.status == 403


def test_file_list_preview_binary_and_bounded_download(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    root = store.session_workspace(session["id"])
    (root / "report.md").write_text("你好，SuuClaw", encoding="utf-8")
    (root / ".env").write_text("OPENAI_API_KEY=hidden", encoding="utf-8")
    (root / "credentials.json").write_text("{}", encoding="utf-8")
    (root / ".suuclaw").mkdir()
    (root / ".suuclaw" / "trace.json").write_text("{}", encoding="utf-8")
    (root / "image.bin").write_bytes(b"\x00\xff\x01")
    (root / "large.txt").write_text("中" * (MAX_PREVIEW_BYTES // 3 + 20), encoding="utf-8")
    files = store.list_files(session["id"])
    assert {item["path"] for item in files["files"]} == {"report.md", "image.bin", "large.txt"}
    assert store.read_file(session["id"], "report.md")["content"] == "你好，SuuClaw"
    assert store.read_file(session["id"], "image.bin")["binary"] is True
    large = store.read_file(session["id"], "large.txt")
    assert large["truncated"] is True
    assert large["binary"] is False
    assert store.read_file(session["id"], "image.bin", download=True)[1] == b"\x00\xff\x01"
    with (root / "huge.bin").open("wb") as stream:
        stream.truncate(MAX_DOWNLOAD_BYTES + 1)
    with pytest.raises(DashboardError) as exc:
        store.read_file(session["id"], "huge.bin", download=True)
    assert exc.value.status == 413


def test_symlinks_and_hardlinks_cannot_expose_source_files(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    session = store.create_session()
    root = store.session_workspace(session["id"])
    source = tmp_path / "private.txt"
    source.write_text("outside", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(source)
    except OSError:
        pass  # Windows may not grant symlink creation; hardlinks still exercise the boundary.
    else:
        with pytest.raises(DashboardError):
            store.read_file(session["id"], "link.txt")
    os.link(source, root / "hardlink.txt")
    with pytest.raises(DashboardError):
        store.read_file(session["id"], "hardlink.txt")
    assert store.list_files(session["id"])["files"] == []


def test_invalid_ids_do_not_address_filesystem(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    for item in ("../", "../../secret", "C:\\Windows", "not-a-session"):
        with pytest.raises(DashboardError) as exc:
            store.get_session(item)
        assert exc.value.status == 404
