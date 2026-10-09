from __future__ import annotations

import json
from unittest.mock import patch

from suuclaw.core.checkpoint import (
    CheckpointManager,
    GitSnapshot,
    build_recovery_markdown,
    resume_command,
)
from suuclaw.core.paths import runtime_path
from suuclaw.core.session import (
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
    session_path,
)
from suuclaw.core.state import RuntimeState
from suuclaw.core.trace import TraceRecorder
from suuclaw.feishu.config import FeishuGatewayConfig
from suuclaw.feishu.isolation import prepare_conversation_workspace
from suuclaw.graph.nodes import CONTEXT_TOKEN_LIMIT_ENVS, _env_context_token_limit
from suuclaw.tools.bash_tool import resolve_shell


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_new_workspaces_write_only_canonical_runtime_metadata(tmp_path):
    state = RuntimeState.create(tmp_path)
    session = load_or_create_session(tmp_path)
    assert session_path(tmp_path) == tmp_path / ".suuclaw/session/session.json"
    assert session["session_id"]
    manager = CheckpointManager(state, "new task")
    with patch("suuclaw.core.checkpoint.snapshot_workspace_git", return_value=GitSnapshot()):
        receipt = manager.save({"task": "new task"})
    assert receipt["type"] == "checkpoint_saved"
    assert manager.checkpoint_json == tmp_path / ".suuclaw/checkpoints/checkpoint.json"
    recorder = TraceRecorder(state, "new task")
    recorder.start({"task": "new task"})
    recorder.end(final_state={"passed": True})
    assert recorder.trace_json.is_file()
    assert recorder.root.is_relative_to(tmp_path / ".suuclaw/traces")


def test_runtime_path_selection_has_no_write_side_effect(tmp_path):
    assert runtime_path(tmp_path, "session") == tmp_path / ".suuclaw/session"
    assert not (tmp_path / ".suuclaw").exists()


def test_existing_session_survives_dashboard_root_and_next_turn(tmp_path):
    legacy_file = _write(
        tmp_path / ".suuclaw/session/session.json",
        json.dumps({"session_id": "keep-me", "turn_index": 7, "recent_turns": []}),
    )
    (tmp_path / ".suuclaw/dashboard").mkdir(parents=True)
    session = load_or_create_session(tmp_path)
    assert session["session_id"] == "keep-me"
    assert append_user_turn(session, "continue") == 8
    assert not save_session(tmp_path, session)["error"]
    assert session_path(tmp_path) == tmp_path / ".suuclaw/session/session.json"
    assert json.loads(session_path(tmp_path).read_text(encoding="utf-8"))["turn_index"] == 8
    assert legacy_file.exists()




def test_existing_checkpoint_remains_readable_and_writable(tmp_path):
    legacy = _write(
        tmp_path / ".suuclaw/checkpoints/checkpoint.json",
        '{"id":"old-checkpoint","task":"unfinished work"}',
    )
    (tmp_path / ".suuclaw/dashboard").mkdir(parents=True)
    manager = CheckpointManager(RuntimeState.create(tmp_path))
    assert manager.checkpoint_json == tmp_path / ".suuclaw/checkpoints/checkpoint.json"
    assert manager.read_payload()["id"] == "old-checkpoint"
    with patch("suuclaw.core.checkpoint.snapshot_workspace_git", return_value=GitSnapshot()):
        assert manager.save({"task": "unfinished work"})["type"] == "checkpoint_saved"
    assert manager.read_payload()["task"] == "unfinished work"
    assert legacy.exists()
    assert "# SuuClaw recovery guide" in manager.recovery_md.read_text(encoding="utf-8")


def test_trace_resume_uses_existing_suuclaw_data(tmp_path):
    legacy = _write(tmp_path / ".suuclaw/traces/old-trace/trace.json", '{"preserved":true}')
    state = RuntimeState.create(tmp_path)
    new_recorder = TraceRecorder(state)
    assert new_recorder.root.is_relative_to(tmp_path / ".suuclaw/traces")
    resumed = TraceRecorder(RuntimeState.create(tmp_path, trace_id="old-trace"))
    assert resumed.trace_json == tmp_path / ".suuclaw/traces/old-trace/trace.json"
    assert legacy.exists()
    assert resumed.trace_json.read_text(encoding="utf-8") == '{"preserved":true}'


def test_feishu_conversations_use_suuclaw_runtime_root(tmp_path):
    credentials = {"FEISHU_APP_ID": "test", "FEISHU_APP_SECRET": "secret"}
    credentials["SUUCLAW_FEISHU_WORKSPACE"] = str(tmp_path)
    config = FeishuGatewayConfig.from_env(credentials, base=tmp_path)
    assert config.session_root == tmp_path / ".suuclaw/feishu/conversations"
    (tmp_path / ".suuclaw/dashboard").mkdir(parents=True)
    legacy = tmp_path / ".suuclaw/feishu/conversations"
    legacy.mkdir(parents=True)
    config = FeishuGatewayConfig.from_env(credentials, base=tmp_path)
    assert config.session_root == tmp_path / ".suuclaw/feishu/conversations"
    assert legacy.exists()


def test_feishu_seed_marker_prevents_reseeding_existing_workspace(tmp_path):
    template = tmp_path / "template"
    _write(template / "later.txt", "do not seed again")
    target = tmp_path / "conversation/workdir"
    marker = _write(target / ".suuclaw/feishu-workspace-seeded", "seeded\n")
    prepare_conversation_workspace(template, target)
    assert not (target / "later.txt").exists()
    assert (target / ".suuclaw/feishu-workspace-seeded").read_text(encoding="utf-8") == "seeded\n"
    assert marker.exists()


def test_new_feishu_workspace_excludes_runtime_data(tmp_path):
    template = tmp_path / "template"
    _write(template / "project.py", "print('work')")
    _write(template / ".suuclaw/dashboard/private.json", "private")
    _write(template / ".suuclaw/session/private.json", "private")
    target = tmp_path / "conversation/workdir"
    prepare_conversation_workspace(template, target)
    assert (target / "project.py").is_file()
    assert (target / ".suuclaw/feishu-workspace-seeded").is_file()
    assert not (target / ".suuclaw/dashboard").exists()
    assert not (target / ".suuclaw/session").exists()


def test_manifests_and_session_context_exclude_runtime_data(tmp_path):
    _write(tmp_path / "visible.py", "print('hello')")
    _write(tmp_path / ".suuclaw/dashboard/canonical-private.txt", "private")
    _write(tmp_path / ".suuclaw/session/legacy-private.txt", "private")
    manager = CheckpointManager(RuntimeState.create(tmp_path))
    assert [file["path"] for file in manager._manifest()["files"]] == ["visible.py"]
    context = build_session_context(tmp_path, {})
    assert "visible.py" in context
    assert "private.txt" not in context


def test_recovery_text_uses_suuclaw_command():
    text = build_recovery_markdown({"workspace": "sandbox"})
    assert "SuuClaw recovery guide" in text
    assert "suuclaw --resume sandbox" in text
    assert resume_command("path with spaces") == 'suuclaw --resume "path with spaces"'


def test_shell_env_and_explicit_override(monkeypatch):
    monkeypatch.setenv("SUUCLAW_SHELL", "configured-shell -c")
    assert resolve_shell() == ["configured-shell", "-c"]
    assert resolve_shell("explicit-shell -c") == ["explicit-shell", "-c"]


def test_context_budget_environment_override(monkeypatch):
    for name in CONTEXT_TOKEN_LIMIT_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUUCLAW_CONTEXT_TOKEN_LIMIT", "5678")
    assert _env_context_token_limit() == 5678
