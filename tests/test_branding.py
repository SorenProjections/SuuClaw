from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from suuclaw import __version__, entrypoints
from suuclaw.feishu.config import FeishuGatewayConfig, GatewayConfigError


def test_public_module_entrypoint_reports_suuclaw():
    module = "suuclaw"
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(project / "src")}
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"SuuClaw {__version__}"


@pytest.mark.parametrize(
    "args",
    [
        ["create a dashboard", "-w", "sandbox"],
        ["dashboard overview"],
        ["--", "dashboard"],
        ["--list-tools"],
        ["--tui", "--workspace", "sandbox"],
    ],
)
def test_dispatch_preserves_existing_task_and_option_arguments(args):
    with (
        patch("suuclaw.cli.app.app") as cli,
        patch.object(entrypoints, "dashboard_main") as dashboard,
    ):
        assert entrypoints.main(args) == 0
    cli.assert_called_once_with(args=args, prog_name="suuclaw")
    dashboard.assert_not_called()


def test_only_exact_leading_dashboard_dispatches_to_server():
    args = ["--workspace", "sandbox", "--port", "7789"]
    with patch.object(entrypoints, "dashboard_main", return_value=3) as dashboard:
        assert entrypoints.main(["dashboard", *args]) == 3
    dashboard.assert_called_once_with(args)


def test_feishu_public_entrypoint_forwards_arguments():
    with patch("suuclaw.feishu.app.main", return_value=2) as feishu:
        assert entrypoints.feishu_main(["--check"]) == 2
    feishu.assert_called_once_with(["--check"])


def test_suuclaw_feishu_settings_build_config(tmp_path):
    env = {
        "FEISHU_APP_ID": "cli_test", "FEISHU_APP_SECRET": "secret",
        "SUUCLAW_FEISHU_WORKSPACE": "agent-workspace",
        "SUUCLAW_FEISHU_SESSION_ROOT": "conversation-workspaces",
        "SUUCLAW_FEISHU_MODEL": "local-test-model",
        "SUUCLAW_FEISHU_TEMPERATURE": "0.3",
        "SUUCLAW_FEISHU_MAX_CONCURRENCY": "2",
        "SUUCLAW_FEISHU_APPROVAL_MODE": "auto",
        "SUUCLAW_FEISHU_WORKFLOW_PROGRESS": "false",
        "SUUCLAW_FEISHU_ALLOW_FROM": "ou_one, ou_two",
        "SUUCLAW_FEISHU_PROCESSING_TEXT": "SuuClaw is working",
    }
    original = dict(env)
    config = FeishuGatewayConfig.from_env(env, base=tmp_path)
    assert config.workspace == tmp_path / "agent-workspace"
    assert config.session_root == tmp_path / "conversation-workspaces"
    assert config.model == "local-test-model"
    assert config.max_concurrency == 2
    assert config.allow_from == ("ou_one", "ou_two")
    assert env == original


def test_explicitly_empty_feishu_processing_text_disables_acknowledgement(tmp_path):
    config = FeishuGatewayConfig.from_env(
        {"FEISHU_APP_ID": "cli_test", "FEISHU_APP_SECRET": "secret",
         "SUUCLAW_FEISHU_PROCESSING_TEXT": ""}, base=tmp_path,
    )
    assert config.processing_text == ""


def test_invalid_feishu_setting_identifies_the_supplied_name():
    name = "SUUCLAW_FEISHU_MAX_CONCURRENCY"
    with pytest.raises(GatewayConfigError, match=name):
        FeishuGatewayConfig.from_env(
            {"FEISHU_APP_ID": "cli_test", "FEISHU_APP_SECRET": "secret", name: "zero"}
        )


def test_tui_wordmark_contains_complete_name():
    pytest.importorskip("textual")
    from suuclaw.cli.tui.app import SuuClawTuiApp
    from suuclaw.cli.tui.logo import _GLYPHS, LOGO_TITLE, banner, logo_frames, wordmark

    assert LOGO_TITLE == "SuuClaw"
    assert all(char in _GLYPHS for char in LOGO_TITLE)
    assert len(wordmark()) == 5
    assert all(len(set(map(len, _GLYPHS[char]))) == 1 for char in LOGO_TITLE)
    assert "SuuClaw" in banner().plain
    assert list(logo_frames())[-1].plain == banner().plain
    assert SuuClawTuiApp.TITLE.endswith("SuuClaw")
