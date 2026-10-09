from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from suuclaw.feishu.config import FeishuGatewayConfig, GatewayConfigError
from suuclaw.feishu.gateway import AgentTurnResult, FeishuGateway, SuuClawAgentRunner


def _config(root: Path, **changes) -> FeishuGatewayConfig:
    values = {
        "app_id": "cli_test",
        "app_secret": "secret",
        "workspace": root / "workspace",
        "session_root": root / "sessions",
        "processing_text": "working",
        "max_concurrency": 2,
    }
    values.update(changes)
    return FeishuGatewayConfig(**values)


def _message(
    message_id: str = "om_1",
    *,
    chat_id: str = "oc_1",
    text: str = "hello",
    kind: str = "text",
    thread_id: str | None = None,
):
    return SimpleNamespace(
        message_id=message_id,
        chat_id=chat_id,
        sender_id="ou_1",
        raw_content_type=kind,
        content=SimpleNamespace(kind=kind),
        content_text=text,
        conversation=SimpleNamespace(thread_id=thread_id),
    )


class FakeChannel:
    def __init__(self) -> None:
        self.handlers = {}
        self.sent = []
        self.edited = []
        self.connected = False
        self.disconnected = False

    def on(self, name, handler):
        self.handlers[name] = handler

    async def send(self, to, message, opts=None):
        self.sent.append((to, message, opts))
        return SimpleNamespace(success=True, message_id=f"om_sent_{len(self.sent)}")

    async def edit_message(self, message_id, message):
        self.edited.append((message_id, message))
        return SimpleNamespace(success=True, message_id=message_id)

    async def connect_until_ready(self, *, timeout):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True


class NativeStreamChannel(FakeChannel):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls = []
        self.stream_contents = []

    async def stream(self, to, spec, opts=None):
        self.stream_calls.append((to, spec, opts))
        contents = self.stream_contents

        class Controller:
            async def set_content(self, full):
                contents.append(full)

        await spec["markdown"](Controller())
        return SimpleNamespace(success=True, message_id="om_stream")


class FeishuConfigTests(unittest.TestCase):
    def test_from_env_resolves_defaults_and_keeps_remote_commands_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = FeishuGatewayConfig.from_env(
                {
                    "FEISHU_APP_ID": "cli_x",
                    "FEISHU_APP_SECRET": "top-secret",
                    "SUUCLAW_FEISHU_MAX_CONCURRENCY": "7",
                    "SUUCLAW_FEISHU_REQUIRE_MENTION": "false",
                    "SUUCLAW_FEISHU_WS_PROXY": "direct",
                    "SUUCLAW_FEISHU_WS_OPEN_TIMEOUT": "45",
                },
                base=tmp,
            )
            self.assertEqual(config.approval_mode, "deny")
            self.assertEqual(config.max_concurrency, 7)
            self.assertFalse(config.require_mention)
            self.assertEqual(config.ws_proxy, "direct")
            self.assertEqual(config.ws_open_timeout, 45.0)
            self.assertEqual(config.entry_timeout, 30.0)
            self.assertTrue(config.deliver_artifacts)
            self.assertEqual(config.artifact_max_files, 8)
            self.assertEqual(config.artifact_max_megabytes, 25.0)
            self.assertEqual(config.workspace, Path(tmp) / "workspace")
            self.assertEqual(
                config.session_root,
                Path(tmp) / "workspace" / ".suuclaw" / "feishu" / "conversations",
            )
            self.assertNotIn("top-secret", repr(config))

    def test_from_env_requires_feishu_credentials(self):
        with self.assertRaisesRegex(GatewayConfigError, "FEISHU_APP_SECRET"):
            FeishuGatewayConfig.from_env({"FEISHU_APP_ID": "cli_x"})

    def test_inline_approval_is_not_allowed_for_headless_gateway(self):
        with self.assertRaisesRegex(GatewayConfigError, "SUUCLAW_FEISHU_APPROVAL_MODE"):
            FeishuGatewayConfig.from_env(
                {
                    "FEISHU_APP_ID": "cli_x",
                    "FEISHU_APP_SECRET": "secret",
                    "SUUCLAW_FEISHU_APPROVAL_MODE": "inline",
                }
            )


class FeishuGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_text_message_gets_progress_and_markdown_answer(self):
        channel = FakeChannel()
        calls = []

        def runner(task, session_workspace):
            calls.append((task, session_workspace))
            return AgentTurnResult("done", True, route="workflow", elapsed=0.2)

        gateway = FeishuGateway(_config(self.root), channel, runner)
        message = _message()
        await gateway.handle_message(message)

        self.assertEqual(calls, [("hello", gateway.session_workspace(message))])
        self.assertEqual(channel.sent[0], ("oc_1", {"text": "working"}, {"reply_to": "om_1"}))
        self.assertEqual(channel.sent[1], ("oc_1", {"markdown": "done"}, {"reply_to": "om_1"}))

    async def test_streaming_runner_relays_workflow_milestones(self):
        channel = FakeChannel()

        class StreamingRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "workflow"},
                    }
                )
                on_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
                on_event(
                    {
                        "type": "tool_call",
                        "node": "codeAgent",
                        "id": "call-1",
                        "name": "file_write",
                        "args": {"path": "demo.html"},
                    }
                )
                on_event(
                    {
                        "type": "tool_result",
                        "node": "codeAgent",
                        "id": "call-1",
                        "name": "file_write",
                        "ok": True,
                        "result": {"output": "secret raw tool output"},
                    }
                )
                on_event({"type": "node_start", "node": "verifier"})
                on_event(
                    {
                        "type": "node_output",
                        "node": "verifier",
                        "data": {"passed": False, "attempts": 1},
                    }
                )
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 2,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("done", True, route="workflow", elapsed=0.2)

        gateway = FeishuGateway(_config(self.root), channel, StreamingRunner())
        await gateway.handle_message(_message(text="create a file"))

        self.assertEqual(len(channel.sent), 3)
        self.assertEqual(channel.sent[0][1], {"text": "working"})
        self.assertIn("SuuClaw 工作流", channel.sent[1][1]["markdown"])
        self.assertEqual(channel.sent[1][2], {"reply_to": "om_1"})
        self.assertEqual(channel.sent[-1][1], {"markdown": "done"})

        self.assertGreaterEqual(len(channel.edited), 3)
        self.assertTrue(all(message_id == "om_sent_2" for message_id, _ in channel.edited))
        final_panel = channel.edited[-1][1]["markdown"]
        self.assertIn("🟢 已完成", final_panel)
        self.assertIn("正在处理工作区文件", final_panel)
        self.assertIn("写入文件", final_panel)
        self.assertIn("demo.html", final_panel)
        self.assertIn("第 1 轮未通过", final_panel)
        self.assertIn("**验证轮次：** 2/3", final_panel)
        self.assertNotIn("secret raw tool output", final_panel)

    async def test_native_stream_card_relays_latest_workflow_state(self):
        channel = NativeStreamChannel()

        class StreamingRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "workflow"},
                    }
                )
                on_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
                on_event({"type": "node_start", "node": "verifier"})
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 1,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("done", True, route="workflow", elapsed=0.2)

        gateway = FeishuGateway(_config(self.root), channel, StreamingRunner())
        await gateway.handle_message(_message(text="create a file"))

        self.assertEqual(len(channel.stream_calls), 1)
        self.assertEqual(channel.stream_calls[0][0], "oc_1")
        self.assertEqual(channel.stream_calls[0][2], {"reply_to": "om_1"})
        self.assertFalse(channel.edited)
        self.assertGreaterEqual(len(channel.stream_contents), 3)
        final_panel = channel.stream_contents[-1]
        self.assertIn("🟢 已完成", final_panel)
        self.assertIn("正在处理工作区文件", final_panel)
        self.assertIn("正在检查实际结果", final_panel)
        self.assertEqual(channel.sent[-1][1], {"markdown": "done"})

    async def test_native_stream_failure_falls_back_to_message_updates(self):
        class FailingNativeStreamChannel(FakeChannel):
            def __init__(self) -> None:
                super().__init__()
                self.stream_calls = 0

            async def stream(self, to, spec, opts=None):
                self.stream_calls += 1
                raise RuntimeError("CardKit unavailable")

        channel = FailingNativeStreamChannel()

        class StreamingRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "workflow"},
                    }
                )
                on_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 1,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("done", True, route="workflow", elapsed=0.2)

        gateway = FeishuGateway(_config(self.root), channel, StreamingRunner())
        with self.assertLogs("suuclaw.feishu.gateway", level="WARNING"):
            await gateway.handle_message(_message())

        self.assertEqual(channel.stream_calls, 1)
        self.assertTrue(channel.edited)
        self.assertIn("🟢 已完成", channel.edited[-1][1]["markdown"])
        self.assertEqual(channel.sent[-1][1], {"markdown": "done"})

    async def test_long_search_keeps_later_workflow_stages_visible(self):
        channel = FakeChannel()

        class LongSearchRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "workflow"},
                    }
                )
                on_event({"type": "handoff", "from": "planner", "to": "searchAgent"})
                for index in range(30):
                    on_event(
                        {
                            "type": "search_results",
                            "node": "search",
                            "ok": True,
                            "result_count": index + 1,
                        }
                    )
                on_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
                on_event({"type": "node_start", "node": "verifier"})
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 1,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("done", True, route="workflow", elapsed=1.0)

        gateway = FeishuGateway(_config(self.root), channel, LongSearchRunner())
        await gateway.handle_message(_message())

        final_panel = channel.edited[-1][1]["markdown"]
        self.assertIn("#### 阶段进度", final_panel)
        self.assertIn("正在查找所需资料", final_panel)
        self.assertIn("正在处理工作区文件", final_panel)
        self.assertIn("正在检查实际结果", final_panel)
        self.assertIn("已生成最终结果", final_panel)
        self.assertIn("较早的 18 条操作已折叠", final_panel)
        self.assertIn("找到 30 条来源", final_panel)
        self.assertNotIn("找到 1 条来源", final_panel)
        self.assertLess(len(final_panel), 3500)

    async def test_streaming_chat_does_not_claim_workflow(self):
        channel = FakeChannel()

        class ChatRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "chat"},
                    }
                )
                return AgentTurnResult("hello", True, route="chat", elapsed=0.1)

        gateway = FeishuGateway(_config(self.root), channel, ChatRunner())
        await gateway.handle_message(_message())

        self.assertEqual(len(channel.sent), 2)
        self.assertEqual(channel.sent[0][1], {"text": "working"})
        self.assertEqual(channel.sent[1][1], {"markdown": "hello"})

    async def test_successful_file_write_is_delivered_as_attachment(self):
        channel = FakeChannel()

        class ArtifactRunner:
            def run_with_events(self, task, session_workspace, on_event):
                session_workspace.mkdir(parents=True, exist_ok=True)
                artifact = session_workspace / "手机预览.html"
                artifact.write_text("<h1>你好</h1>", encoding="utf-8")
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "workflow"},
                    }
                )
                on_event(
                    {
                        "type": "tool_call",
                        "node": "code",
                        "id": "write-1",
                        "name": "file_write",
                        "args": {"file_path": "手机预览.html"},
                    }
                )
                on_event(
                    {
                        "type": "tool_result",
                        "node": "code",
                        "id": "write-1",
                        "name": "file_write",
                        "ok": True,
                    }
                )
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 1,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("完成", True, route="workflow")

        gateway = FeishuGateway(_config(self.root), channel, ArtifactRunner())
        await gateway.handle_message(_message())

        file_messages = [payload["file"] for _, payload, _ in channel.sent if "file" in payload]
        self.assertEqual(len(file_messages), 1)
        self.assertEqual(file_messages[0]["file_name"], "手机预览.html")
        delivered = Path(file_messages[0]["source"])
        self.assertEqual(delivered.read_text(encoding="utf-8"), "<h1>你好</h1>")
        self.assertEqual(delivered.parent.parent.name, "artifacts")
        self.assertEqual(channel.sent[-1][2], {"reply_to": "om_1"})

    async def test_failed_progress_edit_falls_back_to_fresh_final_panel(self):
        class RejectingEditChannel(FakeChannel):
            async def edit_message(self, message_id, message):
                self.edited.append((message_id, message))
                return SimpleNamespace(success=False, error="edit rejected")

        channel = RejectingEditChannel()

        class StreamingRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event(
                    {
                        "type": "node_output",
                        "node": "intent_router",
                        "data": {"intent_route": "workflow"},
                    }
                )
                on_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 1,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("done", True, route="workflow")

        gateway = FeishuGateway(_config(self.root), channel, StreamingRunner())
        await gateway.handle_message(_message())

        markdown_messages = [
            payload["markdown"] for _, payload, _ in channel.sent if "markdown" in payload
        ]
        self.assertTrue(any("🟢 已完成" in message for message in markdown_messages))
        self.assertEqual(markdown_messages[-1], "done")

    async def test_oversized_artifact_is_not_uploaded_and_gets_notice(self):
        channel = FakeChannel()
        workspace = self.root / "workspace"
        workspace.mkdir(parents=True)
        artifact = workspace / "large.bin"
        artifact.write_bytes(b"1234")
        gateway = FeishuGateway(
            _config(self.root, artifact_max_megabytes=0.000001),
            channel,
            lambda *_: None,
        )

        await gateway._deliver_artifacts("oc_1", "om_1", (artifact,))

        self.assertFalse(any("file" in payload for _, payload, _ in channel.sent))
        self.assertIn("large.bin", channel.sent[-1][1]["text"])
        self.assertIn("成功发送 0 个", channel.sent[-1][1]["text"])

    async def test_multifile_web_delivery_adds_in_memory_zip(self):
        channel = FakeChannel()
        workspace = self.root / "workspace"
        workspace.mkdir(parents=True)
        html = workspace / "index.html"
        css = workspace / "style.css"
        html.write_text('<link rel="stylesheet" href="style.css">', encoding="utf-8")
        css.write_text("body { color: navy; }", encoding="utf-8")
        gateway = FeishuGateway(_config(self.root), channel, lambda *_: None)

        await gateway._deliver_artifacts("oc_1", "om_1", (html, css))

        files = [payload["file"] for _, payload, _ in channel.sent if "file" in payload]
        self.assertEqual(
            [file["file_name"] for file in files],
            ["index.html", "style.css", "suuclaw-web-artifacts.zip"],
        )
        self.assertIsInstance(files[-1]["source"], bytes)

    async def test_workflow_panel_survives_router_failure_fallback(self):
        channel = FakeChannel()

        class FallbackRunner:
            def run_with_events(self, task, session_workspace, on_event):
                on_event({"type": "error", "node": "intent_router"})
                on_event({"type": "node_start", "node": "planner"})
                on_event(
                    {
                        "type": "final_answer",
                        "passed": True,
                        "attempts": 1,
                        "max_attempts": 3,
                    }
                )
                return AgentTurnResult("done", True, route="workflow", elapsed=0.2)

        gateway = FeishuGateway(_config(self.root), channel, FallbackRunner())
        await gateway.handle_message(_message())

        self.assertEqual(len(channel.sent), 3)
        self.assertIn("已进入工具工作流", channel.sent[1][1]["markdown"])
        self.assertIn("任务规划", channel.sent[1][1]["markdown"])
        self.assertIn("🟢 已完成", channel.edited[-1][1]["markdown"])

    async def test_unsupported_message_does_not_run_agent(self):
        channel = FakeChannel()

        def runner(task, session_workspace):  # pragma: no cover - must not run
            raise AssertionError("runner should not be called")

        gateway = FeishuGateway(_config(self.root), channel, runner)
        await gateway.handle_message(_message(kind="image", text="<image>"))
        self.assertEqual(len(channel.sent), 1)
        self.assertIn("只支持文本", channel.sent[0][1]["text"])

    async def test_agent_failure_is_reported_without_exception_details(self):
        channel = FakeChannel()

        def runner(task, session_workspace):
            raise RuntimeError("secret backend detail")

        gateway = FeishuGateway(_config(self.root), channel, runner)
        with self.assertLogs("suuclaw.feishu.gateway", level="ERROR"):
            await gateway.handle_message(_message())
        self.assertEqual(channel.sent[-1][1], {"text": "处理失败，请稍后重试。"})
        self.assertNotIn("secret", channel.sent[-1][1]["text"])

    async def test_same_conversation_runs_serially(self):
        channel = FakeChannel()
        guard = threading.Lock()
        active = 0
        maximum = 0

        def runner(task, session_workspace):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with guard:
                active -= 1
            return AgentTurnResult(task, True)

        gateway = FeishuGateway(_config(self.root), channel, runner)
        await asyncio.gather(
            gateway.handle_message(_message("om_1", text="one")),
            gateway.handle_message(_message("om_2", text="two")),
        )
        self.assertEqual(maximum, 1)

    async def test_session_path_is_stable_and_thread_isolated(self):
        gateway = FeishuGateway(_config(self.root), FakeChannel(), lambda *_: None)
        first = gateway.session_workspace(_message(thread_id="omt_a"))
        same = gateway.session_workspace(_message(message_id="om_2", thread_id="omt_a"))
        other = gateway.session_workspace(_message(thread_id="omt_b"))
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertEqual(first.name, "workdir")
        self.assertEqual(first.parent.parent, self.root / "sessions")

    async def test_conversations_get_distinct_seeded_workspaces(self):
        template = self.root / "workspace"
        template.mkdir(parents=True)
        (template / "project.txt").write_text("template", encoding="utf-8")
        gateway = FeishuGateway(_config(self.root), FakeChannel(), lambda *_: None)
        first_message = _message(chat_id="oc_a")
        second_message = _message(chat_id="oc_b")

        first = gateway.prepare_session_workspace(first_message)
        second = gateway.prepare_session_workspace(second_message)
        (first / "project.txt").write_text("first", encoding="utf-8")

        self.assertNotEqual(first, second)
        self.assertEqual((first / "project.txt").read_text(encoding="utf-8"), "first")
        self.assertEqual((second / "project.txt").read_text(encoding="utf-8"), "template")

    async def test_new_command_activates_fresh_core_session_and_preserves_old_workspace(self):
        channel = FakeChannel()

        def runner(task, session_workspace):  # pragma: no cover - command must not run it
            raise AssertionError("runner should not be called for /new")

        gateway = FeishuGateway(_config(self.root), channel, runner)
        message = _message(text="/new")
        old_workspace = gateway.prepare_session_workspace(message)
        (old_workspace / "old-result.txt").write_text("preserved", encoding="utf-8")

        await gateway.handle_message(message)

        new_workspace = gateway.session_workspace(message)
        self.assertNotEqual(new_workspace, old_workspace)
        self.assertEqual(
            (old_workspace / "old-result.txt").read_text(encoding="utf-8"), "preserved"
        )
        self.assertTrue((new_workspace / ".suuclaw" / "session" / "session.json").is_file())
        self.assertEqual(len(channel.sent), 1)
        self.assertIn("已开启新对话", channel.sent[0][1]["markdown"])
        self.assertNotIn({"text": "working"}, [payload for _, payload, _ in channel.sent])

        restarted = FeishuGateway(_config(self.root), FakeChannel(), runner)
        self.assertEqual(restarted.session_workspace(message), new_workspace)

    async def test_status_command_reports_active_session_without_running_agent(self):
        channel = FakeChannel()

        def runner(task, session_workspace):  # pragma: no cover - command must not run it
            raise AssertionError("runner should not be called for /status")

        gateway = FeishuGateway(_config(self.root), channel, runner)
        message = _message(text="开启新会话")
        generation, session_id, workspace = gateway.start_new_session(message)

        await gateway.handle_message(_message(message_id="om_2", text="/status"))

        panel = channel.sent[-1][1]["markdown"]
        self.assertIn("当前对话状态", panel)
        self.assertIn(session_id[:8], panel)
        self.assertIn(generation[:8], panel)
        self.assertIn(str(workspace), panel)
        self.assertNotIn({"text": "working"}, [payload for _, payload, _ in channel.sent])

    async def test_serve_registers_and_disconnects(self):
        channel = FakeChannel()
        gateway = FeishuGateway(_config(self.root), channel, lambda *_: None)
        stop = asyncio.Event()
        stop.set()
        await gateway.serve(stop)
        self.assertTrue(channel.connected)
        self.assertTrue(channel.disconnected)
        self.assertIn("message", channel.handlers)


class SuuClawAgentRunnerTests(unittest.TestCase):
    def test_runner_uses_deny_policy_and_returns_final_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            seen = {}

            def events(task, **kwargs):
                seen["task"] = task
                seen.update(kwargs)
                yield {
                    "type": "final_answer",
                    "content": "answer",
                    "passed": True,
                    "route": "chat",
                }

            runner = SuuClawAgentRunner(config, model=object())
            with patch("suuclaw.feishu.gateway.stream_session_events", events):
                result = runner("question", root / "session")

            self.assertEqual(result.content, "answer")
            self.assertEqual(result.route, "chat")
            self.assertEqual(seen["session_workspace"], root / "session")
            self.assertEqual(seen["state"].approval_mode, "deny")
            self.assertEqual(seen["state"].workspace, root / "session")
            self.assertEqual(seen["entry_timeout"], 30.0)

    def test_runner_exposes_events_to_progress_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = SuuClawAgentRunner(_config(root), model=object())
            observed = []

            def events(task, **kwargs):
                yield {
                    "type": "node_output",
                    "node": "intent_router",
                    "data": {"intent_route": "workflow"},
                }
                yield {
                    "type": "final_answer",
                    "content": "answer",
                    "passed": True,
                    "route": "workflow",
                }

            with patch("suuclaw.feishu.gateway.stream_session_events", events):
                result = runner.run_with_events("question", root / "session", observed.append)

            self.assertEqual(result.route, "workflow")
            self.assertEqual([event["type"] for event in observed], ["node_output", "final_answer"])

    def test_create_gateway_disables_hidden_model_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            channel = FakeChannel()
            model = object()
            with patch(
                "suuclaw.providers.openai_provider.create_model",
                return_value=model,
            ) as create_model:
                from suuclaw.feishu.gateway import create_gateway

                gateway = create_gateway(config, channel=channel)

            create_model.assert_called_once_with(
                config.model,
                temperature=config.temperature,
                max_retries=0,
            )
            self.assertIs(gateway.runner.model, model)


if __name__ == "__main__":
    unittest.main()
