"""Chinese Markdown rendering for Feishu workflow progress."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

__all__ = ["WorkflowPanel"]

_NODE_LABELS = {
    "intent_router": "任务路由",
    "planner": "任务规划",
    "search": "资料检索",
    "searchAgent": "资料检索",
    "code": "工作区操作",
    "codeAgent": "工作区操作",
    "verifier": "结果验证",
    "context_monitor": "状态检查",
    "context_compressor": "上下文整理",
    "final": "生成总结",
}

_TOOL_LABELS = {
    "bash": "运行命令",
    "file_read": "读取文件",
    "file_write": "写入文件",
    "file_edit": "修改文件",
    "grep": "搜索代码",
    "web_search": "网络检索",
    "todo_write": "制定计划",
    "todo_update": "更新计划",
    "notepad_read": "读取工作笔记",
    "notepad_append": "记录工作笔记",
    "call_search_agent": "委派资料检索",
    "call_code_agent": "委派工作区操作",
}


def _one_line(value: Any, limit: int = 88) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace("`", "｀")
    if len(text) > limit:
        return f"{text[: limit - 1]}…"
    return text


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total} 秒"
    minutes, remainder = divmod(total, 60)
    return f"{minutes} 分 {remainder} 秒"


def _tool_detail(event: Mapping[str, Any]) -> str:
    args = event.get("args")
    if not isinstance(args, Mapping):
        return ""
    for key in ("path", "file_path", "query", "command"):
        value = _one_line(args.get(key))
        if value:
            return f" · `{value}`"
    return ""


class WorkflowPanel:
    """Accumulate workflow events into one Rich-like Feishu status panel."""

    _RECENT_LIMIT = 12

    def __init__(self, task: str, heartbeat_seconds: float) -> None:
        self.task = _one_line(task, 180)
        self.started = time.monotonic()
        self.heartbeat_seconds = heartbeat_seconds
        self.last_heartbeat = self.started
        self.last_render = 0.0
        self.workflow = False
        self.status = "正在判断"
        self.phase = "任务路由"
        self.attempts = 0
        self.max_attempts = 0
        self.tool_calls = 0
        self.milestones: list[str] = []
        self.entries: list[str] = []
        self._seen: set[str] = set()
        self._hidden = 0

    def _append(self, key: str, text: str, *, milestone: bool = False) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        if milestone:
            self.milestones.append(text)
            return True
        if len(self.entries) >= self._RECENT_LIMIT:
            self.entries.pop(0)
            self._hidden += 1
        self.entries.append(text)
        return True

    def _render(self) -> str:
        elapsed = _elapsed_text(time.monotonic() - self.started)
        attempts = ""
        if self.attempts:
            total = f"/{self.max_attempts}" if self.max_attempts else ""
            attempts = f"　**验证轮次：** {self.attempts}{total}"
        tools = f"　**工具调用：** {self.tool_calls}" if self.tool_calls else ""
        task = f"> {self.task}" if self.task else "> （未提供任务摘要）"
        milestones = "\n".join(self.milestones) or "- ⏳ 等待路由结果"
        recent = "\n".join(self.entries) or "- ⏳ 暂无工具操作"
        hidden = f"- …较早的 {self._hidden} 条操作已折叠\n" if self._hidden else ""
        return (
            "### 🐾 SuuClaw 工作流\n"
            f"**状态：** {self.status}　**当前阶段：** {self.phase}\n"
            f"**耗时：** {elapsed}{attempts}{tools}\n\n"
            f"{task}\n\n"
            "---\n"
            "#### 阶段进度\n"
            f"{milestones}\n\n"
            "#### 最近操作\n"
            f"{hidden}{recent}\n\n"
            "> 此面板会持续更新；任务完成后将另行发送完整结果。"
        )

    def render_for(self, event: Mapping[str, Any]) -> str:
        """Consume one event and return updated Markdown when it should be sent."""
        kind = str(event.get("type") or "")
        node = str(event.get("node") or "")
        data = event.get("data")
        payload = data if isinstance(data, Mapping) else {}
        now = time.monotonic()
        changed = False
        force = False

        if node == "intent_router" and kind == "node_output":
            route = str(payload.get("intent_route") or "")
            self.workflow = route == "workflow"
            if not self.workflow:
                return ""
            self.status = "🟡 运行中"
            self.phase = "任务规划"
            changed = self._append("route", "- 🎯 **路由：** 已进入工具工作流", milestone=True)
            force = True

        # The router deliberately fails open to the workflow when its model
        # call fails. In that case the core stream has no router output, so
        # infer the route from the first real workflow node instead of leaving
        # the user with only the initial acknowledgement.
        if (
            not self.workflow
            and kind == "node_start"
            and node not in {"", "intent_router", "chat_responder"}
        ):
            self.workflow = True
            self.status = "🟡 运行中"
            self.phase = _NODE_LABELS.get(node, node)
            changed = self._append("route", "- 🎯 **路由：** 已进入工具工作流", milestone=True)
            force = True

        if not self.workflow:
            return ""

        if kind == "node_start" and node:
            self.phase = _NODE_LABELS.get(node, node)
            if node == "verifier":
                changed |= self._append(
                    "verify", "- 🧪 **验证：** 正在检查实际结果", milestone=True
                )
                force = True
            elif node == "context_compressor":
                changed |= self._append(
                    "compress",
                    "- 🧠 **上下文：** 正在整理长期任务信息",
                    milestone=True,
                )
                force = True

        elif kind == "handoff":
            target = str(event.get("to") or event.get("to_agent") or "")
            if target == "searchAgent":
                self.phase = "资料检索"
                changed |= self._append(
                    "search", "- 🔎 **检索：** 正在查找所需资料", milestone=True
                )
            elif target == "codeAgent":
                self.phase = "工作区操作"
                changed |= self._append("code", "- 🛠️ **执行：** 正在处理工作区文件", milestone=True)
            force = changed

        elif kind == "tool_call":
            name = str(event.get("name") or "未知工具")
            label = _TOOL_LABELS.get(name, name)
            call_id = str(event.get("id") or f"{name}-{self.tool_calls + 1}")
            self.tool_calls += 1
            changed |= self._append(
                f"call-{call_id}",
                f"- 🔧 **工具：** {label}{_tool_detail(event)}",
            )

        elif kind == "tool_result":
            name = str(event.get("name") or "工具")
            label = _TOOL_LABELS.get(name, name)
            call_id = str(event.get("id") or f"{name}-{self.tool_calls}")
            ok = bool(event.get("ok"))
            icon, result = ("✅", "完成") if ok else ("⚠️", "失败")
            changed |= self._append(f"result-{call_id}", f"- {icon} **结果：** {label}{result}")
            force = not ok

        elif kind == "search_results":
            count = int(event.get("result_count") or 0)
            ok = bool(event.get("ok"))
            text = f"找到 {count} 条来源" if ok else "检索失败，正在尝试其他方案"
            changed |= self._append(
                f"search-result-{len(self._seen)}", f"- {'✅' if ok else '⚠️'} **检索：** {text}"
            )
            force = True

        elif kind == "node_output" and node == "verifier":
            self.attempts = int(payload.get("attempts") or self.attempts)
            if not payload.get("passed"):
                changed |= self._append(
                    f"retry-{self.attempts}",
                    f"- 🔁 **验证：** 第 {self.attempts} 轮未通过，正在修正",
                    milestone=True,
                )
                self.phase = "修正问题"
                force = True

        elif kind == "retry":
            attempt = int(event.get("attempt") or 0)
            changed |= self._append(
                f"model-retry-{node}-{attempt}",
                f"- 🔄 **重试：** {_NODE_LABELS.get(node, node or '模型')} 第 {attempt} 次",
                milestone=True,
            )
            force = True

        elif kind == "error":
            self.status = "🔴 遇到错误"
            changed |= self._append(
                f"error-{len(self._seen)}",
                "- ❌ **错误：** 当前步骤执行失败",
                milestone=True,
            )
            force = True

        elif kind == "heartbeat" and now - self.last_heartbeat >= self.heartbeat_seconds:
            self.last_heartbeat = now
            changed = True
            force = True

        elif kind == "final_answer":
            passed = bool(event.get("passed"))
            self.status = "🟢 已完成" if passed else "🟠 未通过验证"
            self.phase = "完成"
            self.attempts = int(event.get("attempts") or self.attempts)
            self.max_attempts = int(event.get("max_attempts") or self.max_attempts)
            changed |= self._append(
                "final",
                "- ✅ **完成：** 已生成最终结果"
                if passed
                else "- ⚠️ **完成：** 已停止，但未通过全部验证",
                milestone=True,
            )
            force = True

        if not changed:
            return ""
        if not force and now - self.last_render < 1.5:
            return ""
        self.last_render = now
        return self._render()
