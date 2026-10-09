"""Read-only presentation of the graphs the session driver actually executes.

Compile without a model, then read LangGraph's own nodes and edges. The entry
graph's hand-off is joined to the main graph exactly where the session driver
joins them. Specialists are explicitly marked as delegations inside the
supervisor, rather than invented LangGraph nodes or parallel execution paths.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .workflow import build_complex_workflow, build_entry_workflow

# Presentation only: execution nodes and edges are still read from LangGraph.
# Positions are centres in an 840 x 610 diagram; English names and Chinese
# subtitles deliberately separate familiar AI terms from each node's purpose.
_PRESENTATION = {
    "__start__": ("Input", "消息输入", "boundary", "本轮会话的用户消息。", 64, 81, "entry"),
    "intent_router": (
        "Intent Router",
        "意图路由",
        "router",
        "判断直接回答还是交给 Supervisor 使用工具完成任务。",
        224,
        81,
        "entry",
    ),
    "chat_responder": (
        "Chat Responder",
        "直接对话",
        "agent",
        "无需工具的会话在这里生成答复，不经过独立验证。",
        474,
        81,
        "entry",
    ),
    "planner": (
        "Supervisor",
        "任务编排",
        "supervisor",
        "制定计划，按需委派研究或实现专家；接收结果后再继续编排。",
        188,
        293,
        "workflow",
    ),
    "context_monitor": (
        "Context Monitor",
        "上下文监测",
        "context",
        "检查上下文预算，根据已记录的下一步决定验证、重试、压缩或交付。",
        445,
        293,
        "workflow",
    ),
    "context_compressor": (
        "Context Compaction",
        "上下文压缩",
        "context",
        "压缩已有上下文，再回到压缩前指定的目标节点。",
        610,
        486,
        "context",
    ),
    "verifier": (
        "Verifier",
        "结果验证",
        "verifier",
        "用只读工具检查产物与验收条件；未通过时可返回 Supervisor 重试。",
        688,
        293,
        "workflow",
    ),
    "final": (
        "Final Response",
        "结果交付",
        "output",
        "整理本轮结果与验证结论；交付并不自动代表验证通过。",
        674,
        172,
        "delivery",
    ),
    "__end__": (
        "End",
        "本轮结束",
        "boundary",
        "本轮已结束，是否成功以实际结果为准。",
        777,
        81,
        "delivery",
    ),
    "search": (
        "Search Agent",
        "研究专家",
        "specialist",
        "在主管内部运行搜索循环，收集资料并返回来源与摘要。",
        137,
        486,
        "specialists",
    ),
    "code": (
        "Code Agent",
        "实现专家",
        "specialist",
        "在主管内部运行工具循环，读取、修改与执行工作区文件。",
        319,
        486,
        "specialists",
    ),
}


def _node(node_id: str, *, source: str, position: int = 0) -> dict[str, Any]:
    fallback = (
        node_id,
        "工作流节点",
        "node",
        "实际工作流中的节点。",
        100 + position * 180,
        580,
        "workflow",
    )
    title, subtitle, kind, description, x, y, section = _PRESENTATION.get(node_id, fallback)
    return {
        "id": node_id,
        "title": title,
        "label": subtitle,
        "subtitle": subtitle,
        "kind": kind,
        "description": description,
        "source": source,
        "x": x,
        "y": y,
        "section": section,
    }


_EDGE_PRESENTATION = {
    ("__start__", "intent_router"): ("消息进入", "primary"),
    ("intent_router", "chat_responder"): ("直接对话", "primary"),
    ("intent_router", "planner"): ("需要执行", "primary"),
    ("chat_responder", "__end__"): ("返回答复", "primary"),
    ("planner", "context_monitor"): ("提交状态", "primary"),
    ("context_monitor", "verifier"): ("进入验证", "primary"),
    ("verifier", "context_monitor"): ("回传验证结论", "primary"),
    ("context_monitor", "planner"): ("返回编排 / 重试", "secondary"),
    ("context_monitor", "final"): ("进入交付", "primary"),
    ("context_monitor", "context_compressor"): ("需要压缩", "primary"),
    ("context_compressor", "planner"): ("恢复编排", "secondary"),
    ("context_compressor", "verifier"): ("恢复验证", "secondary"),
    ("context_compressor", "final"): ("恢复交付", "secondary"),
    ("final", "__end__"): ("返回结果", "primary"),
}


def _edge(raw: dict[str, Any], *, source: str, **changes: Any) -> dict[str, Any]:
    value = {
        "source": raw["source"],
        "target": raw["target"],
        "conditional": bool(raw.get("conditional")),
        "kind": "flow",
        "graph": source,
    }
    if raw.get("data"):
        value["label"] = str(raw["data"])
    value.update(changes)
    caption, emphasis = _EDGE_PRESENTATION.get(
        (value["source"], value["target"]), (str(value.get("label", "连接")), "secondary")
    )
    value.update(caption=caption, emphasis=emphasis)
    return value


@lru_cache(maxsize=1)
def describe_workflow() -> dict[str, Any]:
    """Return JSON-ready topology, with no task, tools, API calls or writes.

    Every workflow edge comes from the compiled graph. The two specialist
    tool names and event identities come from their actual implementations.
    Consumers must treat this cached object as read-only.
    """
    from ..agents.code_agent import CODE_NODE_NAME
    from ..agents.search_agent import SEARCH_NODE_NAME
    from .nodes import CallCodeAgentTool, CallSearchAgentTool

    entry = build_entry_workflow().get_graph().to_json()
    workflow = build_complex_workflow().get_graph().to_json()
    nodes: dict[str, dict[str, Any]] = {}
    for source, graph in (("entry", entry), ("workflow", workflow)):
        for item in graph["nodes"]:
            node_id = str(item["id"])
            nodes.setdefault(node_id, _node(node_id, source=source, position=len(nodes)))

    starts = [e["target"] for e in workflow["edges"] if e["source"] == "__start__"]
    edges = []
    for item in entry["edges"]:
        if item["target"] == "__end__" and item.get("data") == "planner":
            edges.extend(
                _edge(item, source="entry", target=target, kind="handoff", label="交给工作流")
                for target in starts
            )
        else:
            edges.append(_edge(item, source="entry"))
    edges.extend(
        _edge(item, source="workflow")
        for item in workflow["edges"]
        if item["source"] != "__start__"
    )
    for event_node, tool in (
        (SEARCH_NODE_NAME, CallSearchAgentTool),
        (CODE_NODE_NAME, CallCodeAgentTool),
    ):
        nodes[event_node] = {**_node(event_node, source="delegation"), "parent": "planner"}
        edges.append(
            {
                "source": "planner",
                "target": event_node,
                "kind": "delegation",
                "conditional": True,
                "label": "按需委派",
                "tool": tool.name,
                "graph": "delegation",
                "caption": "按需委派 / 返回结果",
                "emphasis": "primary",
            }
        )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "width": 840,
        "height": 610,
        "description": "主路径展示编排与验证；专家在主管内部按需调用。完整拓扑保留全部条件与回路。",
    }
