"""Describe both shared and role-specific tools without constructing an agent."""

from __future__ import annotations

import json
from typing import Any

from ..core.state import RuntimeState


def describe_workflow_tools(state: RuntimeState) -> list[dict[str, Any]]:
    """Read descriptions and schemas from the same classes used by each role."""
    from ..graph.nodes import CallCodeAgentTool, CallSearchAgentTool
    from ..tools.notepad_tools import NotepadAppendTool, NotepadReadTool
    from ..tools.registry import describe_tools
    from ..tools.todo_tools import TodoUpdateTool, TodoWriteTool
    from ..tools.web_search_tool import WebSearchTool

    rows = [
        {"name": name, "description": description, "signature": signature}
        for name, description, signature in describe_tools(state)
    ]
    for tool in (
        TodoWriteTool,
        TodoUpdateTool,
        CallSearchAgentTool,
        CallCodeAgentTool,
        WebSearchTool,
        NotepadReadTool,
        NotepadAppendTool,
    ):
        rows.append({
            "name": tool.name,
            "description": tool.description,
            "signature": json.dumps(tool.args_schema.model_json_schema(), ensure_ascii=False),
        })
    return rows
