"""Tool layer: the tool contract, the registry, and the built-in tools.

Kept deliberately empty of concrete-tool imports: ``registry`` imports
``core.state`` and ``core.agent`` imports back into ``registry``, so pulling
``bash_tool``/``file_tools`` in at package-import time re-creates the cycle.
Reach for ``.bash_tool`` / ``.file_tools`` at the point of use instead.
"""

from __future__ import annotations

from .registry import SuuClawTool, build_tools, describe_tools, tool_names

__all__ = [
    "SuuClawTool",
    "build_tools",
    "describe_tools",
    "tool_names",
]
