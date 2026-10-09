"""LangGraph workflow: shared state, prompts, the supervisor/verify nodes, the graph.

The pieces are deliberately importable one at a time, because they are useful at
different levels:

* :mod:`~suuclaw.graph.state` — the channel schema.
* :mod:`~suuclaw.graph.nodes` — the nodes, the delegation tools and the
  router, each callable on its own with a hand-built state (which is how they
  are unit-tested).
* :mod:`~suuclaw.graph.prompts` — the request builders and the prompt re-exports.
* :mod:`~suuclaw.graph.memory` — the three-layer memory system (rules,
  working memory, compressed history) and the snapshot builder prompts embed.
* :mod:`~suuclaw.graph.workflow` — the only module that knows the topology.

::

    from suuclaw.core import RuntimeState
    from suuclaw.graph import build_workflow, initial_state

    app = build_workflow()
    result = app.invoke(initial_state("add a --version flag", RuntimeState.create("./ws")))
    print(result["final_answer"])
"""

from __future__ import annotations

from .memory import (
    HISTORY_SUMMARY_FILENAME,
    RULES_LAYER,
    CompressionEvent,
    LayeredMemory,
    append_memory_block,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    read_history_summary,
)
from .nodes import (
    CONTEXT_TOKEN_LIMIT_ENVS,
    DEFAULT_CODE_AGENT_LOOPS,
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    DEFAULT_INTENT_CONFIDENCE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_PLANNER_LOOPS,
    DEFAULT_VERIFIER_LOOPS,
    CallCodeAgentTool,
    CallSearchAgentTool,
    GraphStateError,
    chat_responder_node,
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    extract_json_object,
    final_answer_text,
    final_node,
    intent_route_fn,
    intent_router_node,
    make_chat_responder_node,
    make_code_agent_tool,
    make_context_compressor_node,
    make_context_monitor_node,
    make_final_node,
    make_intent_router_node,
    make_planner_node,
    make_search_agent_tool,
    make_verifier_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from .prompts import (
    FINAL_PROMPT,
    PLANNER_PROMPT,
    PLANNER_REVISION_ADDENDUM,
    VERIFIER_PROMPT,
)
from .state import (
    AgentHandoff,
    SourceItem,
    SuuClawGraphState,
    TodoItem,
    VerificationCheck,
    VerificationResult,
)
from .workflow import (
    CONTEXT_COMPRESSOR_ROUTE_TARGETS,
    CONTEXT_MONITOR_ROUTE_TARGETS,
    INTENT_ROUTE_TARGETS,
    ROUTE_TARGETS,
    build_complex_state_graph,
    build_complex_workflow,
    build_entry_workflow,
    build_state_graph,
    build_workflow,
    initial_state,
)

__all__ = [
    "AgentHandoff",
    "CallCodeAgentTool",
    "CallSearchAgentTool",
    "CONTEXT_COMPRESSOR_ROUTE_TARGETS",
    "CONTEXT_MONITOR_ROUTE_TARGETS",
    "CONTEXT_TOKEN_LIMIT_ENVS",
    "CompressionEvent",
    "DEFAULT_CODE_AGENT_LOOPS",
    "DEFAULT_CONTEXT_TOKEN_LIMIT",
    "DEFAULT_INTENT_CONFIDENCE",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_PLANNER_LOOPS",
    "DEFAULT_VERIFIER_LOOPS",
    "FINAL_PROMPT",
    "GraphStateError",
    "HISTORY_SUMMARY_FILENAME",
    "INTENT_ROUTE_TARGETS",
    "LayeredMemory",
    "SuuClawGraphState",
    "PLANNER_PROMPT",
    "PLANNER_REVISION_ADDENDUM",
    "ROUTE_TARGETS",
    "RULES_LAYER",
    "SourceItem",
    "TodoItem",
    "VERIFIER_PROMPT",
    "VerificationCheck",
    "VerificationResult",
    "append_memory_block",
    "build_complex_state_graph",
    "build_complex_workflow",
    "build_entry_workflow",
    "build_layered_memory",
    "build_state_graph",
    "build_workflow",
    "chat_responder_node",
    "context_compressor_node",
    "context_compressor_route",
    "context_monitor_node",
    "context_monitor_route",
    "extract_json_object",
    "final_answer_text",
    "final_node",
    "format_layered_memory_for_prompt",
    "initial_state",
    "intent_route_fn",
    "intent_router_node",
    "make_chat_responder_node",
    "make_code_agent_tool",
    "make_context_compressor_node",
    "make_context_monitor_node",
    "make_final_node",
    "make_intent_router_node",
    "make_planner_node",
    "make_search_agent_tool",
    "make_verifier_node",
    "memory_event",
    "planner_node",
    "read_history_summary",
    "verifier_node",
    "verifier_route",
]
