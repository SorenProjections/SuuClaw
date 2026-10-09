"""Stage-3 agents: single-purpose specialists a workflow can hand one step to.

The stage-3 supervisor (the graph's planner node) cannot touch the workspace or
the web itself — it delegates whole steps to specialists through tools. Each
agent lives in its own module, is bound to exactly the tools its job needs, and
returns a plain dict so the delegation can fold the result into graph state.

``search_agent`` researches the web; ``code_agent`` implements in the
workspace. Both are complete today.

Importing this package pulls in the agent modules, which import the tool layer,
which imports ``core.state`` — safe in that direction. Nothing here imports
``graph`` or ``cli``, so an agent can be driven from a node, a script or a test
without a graph existing.
"""

from __future__ import annotations

from .code_agent import (
    CODE_AGENT_PROMPT,
    CODE_NODE_NAME,
    DEFAULT_CODE_LOOPS,
    CodeAgentError,
    make_code_agent,
    run_code_agent,
)
from .search_agent import (
    DEFAULT_SEARCH_LOOPS,
    SEARCH_AGENT_PROMPT,
    SEARCH_NODE_NAME,
    SearchAgentError,
    make_search_agent,
    run_search_agent,
)

__all__ = [
    "CODE_AGENT_PROMPT",
    "CODE_NODE_NAME",
    "DEFAULT_CODE_LOOPS",
    "CodeAgentError",
    "DEFAULT_SEARCH_LOOPS",
    "SEARCH_AGENT_PROMPT",
    "SEARCH_NODE_NAME",
    "SearchAgentError",
    "make_code_agent",
    "make_search_agent",
    "run_code_agent",
    "run_search_agent",
]
