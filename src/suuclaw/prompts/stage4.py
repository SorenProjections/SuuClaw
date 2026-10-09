"""Stage 4 prompts: context compression.

The ``context_compressor`` node (:mod:`suuclaw.graph.nodes`) is the one reader
of :data:`CONTEXT_COMPRESSION_PROMPT`. The prompt asks for a *structured* JSON
digest rather than prose on purpose: the node folds each key into the channels
and the ``HISTORY_SUMMARY.md`` file, so a missing key degrades one section
instead of costing the whole summary.
"""

from __future__ import annotations

__all__ = ["CONTEXT_COMPRESSION_PROMPT"]

#: System prompt for the context_compressor node. Verbatim contract: the model
#: returns exactly one JSON object with the nine keys listed at the bottom.
CONTEXT_COMPRESSION_PROMPT = """You are the context_compressor node in SuuClaw stage 4.
Your job is to compress the graph context so the task can continue with a much
smaller message window.
Keep everything needed to resume work:
- user task and active goal
- current plan, todos, acceptance criteria, verification commands
- completed work and current files/artifacts
- important tool findings and command results
- research notes and source URLs
- latest verifier failure and recommended next step
- risks, blockers, and assumptions
Remove redundant transcript detail:
- repeated tool calls
- long stdout/stderr
- duplicate search snippets
- stale intermediate reasoning
Return only JSON with these keys:
- summary
- active_goal
- completed_work
- open_todos
- important_files
- tool_findings
- sources
- next_steps
- risks
"""
