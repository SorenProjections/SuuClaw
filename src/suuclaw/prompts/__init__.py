"""Every string a model reads, one module per workflow stage.

Keeping prompt text in its own package — rather than inline in the module that
uses it — is what makes wording reviewable: a change to how the planner is asked
for a plan shows up as a diff in a file that contains no control flow, and can be
unit-tested without building a graph.

Stages:

* ``stage1`` — the single-agent ReAct loop (``suuclaw.core.agent``). Not written
  yet; :data:`~suuclaw.prompts.stage2.ACTOR_PROMPT` is what that loop uses.
* ``stage2`` — the plan / act / verify graph (``suuclaw.graph``): the actor
  and final prompts, plus the revision addendum.
* ``stage3`` — the multi-agent flow (``suuclaw.graph`` supervisor +
  ``suuclaw.agents``): the planner-as-supervisor, the verifier and one
  specialist per job.
* ``stage4`` — context management (``suuclaw.graph.nodes``): the
  context_compressor's digest prompt.
* ``stage5`` — the entry graph (``suuclaw.graph``): the intent router's
  classification prompt and the chat responder's no-tools prompt.

Re-exported here so callers can write ``from suuclaw.prompts import
VERIFIER_PROMPT`` without knowing which stage module holds it.
"""

from __future__ import annotations

from .stage2 import ACTOR_PROMPT, FINAL_PROMPT, PLANNER_REVISION_ADDENDUM
from .stage3 import CODE_AGENT_PROMPT, PLANNER_PROMPT, SEARCH_AGENT_PROMPT, VERIFIER_PROMPT
from .stage4 import CONTEXT_COMPRESSION_PROMPT
from .stage5 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT

__all__ = [
    "ACTOR_PROMPT",
    "CHAT_RESPONDER_PROMPT",
    "CODE_AGENT_PROMPT",
    "CONTEXT_COMPRESSION_PROMPT",
    "FINAL_PROMPT",
    "INTENT_ROUTER_PROMPT",
    "PLANNER_PROMPT",
    "PLANNER_REVISION_ADDENDUM",
    "SEARCH_AGENT_PROMPT",
    "VERIFIER_PROMPT",
]
