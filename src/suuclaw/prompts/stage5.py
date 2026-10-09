"""Stage 5 prompts: the entry graph's intent router and chat responder.

Two readers, two nodes (:mod:`suuclaw.graph.nodes`):

* :data:`INTENT_ROUTER_PROMPT` — the ``intent_router`` node classifies the
  latest input into ``chat`` or ``workflow`` and answers with one JSON object;
  a low-confidence or unparseable answer defaults to ``workflow``, because
  sending real work down the chat branch silently loses it, while sending
  small talk down the workflow branch only costs tokens.
* :data:`CHAT_RESPONDER_PROMPT` — the ``chat_responder`` node answers without
  tools, so the prompt's main job is honesty: never claim workspace access it
  does not have.
"""

from __future__ import annotations

__all__ = ["CHAT_RESPONDER_PROMPT", "INTENT_ROUTER_PROMPT"]

#: System prompt for the intent_router node. Verbatim contract: the model
#: returns exactly one JSON object ``{"route": ..., "reason": ..., "confidence":
#: 0.0}``; anything else is treated as "workflow" by the node.
INTENT_ROUTER_PROMPT = """You are the intent router for SuuClaw.
Classify the user's latest input into exactly one route:
- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A,
or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands,
installing packages, searching the web, checking the current project, verifying a
result, or producing a concrete deliverable.
When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.
Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}
If uncertain, choose workflow.
"""

#: System prompt for the chat_responder node. No tools are bound to it, so the
#: prompt's one hard rule is that the reply must not invent workspace facts.
CHAT_RESPONDER_PROMPT = """You are SuuClaw's lightweight chat node.
Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.
If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""
