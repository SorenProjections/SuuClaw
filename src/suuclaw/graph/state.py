"""Shared state for SuuClaw's LangGraph workflow.

One ``SuuClawGraphState`` dict flows through every node — plan, act, verify — so
each step reads what earlier steps wrote instead of having arguments threaded
through call signatures. LangGraph reads the ``TypedDict`` as the graph's
*channel schema*: every key is a channel, and a node returns only the keys it
changed (a partial update); everything else keeps its previous value.

The channels are last-write-wins by default, which is exactly what ``passed`` or
``final_answer`` want. The one exception is ``messages``: it carries an
:func:`~langgraph.graph.message.add_messages` reducer, so a node returning
``{"messages": [reply]}`` *appends* to the transcript instead of clobbering it.
That is what keeps the conversation intact without any node having to read and
re-emit the whole history.

Three row shapes are shared across nodes:

* :class:`TodoItem` — one line of the plan, whose ``status`` the actor advances.
* :class:`VerificationResult` — the outcome of one acceptance *command*, captured
  deterministically by the verifier node running it.
* :class:`VerificationCheck` — one item of the verifier *model*'s judgement, which
  may be about evidence no command can express ("the README explains the flags").

The layered-memory shapes (:class:`CompressionEvent`,
:class:`LayeredMemory`) are defined in :mod:`~suuclaw.graph.memory` — the
memory system owns its own types — and re-exported here so the channel schema
stays the one place a reader looks them up.

Note:
    ``total=False`` makes every key optional, so a node may return a subset and
    the initial invocation needs only ``task`` + ``runtime``. Read with
    ``state.get("todos", [])`` rather than ``state["todos"]``.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..core.state import RuntimeState
from .memory import CompressionEvent, LayeredMemory

__all__ = [
    "AgentHandoff",
    "CompressionEvent",
    "LayeredMemory",
    "SuuClawGraphState",
    "SourceItem",
    "TodoItem",
    "VerificationCheck",
    "VerificationResult",
]


class TodoItem(TypedDict):
    """One line of the working plan.

    ``status`` walks ``pending`` → ``in_progress`` → ``completed``; ``blocked``
    is the side exit taken when the worker cannot make progress, and ``note``
    should then say why.
    """

    id: str
    content: str
    status: str  # "pending" | "in_progress" | "completed" | "blocked"
    note: str


class VerificationResult(TypedDict):
    """Outcome of running one command from ``verification_commands``.

    ``exit_code`` is ``None`` only when the command never produced one — a
    timeout or a spawn failure — in which case ``stderr`` carries the reason.
    """

    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class VerificationCheck(TypedDict):
    """One item of the verifier model's judgement.

    Separate from :class:`VerificationResult` on purpose: a command answers
    "did it exit 0?", which is not the same question as "is the acceptance
    criterion actually met?". Both are kept so the planner can tell a broken
    build apart from a wrong build.
    """

    name: str
    passed: bool
    detail: str


class SourceItem(TypedDict, total=False):
    """One web source the search agent collected, for citation and dedup."""

    title: str
    url: str
    score: float


class AgentHandoff(TypedDict, total=False):
    """One delegation from the planner to a specialist, with the outcome.

    The planner cannot do specialist work itself — it can only hand an
    instruction to ``searchAgent`` or ``codeAgent`` — so the handoff *record*
    is the audit trail of who was asked what and what came back. ``result``
    holds the specialist's own summary, which is what the planner reads to
    decide the next delegation.
    """

    from_agent: str
    to_agent: str
    instruction: str
    result: str


class SuuClawGraphState(TypedDict, total=False):
    """State shared by every node in the SuuClaw graph.

    Channels:
        task: The user's original instruction, verbatim.
        runtime: Sandbox policy (workspace root, command timeout, output budget)
            handed to every tool. Built once per run and then only read —
            nodes must not mutate it.
        messages: An append-only trace of what the nodes did — the AI and Tool
            messages the actor produced. Reducer-merged, never overwritten.
            Nodes build their prompts from ``task`` / ``todos`` instead of
            replaying this channel, because the scaffolding (system prompt, plan
            digest) differs between attempts; replaying it would splice several
            half-conversations together.
        last_actor_summary: The actor's closing prose from its latest attempt.
        plan_summary: One-paragraph restatement of the plan, for the prompt.
        todos: The plan as rows the actor ticks off.
        acceptance_criteria: What "done" means, in plain language.
        verification_commands: Shell commands that decide whether it is done.
            Run by the verifier node itself, never by the model.
        verification_results: One entry per command that has been run.
        verification_checks: The verifier model's per-criterion verdicts.
        passed: The verifier's verdict on the latest attempt. Deterministic
            command failures veto a model that said "fine".
        last_error: Why the last attempt failed; the planner reads this to
            decide what to change. Empty once an attempt passes.
        attempts: How many plan→verify rounds have finished.
        max_attempts: The retry budget; the loop stops asking at this count.
        research_notes: What the search agent established, accumulated across
            delegations. The code agent reads this when the task needs
            researched content.
        sources: The web sources behind ``research_notes``, deduplicated by
            URL in first-seen order.
        agent_handoffs: One record per planner→specialist delegation, in the
            order they happened — the run's audit trail of who did what.
        code_agent_summary: The code agent's closing report from the latest
            delegation: files changed and checks run. The verifier treats it
            as a claim, not evidence.
        recommended_next_instruction: The verifier's own idea of what the
            planner should ask a specialist to fix, from its JSON verdict.
            Empty once an attempt passes.
        final_answer: The finished reply, written once the work is accepted.
        context_summary: The running compressed summary of the conversation so
            far, maintained by the compression step; feeds the memory
            snapshot's history layer.
        context_token_count: The estimated tokens the current context occupies.
        context_token_limit: The budget that, once exceeded, trips
            ``context_should_compress``.
        context_should_compress: Whether the next node should compress before
            prompting instead of continuing on the raw transcript.
        context_next_node: Where the run should continue after a compression
            pass — the compression step's answer to "resume from where?".
        compression_events: The audit trail of compression passes already
            applied (see :class:`~suuclaw.graph.memory.CompressionEvent`).
        memory_snapshot: The assembled three-layer memory
            (:class:`~suuclaw.graph.memory.LayeredMemory`), built by
            :func:`~suuclaw.graph.memory.build_layered_memory` — rules,
            working memory and the history summary store in one dict.
        history_summary: The compressed history text. The durable copy lives
            in ``HISTORY_SUMMARY.md``; this channel holds the freshest
            in-memory version, which the memory snapshot prefers over the
            file.
        intent_route: The entry graph's routing decision for the latest input:
            ``"chat"`` or ``"workflow"``. Written only by the intent router
            node; absent until an entry-graph run reaches it.
        intent_reason: The intent router's one-line justification for the
            route, or why the router fell back to the default.
        intent_confidence: The router's normalized confidence, 0.0-1.0. A
            value below the router's threshold means "chat" was overruled.
        chat_response: The chat branch's direct reply. The chat path never
            reaches the main workflow's ``final`` node, so the chat responder
            is the one node that also mirrors it into ``final_answer``.
    """

    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    # stage-2 channel name, kept so old state replays; superseded by code_agent_summary
    last_actor_summary: str
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    verification_checks: list[VerificationCheck]
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    recommended_next_instruction: str
    passed: bool
    last_error: str
    attempts: int
    max_attempts: int
    final_answer: str
    # layered memory (see suuclaw.graph.memory)
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    history_summary: str
    # entry graph (intent routing / chat branch)
    intent_route: str
    intent_reason: str
    intent_confidence: float
    chat_response: str
