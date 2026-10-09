"""The supervisor/verify nodes' request builders, and the re-export of their prompts.

The prompt *text* lives in :mod:`suuclaw.prompts.stage2` (verifier, final, the
revision addendum) and :mod:`suuclaw.prompts.stage3` (the supervisor planner) —
one file per stage, so wording can be reviewed without reading control flow. This
module holds the other half: the code that fills each prompt's user message from
graph state.

The roles have deliberately different instincts baked in:

* the **planner/supervisor** publishes a checkable plan and *delegates* — its
  request shows what the specialists have already produced, so it can decide
  what to hand off next instead of re-asking for finished work;
* the **verifier** is told that the code agent's summary is a *claim, not
  evidence*, and must answer in one strict JSON object;
* the **final** node is told the run is over and it must not claim anything the
  recorded evidence does not support.

Everything is plain text in, plain text out — no prompt template objects — because
the nodes need to interleave the plan, the previous failure and the recorded
command output in an order that reads naturally to a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..prompts.stage2 import FINAL_PROMPT, PLANNER_REVISION_ADDENDUM
from ..prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT

__all__ = [
    "FINAL_PROMPT",
    "PLANNER_PROMPT",
    "PLANNER_REVISION_ADDENDUM",
    "VERIFIER_PROMPT",
    "final_request",
    "initial_plan_request",
    "render_checks",
    "render_results",
    "render_sources",
    "render_todos",
    "revision_plan_request",
    "verifier_request",
]

#: Per-block cap. The plan and the evidence both feed several prompts, and a
#: single runaway command output must not crowd out the acceptance criteria.
MAX_BLOCK_CHARS = 4000


def _clip(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (+{len(text) - limit} more character(s) withheld)"


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def render_todos(todos: Sequence[Mapping[str, Any]] | None) -> str:
    """The plan as a prompt block, or a note that there is none yet."""
    rows = list(todos or [])
    if not rows:
        return "(no plan yet)"
    lines = []
    for row in rows:
        line = f"- [{row.get('status', 'pending')}] {row.get('id', '?')}: {row.get('content', '')}"
        note = str(row.get("note") or "").strip()
        if note:
            line += f"  (note: {note})"
        lines.append(line)
    return "\n".join(lines)


def render_results(results: Sequence[Mapping[str, Any]] | None) -> str:
    """Verification command outcomes, with exit codes and clipped output."""
    rows = list(results or [])
    if not rows:
        return "(no verification commands were run)"
    blocks = []
    for row in rows:
        exit_code = row.get("exit_code")
        shown_exit = exit_code if exit_code is not None else "none (timeout or spawn failure)"
        head = f"$ {row.get('command', '')}\nexit code: {shown_exit}"
        stdout = _clip(str(row.get("stdout") or ""), 1500)
        stderr = _clip(str(row.get("stderr") or ""), 1500)
        body = [head]
        if stdout:
            body += ["--- stdout ---", stdout]
        if stderr:
            body += ["--- stderr ---", stderr]
        if not stdout and not stderr:
            body.append("(no output)")
        blocks.append("\n".join(body))
    return "\n\n".join(blocks)


def render_checks(checks: Sequence[Mapping[str, Any]] | None) -> str:
    """The previous verifier's per-criterion findings."""
    rows = list(checks or [])
    if not rows:
        return "(the previous attempt has no recorded checks)"
    lines = []
    for row in rows:
        verdict = "PASS" if row.get("passed") else "FAIL"
        line = f"- [{verdict}] {row.get('name', '')}"
        detail = str(row.get("detail") or "").strip()
        if detail:
            line += f"\n    {detail}"
        lines.append(line)
    return "\n".join(lines)


def render_sources(sources: Sequence[Mapping[str, Any]] | None) -> str:
    """The web sources the search agent collected so far."""
    rows = list(sources or [])
    if not rows:
        return "(no sources collected yet)"
    lines = []
    for row in rows[:20]:
        title = str(row.get("title") or row.get("url") or "").strip()
        url = str(row.get("url") or "").strip()
        lines.append(f"- {title} — {url}" if url else f"- {title}")
    if len(rows) > 20:
        lines.append(f"(+{len(rows) - 20} more)")
    return "\n".join(lines)


def _handoff_lines(handoffs: Sequence[Mapping[str, Any]] | None) -> str:
    """The delegation history, newest last — the supervisor's own memory."""
    rows = list(handoffs or [])
    if not rows:
        return "(no specialist has been called yet)"
    lines = []
    for row in rows:
        outcome = str(row.get("result") or "").strip()
        shown = _clip(outcome, 400) if outcome else "(no result recorded)"
        instruction = _clip(str(row.get("instruction") or ""), 300)
        lines.append(f"- planner → {row.get('to_agent', '?')}: {instruction}\n    result: {shown}")
    return "\n".join(lines)


def _bullet_list(items: Sequence[Any] | None, *, empty: str = "(none given)") -> str:
    rows = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not rows:
        return empty
    return "\n".join(f"- {row}" for row in rows)


def _context(state: Mapping[str, Any]) -> str:
    return f"# Task\n{_clip(str(state.get('task') or '(no task given)'), 2000)}"


# --------------------------------------------------------------------------- #
# request builders
# --------------------------------------------------------------------------- #
def initial_plan_request(state: Mapping[str, Any]) -> str:
    """First-pass supervision request: the task, and what is already known."""
    return "\n\n".join(
        [
            _context(state),
            f"# Workspace notes already gathered\n{render_sources(state.get('sources'))}",
            f"# Delegations so far\n{_handoff_lines(state.get('agent_handoffs'))}",
            "# Your job\n"
            "Publish the plan with `todo_write`, then delegate the work: research "
            "questions to `call_search_agent`, implementation to `call_code_agent`. "
            "Read each specialist's report before deciding the next delegation, and "
            "end with a concise supervisor summary.",
        ]
    )


def revision_plan_request(state: Mapping[str, Any]) -> str:
    """Revision request: the current plan plus exactly why it failed."""
    commands = _bullet_list(state.get("verification_commands"))
    reason = _clip(str(state.get("last_error") or "(no reason recorded)"))
    observed = render_results(state.get("verification_results"))
    findings = render_checks(state.get("verification_checks"))
    worker = _clip(str(state.get("code_agent_summary") or ""), 2000)
    blocks = [
        _context(state),
        f"# Current plan\n{render_todos(state.get('todos'))}",
        f"# Acceptance criteria\n{_bullet_list(state.get('acceptance_criteria'))}",
        f"# Verification commands (already run by the harness)\n{commands}",
        f"# What the verifier actually observed\n{observed}",
        f"# Per-criterion findings\n{findings}",
        f"# The code agent's last report\n{worker or '(none)'}",
        f"# Why the attempt failed\n{reason}",
    ]

    # The verifier's own idea of the next fix, exactly as its JSON promised.
    # Empty on a pass; on a failure the supervisor still decides whether to
    # follow it — this is advice, not an order.
    next_fix = str(state.get("recommended_next_instruction") or "").strip()
    if next_fix:
        blocks.append(f"# The verifier's recommended next fix\n{_clip(next_fix, 1000)}")

    blocks.append(
        "# Your job\n"
        "Revise the plan with `todo_write` (send the COMPLETE plan), then "
        "delegate only the missing fix to `call_code_agent` — do not redo work "
        "that already passed. End with a concise supervisor summary."
    )
    return "\n\n".join(blocks)


def verifier_request(state: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> str:
    """The verifier's brief: criteria, deterministic evidence, then the worker's claim.

    The stage-3 verifier may also read the notepad and judge researched content,
    so the brief carries whatever the search step established — the prompt's
    "confirm the output cites useful sources" needs the sources to confirm
    against, not just the code agent's word that they were used.
    """
    blocks = [
        _context(state),
        f"# Plan that was executed\n{render_todos(state.get('todos'))}",
        f"# Acceptance criteria to judge\n{_bullet_list(state.get('acceptance_criteria'))}",
        "# Verification command results (run by the harness — these are facts)\n"
        f"{render_results(results)}",
    ]

    summary = str(state.get("code_agent_summary") or state.get("last_actor_summary") or "").strip()
    if summary:
        blocks.append(
            "# The code agent's own summary (a CLAIM — verify it, do not repeat it)\n"
            f"{_clip(summary, 2000)}"
        )

    notes = str(state.get("research_notes") or "").strip()
    if notes:
        blocks.append(f"# Research notes from the search agent\n{_clip(notes, 2000)}")
        blocks.append(f"# Sources collected\n{render_sources(state.get('sources'))}")

    # The prompt names the keys but shows no example; the shape here is what
    # keeps ``extract_json_object`` reliable across models.
    blocks.append(
        "# Your job\n"
        "Inspect the workspace as needed with your read-only tools, judge every\n"
        "criterion, then reply with ONE JSON object and nothing else — no prose\n"
        "before or after, no markdown fence:\n"
        "\n"
        "{\n"
        '  "passed": false,\n'
        '  "reason": "one or two sentences the planner can act on",\n'
        '  "checks": [\n'
        '    {"name": "<the acceptance criterion, quoted>", "passed": false,\n'
        '     "detail": "what you actually saw: file, line, command output"}\n'
        "  ],\n"
        '  "recommended_next_instruction": "what the planner should ask a specialist\n'
        '    to fix; empty string when passed"\n'
        "}\n"
        "\n"
        "`passed` may be true only when EVERY criterion is substantiated. A criterion\n"
        "you could not substantiate is NOT passed; a command exiting 0 is evidence of\n"
        "an exit code, not of a criterion being met."
    )
    return "\n\n".join(blocks)


def final_request(state: Mapping[str, Any]) -> str:
    """The final node's brief: the verdict, the evidence, and the worker's claim.

    Written to be sufficient on its own — a run that ends on the retry budget has
    no successful worker summary to fall back on, so the failure has to be fully
    described here.
    """
    outcome = "PASSED" if state.get("passed") else "NOT PASSED"
    attempts = int(state.get("attempts") or 0)
    worker = str(state.get("code_agent_summary") or state.get("last_actor_summary") or "").strip()
    reason = str(state.get("last_error") or "").strip()

    blocks = [
        _context(state),
        f"# Outcome\nVerification {outcome} after {attempts} attempt(s).",
        f"# Plan that was executed\n{render_todos(state.get('todos'))}",
        f"# Acceptance criteria\n{_bullet_list(state.get('acceptance_criteria'))}",
        "# Verification command results (run by the harness — these are facts)\n"
        f"{render_results(state.get('verification_results'))}",
        f"# Per-criterion findings\n{render_checks(state.get('verification_checks'))}",
    ]
    if worker:
        claim = _clip(worker, 2000)
        blocks.append(f"# The code agent's own summary (a CLAIM, not evidence)\n{claim}")
    if reason:
        blocks.append(f"# Why the run did not pass\n{_clip(reason)}")

    blocks.append(
        "# Your job\n"
        "Write the user-facing reply now, following the rules in your "
        "instructions. Plain prose, no tool calls."
    )
    return "\n\n".join(blocks)
