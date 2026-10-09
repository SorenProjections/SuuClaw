"""Command-risk classification and human approval for the ``bash`` tool.

A shell command is arbitrary code, so the workspace boundary that keeps
``file_*`` and ``grep`` honest does nothing here. This module does not pretend
to fix that — it is a *speed bump*, not a sandbox. It recognises the handful of
command shapes that reach outside the workspace or start something that
outlives the call (package installs, network downloads, long-running dev
servers) and routes them to whoever is watching the stream before they run.

Three modes, three policies:

* ``auto`` — run it, and flag the result, so the transcript still records that
  the run touched something riskier than a read or a write.
* ``deny`` — refuse it, and tell the model *why*, so it can find another route
  instead of retrying the same command.
* ``inline`` — ask the human on the stream. With nobody to ask (no handler
  configured) the command runs flagged rather than refused: a headless run that
  cannot ``pip install`` is a run that cannot do its job, and the flag keeps the
  decision auditable afterwards.

Classification is shape-based, not semantic: a regex cannot see through
``python -c "import subprocess; ..."`` and this is not trying to win that fight.
It catches the *honestly written* form of each risky command, which is the form
a model actually emits when it reaches for one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from rich.console import Console
from rich.prompt import Confirm
from rich.text import Text

__all__ = [
    "DEFAULT_APPROVAL_MODE",
    "RISK_PATTERNS",
    "VALID_APPROVAL_MODES",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalRequest",
    "classify_command_risk",
    "console_approval_handler",
    "normalize_approval_mode",
]

#: Every pattern is anchored at a command position — the start of the line, or
#: just after ``&&`` / ``||`` / ``;`` — so ``echo "pip install"`` is not mistaken
#: for an install. ``sudo pip install`` slips through, and that is accepted: this
#: list is a speed bump for the ordinary shape of each command, not a parser.
RISK_PATTERNS: list[tuple[str, str]] = [
    (r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b", "Python package installation"),
    (r"(?:^|&&|\|\||;)\s*uv\s+add\b", "Project dependency change with uv add"),
    (r"(?:^|&&|\|\||;)\s*uv\s+sync\b", "Dependency synchronization with uv sync"),
    (r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b", "Python package installation with uv pip"),
    (r"(?:^|&&|\|\||;)\s*npm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*pnpm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b", "Network download command"),
    (r"(?:^|&&|\|\||;)\s*uvicorn\b", "Long-running development server"),
    (r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b", "Long-running development server"),
]

#: Compiled once, with ``IGNORECASE``: the patterns above are lowercase, but a
#: model writing ``Pip install`` or ``CURL`` means the same command, and shell
#: command names are case-insensitive on Windows anyway.
_COMPILED_RISK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in RISK_PATTERNS
)

#: The three approval policies. ``inline`` is the default because it is the only
#: one that can actually ask — the other two decide without a human in the loop.
VALID_APPROVAL_MODES: set[str] = {"inline", "auto", "deny"}
DEFAULT_APPROVAL_MODE = "inline"


def classify_command_risk(command: str) -> str | None:
    """Return why ``command`` is risky, or ``None`` when it looks safe.

    The answer is the *class* of risk, written for a human reading the
    transcript ("Python package installation"), not the matched text — the
    caller already has the command and shows it alongside this reason.
    """
    if not command or not command.strip():
        return None
    for pattern, reason in _COMPILED_RISK_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def normalize_approval_mode(mode: str | None) -> str:
    """Coerce ``mode`` to one of :data:`VALID_APPROVAL_MODES`, defaulting to inline.

    An unknown mode is *not* an error at the call site: the value arrives from a
    CLI flag or a config file, and refusing every risky command because a typo
    said ``denny`` would be a worse outcome than falling back to the mode that
    asks a human.
    """
    if not mode:
        return DEFAULT_APPROVAL_MODE
    normalized = str(mode).strip().lower()
    return normalized if normalized in VALID_APPROVAL_MODES else DEFAULT_APPROVAL_MODE


@dataclass(frozen=True)
class ApprovalRequest:
    """One risky command, waiting for a verdict.

    ``id`` is declared *last* even though it is the first thing a reader wants,
    because a dataclass requires fields with defaults to come after fields
    without one: ``command`` and ``risk_reason`` are mandatory, ``id`` is
    generated. Keyword construction — how the tool and every handler build it —
    is the same either way, and the id exists to be quoted in a transcript.
    """

    command: str
    risk_reason: str
    id: str = field(default_factory=lambda: f"approval-{uuid4().hex[:8]}")
    tool_name: str = "BashTool"


@dataclass(frozen=True)
class ApprovalDecision:
    """A verdict on an :class:`ApprovalRequest`.

    ``reason`` travels with the decision because it is shown to the model when
    the command is refused: a rejection that only says "no" invites a retry of
    the very command that was refused.
    """

    approved: bool
    reason: str = ""


#: The shape of a decision-maker: called with a request, answers with a verdict.
ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


def console_approval_handler(console: Console) -> ApprovalHandler:
    """Build an inline handler that asks the human on ``console``.

    Answering reads stdin, so this belongs on an interactive run only. A console
    that is not a terminal — a piped or ``--quiet`` run — answers *no* instead
    of prompting: a question nobody can see must not be able to hang the run, and
    a default that silently approves would make the prompt theatre. The default
    is the safe answer, ``False``: the operator has to type ``y``.
    """

    def handler(request: ApprovalRequest) -> ApprovalDecision:
        if not console.is_terminal:
            return ApprovalDecision(
                approved=False,
                reason="no interactive terminal, so nobody could be asked",
            )
        console.print(Text.assemble(("⚠ approval requested ", "bold yellow"), (request.id, "dim")))
        console.print(Text(f"  {request.command}", style="bold"))
        console.print(Text(f"  risk: {request.risk_reason}", style="yellow"))
        try:
            approved = Confirm.ask("  run it?", console=console, default=False)
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision(
                approved=False,
                reason="the approval prompt was interrupted",
            )
        return ApprovalDecision(
            approved=approved,
            reason="approved by the user" if approved else "declined by the user",
        )

    return handler
