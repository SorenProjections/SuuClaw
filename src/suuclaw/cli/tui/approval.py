"""Human approval for risky commands, asked through the TUI.

The bash tool's approval gate is a *callback*: an
:class:`~suuclaw.core.approval.ApprovalHandler` takes an
:class:`~suuclaw.core.approval.ApprovalRequest` and returns an
:class:`~suuclaw.core.approval.ApprovalDecision`, and it is called on whichever
thread is running the tool. The one-shot CLI answers it by blocking that thread
on a ``rich`` prompt at the terminal. That cannot work here: the TUI owns the
terminal, the widget tree belongs to the app's event loop, and a tool thread that
touched it would be racing the loop that is waiting for it.

So approval becomes a hand-off in two directions, and this module is that
hand-off:

1. the tool thread :meth:`ApprovalGate.wait`\\ s on a one-shot verdict channel,
2. having posted the gate to the app, which pushes :class:`ApprovalModal`,
3. and the user's answer calls :meth:`ApprovalGate.resolve`, releasing (1).

Nothing here reaches into ``core`` beyond the two frozen value objects, and
nothing here touches the widget tree from a tool thread:
:class:`ApprovalRequestedMessage` is a Textual ``Message``, and ``post_message``
is thread-safe by construction — it detects the foreign thread and hands the
message to the event loop.

The gate refuses, rather than approves, on every ambiguous ending: no answer
inside the timeout, an app that has closed, a second verdict. Approving by
default would turn a prompt nobody can answer into theatre, and the alternative
the user is choosing between is "a risky command did not run".
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Static

from ...core.approval import ApprovalDecision, ApprovalRequest

__all__ = [
    "DEFAULT_APPROVAL_TIMEOUT",
    "REASON_APP_CLOSED",
    "ApprovalGate",
    "ApprovalModal",
    "ApprovalRequestedMessage",
    "deny_pending",
]

#: How long a tool thread waits for a verdict before refusing the command.
#: Five minutes is long enough that an answer is usually a decision rather than a
#: timeout, and short enough that a modal nobody noticed (a minimised window, a
#: user who walked away) does not park the run for good.
DEFAULT_APPROVAL_TIMEOUT = 300.0

#: Why a gate was refused without being asked, when the app goes away mid-run.
REASON_APP_CLOSED = "the app closed while this command was waiting for approval"


class ApprovalGate:
    """A one-shot verdict channel: a tool thread blocks, the UI thread answers.

    Three properties, and each is why a bare ``threading.Event`` is not enough:

    * **One answer wins.** The first :meth:`resolve` decides; a later one is
      ignored, so a modal dismissed twice — by button *and* by key — cannot flip
      a verdict that has already been handed to the tool.
    * **Never blocks for good.** :meth:`wait` gives up after ``timeout`` and
      refuses, because a tool thread parked forever would hang the run and, with
      it, the app.
    * **Safe to abandon.** :func:`deny_pending` answers every outstanding gate
      when the app closes, so quitting mid-run does not leave a thread waiting on
      a modal that will never open.
    """

    def __init__(self, *, timeout: float = DEFAULT_APPROVAL_TIMEOUT) -> None:
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._answered = threading.Event()
        self._decision: ApprovalDecision | None = None

    @property
    def answered(self) -> bool:
        """True once a verdict has been recorded, whether or not it was asked for."""
        return self._answered.is_set()

    def resolve(self, approved: bool, reason: str = "") -> bool:
        """Answer the gate, returning whether this call was the one that answered."""
        with self._lock:
            if self._answered.is_set():
                return False
            if not reason:
                reason = "approved in the TUI" if approved else "declined in the TUI"
            self._decision = ApprovalDecision(approved=bool(approved), reason=reason)
            # Set the event last: a waiter that wakes must find a decision.
            self._answered.set()
            return True

    def wait(self) -> ApprovalDecision:
        """Block until answered, or refuse once :attr:`timeout` has passed."""
        if self._answered.wait(self.timeout):
            # Guaranteed non-None: ``resolve`` writes the decision before it sets
            # the event, and only ``resolve`` can set it.
            return self._decision or ApprovalDecision(approved=False, reason=REASON_APP_CLOSED)
        return ApprovalDecision(
            approved=False,
            reason=f"nobody answered the approval prompt within {self.timeout:.0f}s",
        )


def deny_pending(gates: Iterable[ApprovalGate], reason: str = REASON_APP_CLOSED) -> int:
    """Refuse every gate in ``gates``; returns how many this call answered.

    Best-effort by design: gates already answered are skipped, and the count that
    comes back is what a caller can report, not what it must act on.
    """
    answered = 0
    for gate in list(gates):
        if gate.resolve(False, reason):
            answered += 1
    return answered


class ApprovalRequestedMessage(Message):
    """Posted from a tool thread: an approval request plus the gate to answer it.

    The gate travels with the request because the request alone is not enough to
    reply to — a modal needs the channel, not just the question. See the module
    docstring for why this crosses threads as a message rather than a call.
    """

    def __init__(self, request: ApprovalRequest, gate: ApprovalGate) -> None:
        super().__init__()
        self.request = request
        self.gate = gate


class ApprovalModal(ModalScreen[bool]):
    """Ask a human whether a risky command may run.

    Dismissed with ``True`` to approve and ``False`` to refuse. The buttons and
    the keyboard bindings both land on the same two actions, so a verdict cannot
    depend on which one the user reached for — and ``escape`` refuses, because
    the key everyone presses to *leave* a screen must not be an approval.
    """

    CSS = """
    ApprovalModal {
        align: center middle;
        background: $background 60%;
    }
    #approval-box {
        width: 92;
        max-width: 96%;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    #approval-box Static {
        height: auto;
    }
    #approval-command {
        margin: 1 0;
        padding: 0 1;
        background: $panel;
        border: round $panel-lighten-1;
    }
    #approval-buttons {
        height: auto;
        align-horizontal: right;
    }
    #approval-buttons Button {
        margin-left: 2;
        min-width: 12;
    }
    """

    BINDINGS = [
        Binding("y", "approve", "Approve", show=True),
        Binding("n", "deny", "Deny", show=True),
        # ``enter`` approves — the spec's [Y] Approve is the default action — and
        # is hidden in the footer because two keys for one verdict is already one
        # line of noise too many.
        Binding("enter", "approve", "Approve", show=False),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, request: ApprovalRequest, *, workspace: str = "") -> None:
        super().__init__()
        self.request = request
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(
                Text.assemble(
                    ("⚠ approval requested ", "bold yellow"),
                    (self.request.id, "dim"),
                ),
                id="approval-title",
            )
            yield Static(
                Text.assemble(("tool       ", "dim"), (self.request.tool_name, "bold")),
                id="approval-tool",
            )
            yield Static(
                Text.assemble(("risk       ", "dim"), (self.request.risk_reason, "yellow")),
                id="approval-risk",
            )
            if self.workspace:
                yield Static(
                    Text.assemble(("workspace  ", "dim"), (self.workspace, "dim")),
                    id="approval-workspace",
                )
            yield Static(
                Text.assemble(("command", "\n"), (self.request.command, "bold")),
                id="approval-command",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Approve", variant="success", id="approve")
                yield Button("Deny", variant="error", id="deny")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
