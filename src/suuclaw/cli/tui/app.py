"""The Textual front end: a live widget tree over the session event stream.

Layout::

    ┌──────────────────────────────────────────────────────────────┐
    │ /// SuuClaw                               session: 3f9c1a2e │  Header
    ├──────────────────────────────────────────────────────────────┤
    │ /// SuuClaw                                                 │  splash
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                            │  (animated
    │ Local agents · Work in view                        │   on start,
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                            │   dropped
    ├──────────────────────────────────────────────────────────────┤   at turn 1)
    │ ○ idle · ./workspace · gpt-4o-mini · approval inline          │  status
    │ ╭ Plan · 3 step(s) · 1 done ──────────────────────────────╮  │
    │ │ find the bug, then fix it                               │  │
    │ │ todo-1 ✅  todo-2 🔄  todo-3 ⬜                          │  │
    │ ╰──────────────────────────────────────────────────────────╯  │
    │ 📋 Planner                                                    │
    │ ◆ loop 1  plans CallCodeAgentTool                             │  event
    │   ▸ CallCodeAgentTool  "add the missing import"               │  stream
    │   ⇢ handoff → codeAgent  fix the import and re-run            │
    │ ╭ ✅ final answer · workflow · turn 3 ─────────────────────╮  │
    │ ╰──────────────────────────────────────────────────────────╯  │
    │ 💬 Input: _______________________________________________     │  Input
    └──────────────────────────────────────────────────────────────┘

Two things make this more than a re-skin of :mod:`suuclaw.cli.app`.

**A turn runs on its own thread.** :func:`~suuclaw.core.agent.
stream_session_events` is a *blocking* generator — it consumes LangGraph's
synchronous stream through a pump thread of its own — so it can never run on the
app's event loop. Each turn is pumped by :class:`_TurnRunner` on a daemon thread
that posts one :class:`AgentEventMessage` per event. Textual's ``post_message``
is thread-safe (it detects the foreign thread and hands the message to the loop
with ``call_soon_threadsafe``), and the handler below is the only code that
touches the widget tree — which is what keeps the UI single-threaded as a whole.

**Conversation state is the driver's, not the app's.** The app owns no history:
it passes the same :class:`~suuclaw.core.state.RuntimeState` to every turn, and
:func:`~suuclaw.core.agent.stream_session_events` re-reads the session file and
the workspace listing on each call. So the second turn sees what the first one
created, without this file knowing how any of that is stored.

**The splash is a widget, not a log write.** :class:`~suuclaw.cli.tui.logo.
SplashLogo` plays the startup animation on the app's own clock, and the app drops
it when the first turn starts. See :mod:`suuclaw.cli.tui.logo` for why the
animation cannot be written into the stream (a ``RichLog`` is append-only) and
why the wordmark is ASCII rather than more emoji (an emoji's width is the
terminal's opinion).

Event mapping
-------------
The workflow event catalogue is reused as-is; ``final_answer`` carries three
extra keys (``route``, ``session_id``, ``session_turn``) that only a session-aware
front end sees. Per event type:

==============================  ================================================
``node_start``                  section heading in the stream
``node_output`` (planner)       the Plan panel, from ``todos``
``node_output`` (verifier)      the ✅/❌ verdict and its findings
``node_output`` (monitor)       the token budget line
``node_output`` (router)        the route the gate chose, with its confidence
``ai_message``                  ``◆ loop n`` plus the turn's text, as Markdown
``tool_call`` / ``tool_result`` the call, and a clipped preview of what it said
``handoff``                     the planner's delegation to a specialist
``search_results``              one line per finished search
``error`` / ``retry``           surfaced in red / yellow, never swallowed
``heartbeat``                   a dim liveness line for a silent stretch
``memory``                      a dim line that a memory snapshot was installed
``checkpoint_saved``            the checkpoint's status, when a driver emits it
``final_answer``                the answer panel; closes the turn
``approval_requested``          *not* a stream event — see below
==============================  ================================================

The last row is the one place this file deliberately departs from the sketch's
event list. Approval cannot be a passive emission: the bash tool *blocks* on the
verdict, so it arrives as :class:`~suuclaw.cli.tui.approval.
ApprovalRequestedMessage` — a message carrying the gate to answer — rather than
as a line in the stream. Everything else maps one-to-one.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Footer, Header, Input, RichLog, Static

from ...core import RuntimeState
from ...core.agent import stream_session_events
from ...core.approval import ApprovalDecision, ApprovalRequest
from ...core.session import session_path
from ...graph.nodes import DEFAULT_MAX_ATTEMPTS
from .approval import (
    DEFAULT_APPROVAL_TIMEOUT,
    REASON_APP_CLOSED,
    ApprovalGate,
    ApprovalModal,
    ApprovalRequestedMessage,
    deny_pending,
)
from .logo import SplashLogo

__all__ = ["AgentEventMessage", "SuuClawTuiApp", "run_tui"]

#: Node → the icon and word a section heading uses. The TUI keeps its own table
#: rather than importing the one-shot renderer's: this front end also sees the
#: entry graph's two nodes, which never appear in a plain ``run`` transcript, so
#: sharing the table would mean a reverse dependency for two extra rows.
NODE_LABELS: dict[str, str] = {
    "intent_router": "🎯 Intent",
    "chat_responder": "💬 Chat",
    "planner": "📋 Planner",
    "search": "🔎 Search",
    "code": "🛠 Code",
    "verifier": "🔍 Verifier",
    "final": "📝 Final",
    "context_monitor": "🧠 Context",
    "context_compressor": "🗜 Compressed",
}
NODE_STYLES: dict[str, str] = {
    "intent_router": "yellow",
    "chat_responder": "green",
    "planner": "blue",
    "search": "cyan",
    "code": "cyan",
    "verifier": "magenta",
    "final": "green",
    "context_monitor": "yellow",
    "context_compressor": "yellow",
}

#: Todo status → the mark the Plan panel shows beside a step id.
PLAN_MARKS: dict[str, str] = {
    "completed": "✅",
    "in_progress": "🔄",
    "blocked": "⛔",
    "pending": "⬜",
}

#: Lines of a tool's output shown before the rest is cut. Machine output is
#: previewed, not scrolled: the run's own narrative is what the stream is for,
#: and the full text is a separate concern for a later turn.
PREVIEW_LINES = 12

#: Per-argument clip inside a tool call's one-line summary.
MAX_ARG_CHARS = 48

#: The stream is bounded. A long run with 25-loop delegations produces thousands
#: of lines, and a log that grows without limit is a slow leak in a UI that is
#: meant to stay open across a whole conversation.
MAX_LOG_LINES = 4000

#: The marker the bash tool prepends to a result when the approval gate had a
#: say. Surfaced as its own line because the decision is the news, and because it
#: is currently the only signal of it that reaches an event stream at all —
#: ``CommandResult.requires_approval`` does not travel in the payload.
APPROVAL_MARKER = "approval: required"

#: A placeholder for the unknown-event branch, so a surprising driver change is
#: visible in the stream without a hundred-line dict taking over the panel.
MAX_UNKNOWN_EVENT_CHARS = 200


def _session_id_of(session_workspace: Path) -> str:
    """The id of the conversation already on file, or ``""`` for a new one.

    A pure read, on purpose: showing the header must not create a session file.
    A missing or unreadable file is not an error here — the first turn creates a
    real one, and until then "new" is the honest answer.
    """
    try:
        raw = session_path(session_workspace).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return ""
    if not isinstance(data, Mapping):
        return ""
    return str(data.get("session_id") or "")


def _short(session_id: str) -> str:
    """First eight characters of a session id, the way a trace id is quoted."""
    return session_id[:8] if session_id else "(new)"


def _one_line_args(args: Mapping[str, Any] | None, *, budget: int = 140) -> str:
    """One-line form of a tool call's arguments, clipped to ``budget``.

    A single argument drops its key — ``bash(command="ls -a")`` is a shell
    command, and ``command=`` in front of it is scaffolding the reader has to
    skip past. Two or more keep their keys, because then the pairing is the
    information.
    """
    items = list((args or {}).items())
    if not items:
        return ""
    if len(items) == 1:
        joined = str(items[0][1]).replace("\n", "\\n")
    else:
        parts: list[str] = []
        for key, value in items:
            flat = str(value).replace("\n", "\\n")
            if len(flat) > MAX_ARG_CHARS:
                flat = f"{flat[:MAX_ARG_CHARS]}…"
            parts.append(f"{key}={flat}")
        joined = ", ".join(parts)
    return joined if len(joined) <= budget else f"{joined[: max(budget - 1, 1)]}…"


def _preview(body: str, *, lines: int = PREVIEW_LINES) -> tuple[str, int]:
    """Split tool output into ``(shown, hidden_line_count)``."""
    split = body.splitlines() or [""]
    if len(split) <= lines:
        return body, 0
    return "\n".join(split[:lines]), len(split) - lines


def _split_approval_note(body: str) -> tuple[str, str]:
    """Peel the bash tool's leading ``approval: required (...)`` line off ``body``.

    Returns ``(note, rest)``. See :data:`APPROVAL_MARKER` for why the note is
    worth its own line instead of staying at the top of the output blob.
    """
    head, _, rest = body.partition("\n")
    if head.strip().lower().startswith(APPROVAL_MARKER):
        return head.strip(), rest.lstrip("\n")
    return "", body


def _crash_events(exc: BaseException) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two events a turn gets when the driver itself raised.

    :func:`~suuclaw.core.agent.stream_session_events` turns almost everything
    into events, so a leak here means something hard broke — a bad state, an
    import that failed. The UI still needs a ``final_answer`` or the input box
    would stay disabled for good, and a synthetic one is honest: it says the turn
    ended, and why.
    """
    message = f"{type(exc).__name__}: {exc}"
    return (
        {"type": "error", "node": "", "loop": 0, "message": message},
        {
            "type": "final_answer",
            "node": "final",
            "content": f"The turn stopped before it could report an outcome.\n{message}",
            "passed": False,
            "attempts": 0,
            "max_attempts": 0,
            "loops": 0,
            "exhausted": False,
            "error": message,
            "interrupted": True,
        },
    )


class AgentEventMessage(Message):
    """One workflow event, carried from the turn thread to the UI thread."""

    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


class _TurnRunner:
    """Pumps one turn's blocking event stream on a daemon thread.

    Not a Textual thread worker, and the difference is deliberate. A worker's
    thread comes from the loop's default executor, which the interpreter joins at
    shutdown: a run still inside a 120-second ``bash`` call when the user quits
    would keep the process alive until that call returned, with no window left to
    show why. A daemon thread dies with the process, which is what quitting
    mid-run means.

    :meth:`stop` is a request, not a kill: the pump breaks out at its next event
    and closes the generator, which is what makes the driver record the turn as
    interrupted. A run parked inside a tool call cannot be interrupted from here
    — that call is bounded by its own timeout, not by this thread.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[dict[str, Any]]],
        post: Callable[[dict[str, Any]], None],
    ) -> None:
        self._factory = factory
        self._post = post
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._pump,
            name="suuclaw-tui-turn",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the run to wind up at its next event. Never blocks."""
        self._stop.set()

    def _pump(self) -> None:
        stream: Any = None
        try:
            stream = self._factory()
            for event in stream:
                self._post(event)
                if self._stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - reported as events, not a crash
            for event in _crash_events(exc):
                self._post(event)
        finally:
            # Closed from *this* thread, and only once the loop above has stopped
            # running it: a generator may not be closed from a thread that did not
            # create it while it is executing, but this thread is its owner and is
            # idle here.
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - nothing left to report it to
                    pass


class SuuClawTuiApp(App[None]):
    """A conversation with the agent, one session turn at a time.

    Args:
        state: The run's :class:`~suuclaw.core.state.RuntimeState`. One state
            is shared by every turn, so the approval / checkpoint / trace policy
            is decided once. With ``approval_mode="inline"`` the app replaces the
            state's approval handler on mount — the TUI *is* the terminal that
            answers, and a ``rich`` prompt on stdin would fight it for the same
            keyboard.
        model: The chat model the entry graph and the workflow share. Built by
            :func:`run_tui` when omitted.
        session_workspace: Where ``.suuclaw/session/`` lives; defaults to the
            sandbox workspace, so a plain run keeps one conversation in one root.
        max_attempts / max_loops: The workflow's budget, passed through per turn.
        approval_timeout: Seconds a tool thread waits for a verdict before the
            command is refused.
        logo: Play the startup splash. On by default; off is for a caller that
            has its own header, or a test that does not want to wait for (or
            race) an animation.
    """

    TITLE = "/// SuuClaw"

    CSS = """
    Screen {
        layout: vertical;
        background: #0c1822;
        color: #e7f0f2;
    }
    Header {
        background: #153140;
        color: #a6e8e1;
    }
    Footer {
        background: #122733;
        color: #a6c9d4;
    }
    #status {
        height: 1;
        padding: 0 1;
        background: #122733;
        color: #a6c9d4;
    }
    #plan {
        height: auto;
        max-height: 9;
        border: round #38c8d4;
        padding: 0 1;
    }
    #stream {
        height: 1fr;
        border: round #315361;
        padding: 0 1;
    }
    #prompt {
        border: round #53e1c6;
        background: #122733;
        color: #e7f0f2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_log", "Clear log", show=True),
    ]

    def __init__(
        self,
        *,
        state: RuntimeState,
        model: Any | None = None,
        session_workspace: str | os.PathLike[str] | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_loops: int | None = None,
        approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
        logo: bool = True,
    ) -> None:
        super().__init__()
        self.runtime = state
        self.model = model
        self.show_logo = bool(logo)
        self.session_workspace = (
            Path(session_workspace) if session_workspace is not None else Path(state.workspace)
        )
        self.max_attempts = int(max_attempts)
        self.max_loops = max_loops
        self.approval_timeout = float(approval_timeout)
        #: Filled from the first turn's ``final_answer``, or read from disk at
        #: mount when an earlier session is already on file.
        self.session_id = ""
        self.session_turns = 0
        self._turn: _TurnRunner | None = None
        self._node = ""
        self._route = ""
        self._saw_error = False
        self._pending_gates: set[ApprovalGate] = set()
        self._gate_lock = threading.Lock()

    # -- layout ----------------------------------------------------------- #
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        if self.show_logo:
            # Above the status line and outside both panels, so the animation
            # repaints a row of its own instead of fighting the stream for it.
            yield SplashLogo(id="splash")
        yield Static(id="status")
        yield Static(
            Text("waiting for a plan — the planner publishes one when work is needed", style="dim"),
            id="plan",
        )
        yield RichLog(id="stream", max_lines=MAX_LOG_LINES, wrap=True, highlight=False)
        yield Input(placeholder="Ask for work, or just say hi — Enter submits", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.session_id = _session_id_of(self.session_workspace)
        if self.runtime.approval_mode == "inline":
            # The one place the app writes to the state, and the reason is in the
            # class docstring: in inline mode the TUI is the thing that answers,
            # so whatever handler the caller installed (a rich prompt on stdin)
            # has to give way to the modal.
            self.runtime.approval_handler = self._ask_approval
        self.sub_title = f"session: {_short(self.session_id)}"
        self._write_welcome()
        self._refresh_status()
        self.query_one("#prompt", Input).focus()

    def on_unmount(self) -> None:
        # A second net under ``action_quit``: whatever ends the app, no tool
        # thread is left waiting on a modal that can no longer open.
        self._release_gates(REASON_APP_CLOSED)

    # -- writing to the stream -------------------------------------------- #
    def _log(self) -> RichLog:
        return self.query_one("#stream", RichLog)

    def _write(self, renderable: Any) -> None:
        self._log().write(renderable, expand=True)

    def _gap(self) -> None:
        """A blank line, unless the stream is still empty."""
        if self._log().lines:
            self._log().write("")

    def _write_welcome(self) -> None:
        """The one-time setup block: everything a reader needs to trust the run."""
        model_name = str(getattr(self.model, "model_name", None) or "unknown")
        modes = (
            f"approval {self.runtime.approval_mode} · checkpoint "
            f"{self.runtime.checkpoint_mode} · trace {self.runtime.trace_mode}"
        )
        rows = [
            ("workspace", str(self.runtime.workspace)),
            ("session", str(self.session_workspace)),
            ("model", model_name),
            ("modes", modes),
            ("hint", "type a message and press Enter · Ctrl+L clears · Ctrl+C quits"),
        ]
        line = Text()
        for label, value in rows:
            line.append(f"{label:<10}", style="dim")
            line.append(f"{value}\n", style="dim" if label == "hint" else "bold")
        self._write(line)

    def _drop_splash(self) -> None:
        """Retire the logo, once there is work to show in its place.

        Four rows of decoration are the right trade when the screen is empty and
        the wrong one as soon as a run starts filling the stream, so this runs at
        the *first* turn and never again. ``query`` rather than a saved
        reference, because the widget is created by :meth:`compose` and may not
        exist at all (``logo=False``).
        """
        for splash in self.query(SplashLogo):
            splash.remove()

    def _section(self, node: str) -> None:
        """Announce a node's section: an icon and a colour, no rule."""
        label = NODE_LABELS.get(node, node or "node")
        style = NODE_STYLES.get(node, "white")
        self._gap()
        self._write(Text(label, style=f"bold {style}"))

    # -- turn lifecycle ---------------------------------------------------- #
    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        task = event.value.strip()
        if not task:
            return
        if self._turn is not None:
            # One turn at a time: the session turn counter and the workspace both
            # assume turns are sequential, which is what a queue would break. The
            # input is disabled while a turn runs, so this is a belt-and-braces
            # guard for a keystroke that slipped through.
            self._write(Text("a turn is already running — wait for it to finish", style="yellow"))
            return
        prompt = self.query_one("#prompt", Input)
        prompt.value = ""
        prompt.disabled = True
        self._begin_turn(task)

    def _begin_turn(self, task: str) -> None:
        self._saw_error = False
        self._node = ""
        self._route = ""
        self._drop_splash()
        self._gap()
        self._write(Text.assemble(("💬 you  ", "bold cyan"), (task, "")))
        self._turn = _TurnRunner(
            lambda: stream_session_events(
                task,
                session_workspace=self.session_workspace,
                state=self.runtime,
                model=self.model,
                max_attempts=self.max_attempts,
                max_loops=self.max_loops,
            ),
            lambda event: self.post_message(AgentEventMessage(event)),
        )
        self._turn.start()
        self._refresh_status()

    def _end_turn(self) -> None:
        self._turn = None
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self._refresh_status()

    def action_clear_log(self) -> None:
        self._log().clear()

    def action_quit(self) -> None:
        """Leave the app, refusing anything still waiting for a verdict.

        The gates go first, and the order is the point: a tool thread parked in
        :meth:`~suuclaw.cli.tui.approval.ApprovalGate.wait` is waiting on a
        modal that can never open once the app is gone, and closing the app must
        not be read as an approval.
        """
        self._release_gates(REASON_APP_CLOSED)
        if self._turn is not None:
            self._turn.stop()
        self.exit()

    # -- approval bridge --------------------------------------------------- #
    def _release_gates(self, reason: str) -> None:
        with self._gate_lock:
            pending = list(self._pending_gates)
            self._pending_gates.clear()
        deny_pending(pending, reason)

    def _ask_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """The state's approval handler. Runs on the *tool* thread, and blocks.

        Posts the gate, then waits for the modal's verdict — see the module
        docstring of :mod:`suuclaw.cli.tui.approval` for why the answer cannot
        be computed on this thread. A post that is refused (the app is already
        closing) short-circuits to a refusal instead of waiting out the timeout.
        """
        gate = ApprovalGate(timeout=self.approval_timeout)
        with self._gate_lock:
            self._pending_gates.add(gate)
        try:
            if not self.post_message(ApprovalRequestedMessage(request, gate)):
                return ApprovalDecision(
                    approved=False,
                    reason="the app is not accepting requests; the command was refused",
                )
            return gate.wait()
        finally:
            with self._gate_lock:
                self._pending_gates.discard(gate)

    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        """Show the modal, and let its verdict release the tool thread."""
        request = message.request
        gate = message.gate
        self._gap()
        self._write(
            Text.assemble(
                ("⚠ approval requested ", "bold yellow"),
                (f"{request.tool_name} · {request.risk_reason}", "yellow"),
            )
        )
        self.push_screen(
            ApprovalModal(request, workspace=str(self.runtime.workspace)),
            lambda approved: gate.resolve(
                bool(approved),
                "approved in the TUI" if approved else "declined in the TUI",
            ),
        )

    # -- event dispatch ---------------------------------------------------- #
    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        """Render one workflow event. The only code that touches widgets."""
        event = message.event
        kind = str(event.get("type") or "")

        if kind == "node_start":
            node = str(event.get("node") or "")
            self._node = node
            # ``final`` gets no heading: the answer panel titles itself, and a
            # "📝 Final" line directly above it is the same word twice.
            if node != "final":
                self._section(node)
            self._refresh_status()

        elif kind == "node_output":
            self._render_node_output(event)

        elif kind == "node_summary":
            self._render_node_summary(event)

        elif kind == "ai_message":
            self._render_ai_message(event)

        elif kind == "tool_call":
            name = str(event.get("name") or "tool")
            line = Text("  ▸ ", style="cyan")
            line.append(name, style="bold cyan")
            line.append(f"  {_one_line_args(event.get('args'))}", style="dim")
            self._write(line)

        elif kind == "tool_result":
            self._render_tool_result(event)

        elif kind == "handoff":
            to_agent = str(event.get("to") or "specialist")
            instruction = " ".join(str(event.get("instruction") or "").split())
            if len(instruction) > 120:
                instruction = instruction[:119] + "…"
            line = Text("  ⇢ ", style="yellow")
            line.append(f"handoff → {to_agent}", style="bold yellow")
            if instruction:
                line.append(f"  {instruction}", style="dim")
            self._write(line)

        elif kind == "search_results":
            ok = bool(event.get("ok"))
            query = " ".join(str(event.get("query") or "").split()) or "(no query)"
            if ok:
                tail = f"{int(event.get('result_count') or 0)} source(s)"
            else:
                tail = str(event.get("error") or "failed")
            line = Text("  ⤷ ", style="cyan")
            line.append(query, style="cyan")
            line.append(f"  → {tail}", style="dim" if ok else "red")
            self._write(line)

        elif kind == "memory":
            self._write(
                Text(
                    f"  🧠 layered memory ready · {event.get('node') or '?'}",
                    style="dim",
                )
            )

        elif kind == "checkpoint_saved":
            note = f"  💾 checkpoint {event.get('status') or 'saved'}"
            where = str(event.get("path") or "")
            if where:
                note += f" · {where}"
            self._write(Text(note, style="dim"))

        elif kind == "heartbeat":
            quiet = float(event.get("quiet_for") or 0.0)
            note = f"  ⏳ no events for {quiet:.0f}s"
            node = str(event.get("node") or "")
            if node:
                note += f" · {node} still working"
            self._write(Text(note, style="dim"))

        elif kind == "retry":
            self._write(
                Text(
                    f"  ↻ transient failure ({event.get('message')}) — retrying "
                    f"{event.get('attempt')}/{event.get('max_retries')} in {event.get('delay')}s",
                    style="yellow",
                )
            )

        elif kind == "error":
            self._saw_error = True
            self._write(Text(f"  ✖ {event.get('message')}", style="bold red"))

        elif kind == "final_answer":
            self._render_final(event)

        else:
            # Unknown types are shown rather than dropped: a driver that grows a
            # new event should be visible in the stream, not silently ignored.
            shown = str(event)[:MAX_UNKNOWN_EVENT_CHARS]
            self._write(Text(f"  · {kind or 'event'}  {shown}", style="dim"))

    # -- per-node rendering ----------------------------------------------- #
    def _render_node_output(self, event: Mapping[str, Any]) -> None:
        data = event.get("data")
        if not isinstance(data, Mapping):
            return
        node = str(event.get("node") or "")

        if node == "intent_router":
            route = str(data.get("intent_route") or "workflow")
            confidence = data.get("intent_confidence")
            reason = " ".join(str(data.get("intent_reason") or "").split())
            self._route = route
            line = Text("  🎯 route → ", style="dim")
            line.append(route, style=f"bold {'green' if route == 'chat' else 'blue'}")
            if isinstance(confidence, (int, float)):
                line.append(f"  confidence {float(confidence):.2f}", style="dim")
            if reason:
                line.append(f"  {reason}", style="dim")
            self._write(line)
            self._refresh_status()

        elif node == "planner":
            self._render_plan(data)

        elif node == "verifier":
            passed = bool(data.get("passed"))
            reason = str(data.get("last_error") or "").strip()
            if not reason:
                reason = "verification passed" if passed else "the verifier recorded no reason"
            tone = "green" if passed else "red"
            line = Text("  ")
            line.append("✅ passed" if passed else "❌ failed", style=f"bold {tone}")
            line.append(f"  {reason}", style=tone)
            self._write(line)
            for row in data.get("verification_checks") or []:
                if not isinstance(row, Mapping):
                    continue
                check = Text(f"     {'✅' if row.get('passed') else '❌'}  {row.get('name', '')}")
                detail = str(row.get("detail") or "").strip()
                if detail:
                    check.append(f"  {detail[:160]}", style="dim")
                self._write(check)

        elif node == "context_monitor":
            counted = data.get("context_token_count")
            limit = data.get("context_token_limit")
            if counted is None or limit is None:
                return
            counted, limit = int(counted), int(limit)
            share = f"{counted / limit:.0%}" if limit > 0 else "n/a"
            over = bool(data.get("context_should_compress"))
            note = f"  🧠 {counted:,} / {limit:,} tokens · {share} — "
            note += "over budget, compressing next" if over else "under budget"
            self._write(Text(note, style="yellow" if over else "dim"))

        # ``chat_responder`` and ``final`` are deliberately absent: their payload
        # *is* the answer, and the answer panel below prints it once.

    def _render_plan(self, data: Mapping[str, Any]) -> None:
        """Update the Plan panel: the summary, then one chip per step."""
        rows = [row for row in (data.get("todos") or []) if isinstance(row, Mapping)]
        if not rows:
            return
        done = sum(1 for row in rows if str(row.get("status") or "") == "completed")
        chips = Text()
        for row in rows:
            status = str(row.get("status") or "pending")
            chips.append(f"{row.get('id') or '?'} ", style="bold")
            chips.append(PLAN_MARKS.get(status, "⬜"))
            chips.append("  ")
        summary = " ".join(str(data.get("plan_summary") or "").split())
        body: Any = Group(Text(summary, style="dim"), chips) if summary else chips
        self.query_one("#plan", Static).update(
            Panel(
                body,
                title=f"Plan · {len(rows)} step(s) · {done} done",
                title_align="left",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def _render_ai_message(self, event: Mapping[str, Any]) -> None:
        node = str(event.get("node") or "")
        loop = int(event.get("loop") or 0)
        planned = [str(name) for name in (event.get("tool_calls") or [])]
        head = Text("◆ ", style="magenta")
        if node:
            head.append(f"{node} ", style=NODE_STYLES.get(node, "white"))
        head.append(f"loop {loop}", style="dim")
        if planned:
            head.append("  plans ", style="dim")
            head.append(", ".join(planned), style="cyan")
        self._write(head)
        content = str(event.get("content") or "").strip()
        # The verifier's closing turn *is* its JSON report, and the verdict block
        # already renders that as findings — echoing the raw object here buries it.
        if content and not (node == "verifier" and not planned):
            self._write(Markdown(content))

    def _render_tool_result(self, event: Mapping[str, Any]) -> None:
        ok = bool(event.get("ok"))
        raw = event.get("result")
        result = raw if isinstance(raw, Mapping) else {}
        fallback = "" if ok else "unknown error"
        # Two shapes reach here: the inner ReAct loop nests the text under
        # ``result``, the delegation tools' specialists emit it flat.
        if "output" in event or "error" in event:
            body = str(event.get("output" if ok else "error", fallback))
        else:
            body = str(result.get("output" if ok else "error", fallback))
        note, body = _split_approval_note(body)
        if note:
            self._write(Text(f"  ⚠ {note}", style="yellow"))
        clipped, hidden = _preview(body)
        if clipped.strip():
            self._write(Text(clipped, style="dim" if ok else "red"))
        if hidden > 0:
            self._write(Text(f"     … {hidden} more line(s)", style="dim italic"))

    def _render_node_summary(self, event: Mapping[str, Any]) -> None:
        """Only the news a loop could not say itself: it was cut short, or ran out."""
        node = str(event.get("node") or "")
        label = NODE_LABELS.get(node, node or "node")
        if event.get("interrupted"):
            detail = str(event.get("error") or "unknown error")
            self._write(Text(f"  {label} was cut short: {detail}", style="red"))
        elif event.get("exhausted"):
            self._write(
                Text(
                    f"  {label} used all {int(event.get('loops') or 0)} of its loops without "
                    "a closing turn; its final tool result was never read.",
                    style="yellow",
                )
            )

    def _render_final(self, event: Mapping[str, Any]) -> None:
        content = str(event.get("content") or "").strip()
        passed = bool(event.get("passed"))
        route = str(event.get("route") or "workflow")
        turn = event.get("session_turn")
        session_id = str(event.get("session_id") or "")
        if session_id:
            self.session_id = session_id
            self.sub_title = f"session: {_short(session_id)}"
        if isinstance(turn, int):
            self.session_turns = max(self.session_turns, turn)
        if bool(event.get("interrupted")) or self._saw_error:
            title, border = f"⚠ interrupted · {route} · turn {turn}", "red"
        elif passed:
            title, border = f"✅ final answer · {route} · turn {turn}", "green"
        else:
            title, border = f"❌ not verified · {route} · turn {turn}", "yellow"
        if content:
            body: Any = Markdown(content)
        else:
            body = Text("the workflow produced no answer", style="yellow")
        self._gap()
        self._write(
            Panel(body, title=title, title_align="left", border_style=border, padding=(1, 2))
        )
        self._end_turn()

    # -- status ------------------------------------------------------------ #
    def _refresh_status(self) -> None:
        busy = self._turn is not None
        bits = ["● working" if busy else "○ idle"]
        if self._route:
            bits.append(f"route {self._route}")
        if busy and self._node:
            bits.append(self._node)
        bits.append(str(self.runtime.workspace))
        bits.append(str(getattr(self.model, "model_name", None) or "unknown"))
        bits.append(f"approval {self.runtime.approval_mode}")
        if self.session_turns:
            bits.append(f"turn {self.session_turns}")
        self.query_one("#status", Static).update(Text(" · ".join(bits)))


# Existing Python integrations can keep importing the original class name.


def run_tui(
    *,
    state: RuntimeState,
    model: Any | None = None,
    session_workspace: str | os.PathLike[str] | None = None,
    temperature: float = 0.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_loops: int | None = None,
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    logo: bool = True,
) -> int:
    """Open the TUI and block until it is closed. Returns a process exit code.

    The model is built here when the caller has none, so ``suuclaw --tui``
    reaches this function with the same inputs a one-shot run would get, and a
    missing key is treated the same way: a message and exit code ``2``, rather
    than a TUI that can do nothing.

    A non-interactive stream is refused up front. A Textual app drives a real
    terminal — stdin, raw mode, resize events — and the honest answer to
    ``suuclaw --tui | tee`` is that there is no terminal to drive.
    """
    if not sys.stdout.isatty():
        print("error: the TUI needs an interactive terminal", file=sys.stderr)
        return 2
    if model is None:
        from ...providers.openai_provider import create_model

        try:
            model = create_model(temperature=temperature)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    app = SuuClawTuiApp(
        state=state,
        model=model,
        session_workspace=session_workspace,
        max_attempts=max_attempts,
        max_loops=max_loops,
        approval_timeout=approval_timeout,
        logo=logo,
    )
    app.run()
    if app.session_id:
        # Printed on the way out so a later ``--resume``-style entry point, or a
        # second TUI pointed at the same workspace, can name the conversation.
        print(f"session: {app.session_id}", file=sys.stderr)
    return 0
