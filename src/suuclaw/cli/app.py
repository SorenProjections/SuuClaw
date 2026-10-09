"""Typer entry point.

Usage::

    suuclaw "summarise every python file in ./src" --workspace ./scratch
    suuclaw --list-tools          # inspect the registry without an API key
    python -m suuclaw "..." -w ./scratch

The CLI owns three things and nothing else:

1. it builds the :class:`RuntimeState` and turns the registry into tools,
2. it builds the chat model that the workflow's nodes share,
3. it renders the :func:`stream_agent_events` event stream live with rich,
   labelling each node's output — 📋 planner, 🔎 search, 🛠 code, ✅/❌ verifier,
   📝 final — as it arrives.

Neither the graph nor the per-node ReAct loops live here: ``suuclaw.core.agent``
drives the workflow and ``suuclaw.graph`` decides what runs next. The CLI never
picks the next node itself.

``--tui`` hands the same state and model to a second front end
(:mod:`suuclaw.cli.tui`), which keeps a live widget tree and runs a whole
conversation of session turns. It is imported inside the command body because
``textual`` is an optional dependency; the one-shot path above is untouched by
it, and that is the point of keeping the two apart.

Typography
----------
A transcript of a long run is read, not skimmed, so the rendering follows three
rules that the layout helpers below enforce rather than leave to each call site:

* **One gutter.** Text is indented by wrapping it in a measured block
  (:func:`_measured`), never by prefixing spaces to a string. rich soft-wraps at
  the console width, and a hand-prefixed indent applies only to the first line —
  every wrapped continuation falls back to column 0, which is what made a
  paragraph look like it had escaped its own section.
* **One measure.** Narrative is capped at :data:`MEASURE` columns even on a wide
  terminal. Machine output is not: a tool's own columns are data, and folding them
  to 88 characters destroys the alignment that makes them readable.
* **One blank line between blocks, none inside one.** A loop heading, its tool
  call and that tool's output are one block; 25 loops are 25 blocks. Without the
  distinction the run reads as an undifferentiated wall.

Sections are announced with a coloured label, not a rule: ``console.rule`` draws
across the whole terminal, and 110 columns of dashes per node competes with the
text it is meant to delimit.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..core import RuntimeState, console_approval_handler, stream_agent_events
from ..core.agent import DEFAULT_RETRIES
from ..graph.nodes import DEFAULT_MAX_ATTEMPTS
from ..tools.registry import build_tools, describe_tools

__all__ = ["app", "render_agent_events", "run_agent"]

#: Generous by design: "write code → run it → fix what broke → re-run" legitimately
#: spends several loops, and being cut off mid-repair reads as a failure to the user.
#: In the workflow this bounds the code agent's loops *per delegation*, not per run.
DEFAULT_MAX_LOOPS = 25
PREVIEW_LINES = 12
MAX_ARG_CHARS = 48
MAX_ARG_LINE_CHARS = 160
#: Verifier findings are evidence, not summaries; this is long enough for the
#: substantive ones and short enough that six of them stay scannable.
MAX_DETAIL_CHARS = 200

#: Column at which *narrative* folds. A 110-column paragraph is harder to read
#: than one wrapped at ~88 — the eye loses its place on the return sweep — and
#: capping it also aligns every section's body under a single right edge. Tool
#: output deliberately ignores this (see :func:`_measured`).
MEASURE = 88

#: Indentation steps, in spaces. Three levels: a section's body, the detail under
#: one of its items, and a tool's output under the call that produced it.
INDENT_BODY = 2
INDENT_DETAIL = 4
INDENT_OUTPUT = 6

#: One label per graph node. The icon is the whole point: a run's output is four
#: different kinds of thing (a plan, work, a verdict, a report) and a reader should
#: be able to tell which is which without reading the text.
NODE_LABELS: dict[str, str] = {
    "planner": "📋 Planner",
    "search": "🔎 Search",
    "code": "🛠 Code",
    "verifier": "🔍 Verifier",
    "final": "📝 Final",
    # Context housekeeping, not work on the task: the monitor decides whether the
    # run is over budget and the compressor pays for it when it is.
    "context_monitor": "🧠 Context",
    "context_compressor": "🗜 Compressed",
}
NODE_STYLES: dict[str, str] = {
    "planner": "blue",
    "search": "cyan",
    "code": "cyan",
    "verifier": "magenta",
    "final": "green",
    "context_monitor": "yellow",
    "context_compressor": "yellow",
}
#: The section header already names the node, so the verdict line does not repeat
#: it — "✅ Verifier" under a "🔍 Verifier" heading was saying it twice.
VERIFIER_PASSED = "✅ passed"
VERIFIER_FAILED = "❌ failed"
STATUS_MARKS: dict[str, str] = {
    "completed": "✔",
    "in_progress": "▶",
    "pending": "·",
    "blocked": "!",
}
STATUS_STYLES: dict[str, str] = {
    "completed": "green",
    "in_progress": "cyan",
    "pending": "dim",
    "blocked": "red",
}

#: Shown under the options block of ``--help``.
#:
#: Where it is attached is load-bearing. With one ``@app.command()`` and no
#: ``callback``, Typer compiles the app into a bare ``TyperCommand`` instead of a
#: group, and that branch copies *only* ``context_settings`` off ``Typer(...)``.
#: An app-level ``epilog=`` (and ``help=``) never reaches click, so it never
#: reaches the terminal — which is exactly where this text used to be written.
#: Hanging it on the command is what makes it survive into ``--help``.
HELP_EPILOG = """
Examples:
  suuclaw dashboard -w ./workspace # browser workspace at http://127.0.0.1:7788
  suuclaw "create hello.py that prints hi, then run it" -w ./sandbox
  suuclaw "find every TODO comment and list it" -w .
  suuclaw --tui -w ./sandbox     # interactive session: many turns, one workspace
  suuclaw --resume ./sandbox   # continue an interrupted run from its checkpoint
  suuclaw --list-tools          # inspect the registry, no API key needed

Every tool call is confined to the workspace; path traversal outside it is rejected.
"""

app = typer.Typer(
    add_completion=False,
    # "rich" is load-bearing twice over: rich prints the help at all, and it is the
    # only one of the two help renderers that keeps the Examples block intact.
    # click's plain formatter runs the epilog through its paragraph re-wrapper,
    # which merges the example lines into one blob (verified both ways).
    rich_markup_mode="rich",
)


# --------------------------------------------------------------------------- #
# console helpers
# --------------------------------------------------------------------------- #
def _harden_console() -> None:
    """Best-effort UTF-8 stdout/stderr so box glyphs survive Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - already-detached stream
            pass


# Called at import, not only inside the command body, and the ordering is the
# whole reason it exists here: click renders ``--help`` *before* the callback
# runs, and with ``rich_markup_mode="rich"`` that rendering goes through
# sys.stdout directly. On a stream that cannot represent the panel glyphs or the
# arrows in this file's help text, rich refuses to write and `--help` dies — a
# plain click formatter would have replaced the character and carried on.
# Reconfiguring an already-UTF-8 stream is a no-op, so this costs nothing.
_harden_console()


def _make_console() -> Console:
    return Console(highlight=False)


@dataclass
class _Layout:
    """Blank-line discipline for the live transcript.

    Consecutive parts of one block — a loop heading, its tool call, that tool's
    output — must not be separated, or 25 loops turn into 150 loose lines.
    Distinct blocks must be, or the eye has nothing to group them by. One field
    records what was written last, so each renderer can ask for the right spacing
    instead of every call site guessing at it.
    """

    last: str = ""

    def gap(self, console: Console) -> None:
        """A blank line before a new block — except straight after a heading.

        A section header is itself a separator, so the body that follows it is
        tight against it; anything else starts a new visual block.
        """
        if self.last and self.last != "header":
            console.print()


def _trim_trailing(segments: list[Any]) -> list[Any]:
    """Drop the end-of-line blanks that column fill and word-wrap leave behind.

    Two invisible sources: a ``Table`` fills every cell to its column width, and
    rich's wrapper keeps the space a line happened to break on. Trailing blanks
    are never load-bearing — alignment lives *inside* a line — so they can go.
    Without this, a transcript padded to the measure carries trailing whitespace
    on most of its lines, which is only invisible until it is copied or diffed.
    """
    out = list(segments)
    while out:
        text = getattr(out[-1], "text", "")
        if not text:
            out.pop()
            continue
        stripped = text.rstrip()
        if stripped:
            if stripped != text:
                last = out[-1]
                out[-1] = Segment(stripped, last.style, last.control)
            break
        out.pop()
    return out


class _Measured:
    """Indent every line of a renderable, cap the fold width, pad nothing.

    This replaces an earlier ``Padding(Table.grid(width=N))``. A fixed-width
    table column fills each cell out to ``N``, which is what put trailing blanks
    on the transcript. ``render_lines(..., pad=False)`` folds to the same width
    without filling, and the fold is still rich's own, so wide glyphs are
    measured in terminal cells rather than in characters — a hand-rolled
    ``textwrap`` would have made every Chinese paragraph twice as wide as asked.
    """

    def __init__(self, renderable: Any, *, indent: int, measure: int | None) -> None:
        self.renderable = renderable
        self.indent = indent
        self.measure = measure

    def __rich_console__(self, console: Console, options: Any) -> Iterable[Any]:
        width = options.max_width if self.measure is None else min(self.measure, options.max_width)
        inner = max(24, width - self.indent)
        lines = console.render_lines(self.renderable, options.update_width(inner), pad=False)
        gutter = Segment(" " * self.indent)
        for line in lines:
            trimmed = _trim_trailing(line)
            if trimmed:
                yield gutter
                yield from trimmed
            # A blank line carries no gutter: an indent on a line with nothing on
            # it is whitespace-only output, the same defect one level down.
            yield Segment("\n")


def _measured(
    renderable: Any,
    console: Console,
    *,
    indent: int,
    measure: int | None = MEASURE,
) -> _Measured:
    """Indent *every* line of ``renderable``, including ones rich inserts.

    ``measure`` caps the fold width; ``None`` means "as wide as the terminal
    allows". The cap is applied to the text, not to the terminal, which is why a
    paragraph folds where it should rather than at whatever the window happens to
    be — and why machine output (:data:`MEASURE`-free) keeps its own columns.
    """
    return _Measured(renderable, indent=indent, measure=measure)


def _block(
    console: Console,
    renderable: Any,
    *,
    indent: int = INDENT_BODY,
    measure: int | None = MEASURE,
) -> None:
    """Print one measured block.

    There is deliberately no ``soft_wrap=True`` anywhere in this module. rich reads
    that flag as "do not wrap at all", not "wrap softly": a line runs past the
    terminal edge instead of folding. That is how a 155-character verdict line
    ended up on a single line on a 110-column terminal — the text was there, but
    off the right edge of the block it belonged to.
    """
    console.print(
        _measured(renderable, console, indent=indent, measure=measure),
        highlight=False,
    )


def _prose(
    console: Console,
    text: str,
    *,
    indent: int = INDENT_BODY,
    style: str = "",
    measure: int | None = MEASURE,
) -> None:
    """Print one paragraph of narrative, wrapped to the measure and indented."""
    _block(console, Text(text, style=style), indent=indent, measure=measure)


def _preview(text: str, *, lines: int | None = PREVIEW_LINES) -> tuple[str, int]:
    """Split a tool's output into ``(shown, hidden_line_count)``.

    The "… N more lines" marker used to be appended to the body itself, which put
    machine output and the renderer's own footnote in the same visual block. The
    count comes back separately so the caller can style it as the note it is.
    """
    if lines is None:
        return text, 0
    split = text.splitlines() or [""]
    if len(split) <= lines:
        return text, 0
    return "\n".join(split[:lines]), len(split) - lines


def _hidden_note(console: Console, hidden: int, *, indent: int) -> None:
    """Say that output was cut, in the renderer's voice rather than the tool's."""
    if hidden <= 0:
        return
    _prose(
        console,
        f"… {hidden} more line(s) — rerun with --show-trace to see them",
        indent=indent,
        style="dim italic",
    )


def _flatten(value: Any) -> str:
    """One-line form of an argument value, newlines made visible."""
    return str(value).replace("\n", "\\n").replace("\t", "\\t")


def _format_args(args: Mapping[str, Any] | None, *, budget: int = MAX_ARG_LINE_CHARS) -> str:
    """Width-budgeted one-line rendering of a tool call's arguments.

    A single argument drops its key: ``bash(command="ls -a")`` is a shell command,
    and ``command=`` in front of it is scaffolding the reader has to skip past.
    Two or more keep their keys, because then the pairing is the information.
    """
    items = list((args or {}).items())
    if len(items) == 1:
        joined = _flatten(items[0][1])
    else:
        parts: list[str] = []
        for key, value in items:
            text = _flatten(value)
            if len(text) > MAX_ARG_CHARS:
                text = f"{text[:MAX_ARG_CHARS]}…"
            parts.append(f"{key}={text}")
        joined = ", ".join(parts)
    return joined if len(joined) <= budget else f"{joined[: max(budget - 1, 1)]}…"


def _short_reason(message: str, limit: int = 56) -> str:
    """One-line error summary for a panel title."""
    text = (message or "unknown error").replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _shell_label(state: RuntimeState) -> str:
    """Human-readable shell, so the WSL-vs-Git-Bash world is visible up front."""
    # lazy: importing bash_tool at module scope re-creates the registry cycle
    from ..tools.bash_tool import resolve_shell

    try:
        return " ".join(resolve_shell(state.shell))
    except ValueError:  # pragma: no cover - misconfigured --shell
        return str(state.shell)


def _print_header(
    console: Console,
    *,
    state: RuntimeState,
    model_name: str,
    tools: Iterable[Any],
    max_loops: int,
    max_attempts: int,
    approval_mode: str = "inline",
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
    resume_workspace: Path | None = None,
) -> None:
    rows: list[tuple[str, str]] = [
        ("workspace", str(state.workspace)),
        ("model", model_name),
        ("shell", _shell_label(state)),
        ("tools", ", ".join(t.name for t in tools)),
        ("code loops", str(max_loops)),
        ("max attempts", str(max_attempts)),
        # One line, not three: policies are metadata, not a section each.
        (
            "modes",
            f"approval {approval_mode} · checkpoint {checkpoint_mode} · trace {trace_mode}",
        ),
    ]
    if resume_workspace is not None:
        rows.append(("resume", str(resume_workspace)))
    console.print()
    console.print(
        Text.assemble(
            ("/// ", "bold #53e1c6"),
            ("SuuClaw", "bold #68a9ef"),
            ("  ·  Local agents, work in view", "dim"),
        )
    )
    for label, value in rows:
        # No soft_wrap: a long tools list or shell path should fold onto the next
        # line rather than run past the terminal edge.
        console.print(Text.assemble((f"{label:<13}", "dim"), (value, "bold")))


# --------------------------------------------------------------------------- #
# event rendering
# --------------------------------------------------------------------------- #
def _print_node_header(console: Console, node: str) -> None:
    """Announce that ``node`` has started: one coloured line, no rule.

    ``console.rule`` spans the terminal, so each node cost a full line of dashes —
    the loudest thing on screen, competing with the text it was meant to delimit.
    An icon, a colour and the blank line above already separate the sections.
    """
    label = NODE_LABELS.get(node, node or "node")
    style = NODE_STYLES.get(node, "white")
    console.print()
    console.print(Text(label, style=f"bold {style}"), highlight=False)


def _bullet_block(
    console: Console,
    title: str,
    items: Any,
    *,
    indent: int = INDENT_BODY,
) -> None:
    """A titled bullet list, skipped entirely when there is nothing to list."""
    rows = [str(item).strip() for item in (items or []) if isinstance(item, (str, int, float))]
    rows = [row for row in rows if row]
    if not rows:
        return
    _prose(console, title, indent=indent, style="bold")
    for row in rows:
        _prose(console, f"· {row}", indent=indent + 2, style="dim")


def _step_table(console: Console, rows: Sequence[Mapping[str, Any]], *, indent: int) -> None:
    """The plan as a hanging-indent table: status and id left, brief hanging off it.

    The old form interpolated all three into one string — ``· inspect. Inspect the
    workspace root …`` — so a step id, a status and a two-sentence brief became a
    single 110-column run whose wrapped lines fell back to column 0. Two columns
    fix both: the id turns into a scannable gutter, and rich indents the wrap under
    the text it belongs to.
    """
    ids = [str(row.get("id") or "?") for row in rows]
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=1)  # status mark
    grid.add_column(width=max(len(step_id) for step_id in ids))  # step id
    grid.add_column()  # brief
    for row, step_id in zip(rows, ids, strict=True):
        status = str(row.get("status") or "pending")
        grid.add_row(
            Text(STATUS_MARKS.get(status, "-"), style=STATUS_STYLES.get(status, "")),
            Text(step_id, style="bold"),
            Text(str(row.get("content") or "").strip(), style="dim"),
        )
    console.print(_measured(grid, console, indent=indent), highlight=False)


def _render_planner_output(
    console: Console,
    update: Mapping[str, Any],
    *,
    show_trace: bool,
    layout: _Layout,
) -> bool:
    """📋 what the run intends to do: the summary, the steps, and the yardstick."""
    layout.gap(console)
    summary = str(update.get("plan_summary") or "").strip()
    if summary:
        _prose(console, summary)

    rows = [row for row in (update.get("todos") or []) if isinstance(row, Mapping)]
    if rows:
        _prose(console, f"{len(rows)} step(s)", style="bold")
        _step_table(console, rows, indent=INDENT_DETAIL)

    sources = [row for row in (update.get("sources") or []) if isinstance(row, Mapping)]
    if sources:
        _prose(console, f"{len(sources)} web source(s) on file", style="dim")

    handoffs = [row for row in (update.get("agent_handoffs") or []) if isinstance(row, Mapping)]
    if handoffs:
        last = handoffs[-1]
        _prose(
            console,
            f"delegations so far: {len(handoffs)} (last → {last.get('to_agent', '?')})",
            style="dim",
        )

    _bullet_block(console, "done when", update.get("acceptance_criteria"))
    _bullet_block(console, "commands", update.get("verification_commands"))

    note = str(update.get("last_error") or "").strip()
    if note:
        _prose(console, f"revising after: {note}", style="yellow")

    layout.last = "body"
    return True


def _render_verifier_output(
    console: Console,
    update: Mapping[str, Any],
    *,
    show_trace: bool,
    layout: _Layout,
) -> bool:
    """✅/❌ the verdict, the per-criterion findings, and what the commands really did."""
    layout.gap(console)
    passed = bool(update.get("passed"))
    reason = str(update.get("last_error") or "").strip()
    if not reason:
        reason = "verification passed" if passed else "the verifier recorded no reason"
    verdict = VERIFIER_PASSED if passed else VERIFIER_FAILED
    tone = "green" if passed else "red"
    _block(console, Text.assemble((verdict, f"bold {tone}"), ("  ", ""), (reason, tone)))

    for row in update.get("verification_checks") or []:
        if not isinstance(row, Mapping):
            continue
        mark = "✅" if row.get("passed") else "❌"
        _prose(console, f"{mark}  {row.get('name', '')}", indent=INDENT_DETAIL)
        detail = str(row.get("detail") or "").strip()
        if detail:
            _prose(
                console,
                _short_reason(detail, MAX_DETAIL_CHARS),
                indent=INDENT_DETAIL + 3,
                style="dim",
            )

    for row in update.get("verification_results") or []:
        if not isinstance(row, Mapping):
            continue
        ok = bool(row.get("ok"))
        exit_code = row.get("exit_code")
        if exit_code is None:
            shown = "no exit code (timeout or spawn failure)"
        else:
            shown = f"exit {exit_code}"
        head = f"{'ran ok' if ok else 'FAILED'}: {row.get('command', '')} → {shown}"
        _prose(console, head, indent=INDENT_DETAIL, style="dim" if ok else "red")
        if ok:
            continue
        for key in ("stdout", "stderr"):
            body = str(row.get(key) or "").strip()
            if not body:
                continue
            clipped, hidden = _preview(body, lines=None if show_trace else 6)
            _prose(console, clipped, indent=INDENT_OUTPUT, style="red", measure=None)
            _hidden_note(console, hidden, indent=INDENT_OUTPUT)

    layout.last = "body"
    return True


def _render_monitor_output(
    console: Console,
    update: Mapping[str, Any],
    *,
    show_trace: bool,
    layout: _Layout,
) -> bool:
    """🧠 the budget line: tokens spent, against the limit they were measured by.

    The monitor's entire output is one comparison, and a comparison a reader
    cannot see is not one they can trust — least of all on a run started *with* a
    lowered budget precisely to watch the flag fire. One line: the count, the
    limit, and which side of it the run is on, so a budget that was never
    approached looks different from one that was never armed.
    """
    counted = update.get("context_token_count")
    limit = update.get("context_token_limit")
    if counted is None or limit is None:
        return False
    counted = int(counted)
    limit = int(limit)
    share = f"{counted / limit:.0%}" if limit > 0 else "n/a"
    if update.get("context_should_compress"):
        note = f"{counted:,} / {limit:,} tokens · {share} — over budget, compressing next"
        tone = "yellow"
    else:
        note = f"{counted:,} / {limit:,} tokens · {share} — under budget"
        tone = "dim"
    layout.gap(console)
    _prose(console, note, style=tone)
    layout.last = "body"
    return True


#: One renderer per node that owns a state update worth showing. ``final`` has none
#: on purpose: the terminal ``final_answer`` event already prints it as a panel, and
#: doing it twice is how a report starts contradicting itself.
_NODE_RENDERERS: dict[str, Any] = {
    "planner": _render_planner_output,
    "verifier": _render_verifier_output,
    "context_monitor": _render_monitor_output,
}


def _render_node_output(
    console: Console,
    event: Mapping[str, Any],
    *,
    show_trace: bool,
    layout: _Layout,
) -> bool:
    update = event.get("data")
    if not isinstance(update, Mapping):
        return False
    renderer = _NODE_RENDERERS.get(str(event.get("node") or ""))
    if renderer is None:
        return False
    return bool(renderer(console, update, show_trace=show_trace, layout=layout))


def _render_node_summary(
    console: Console,
    event: Mapping[str, Any],
    *,
    echoed: Mapping[str, str] | None = None,
    layout: _Layout,
) -> bool:
    """An inner ReAct loop's closing turn.

    Only the cases where the closing turn says something *new* are printed. The
    text itself has already gone past — ``content`` is by construction the last
    assistant message the loop produced, which the ``ai_message`` branch printed
    live — so echoing it here would show the same sentence twice in a row. What
    is worth printing is what the loop could not say itself: that it was cut
    short, or that it never reached a closing turn at all.
    """
    node = str(event.get("node") or "")
    label = NODE_LABELS.get(node, node or "node")
    if event.get("interrupted"):
        detail = str(event.get("error") or "unknown error")
        layout.gap(console)
        _prose(console, f"{label} was cut short: {detail}", style="red")
        layout.last = "body"
        return True
    if node == "verifier":
        # The verifier's last turn is its JSON report, already rendered as
        # findings above. Printing the raw object would bury the verdict.
        return False
    if event.get("exhausted"):
        # The loop spent its whole budget without ever closing its turn. ``content``
        # is then the last thing it said *before* its final tool call, and replaying
        # it out of context reads as a conclusion the run never reached. Report what
        # actually happened instead.
        layout.gap(console)
        _prose(
            console,
            f"{label} used all {int(event.get('loops') or 0)} of its loops without a "
            "closing turn; its final tool result was never read.",
            style="yellow",
        )
        layout.last = "body"
        return True
    content = str(event.get("content") or "").strip()
    if content and content != (echoed or {}).get(node):
        # Fallback for a closing turn whose text never went past as an
        # ``ai_message`` — a node the graph did not stream, or events a caller
        # assembled by hand. In a normal graph run this branch is not reached,
        # because ``_act``/``_judge`` record everything their loop says.
        layout.gap(console)
        _block(console, Markdown(content), indent=INDENT_BODY)
        layout.last = "body"
        return True
    return False


def render_agent_events(
    events: Iterable[dict[str, Any]],
    *,
    console: Console | None = None,
    show_trace: bool = False,
    quiet: bool = False,
) -> int:
    """Print each workflow event as it arrives and return a process exit code.

    Exit codes: ``0`` a verified answer, ``1`` a run that finished without one,
    ``2`` a run that broke. Only a verified answer is written to stdout when the
    console is not a terminal, so a script piping the output cannot mistake a
    failure report for a deliverable.
    """
    console = console or _make_console()
    errors = Console(stderr=True, highlight=False)
    interactive = console.is_terminal and not quiet
    layout = _Layout()

    final = ""
    passed = False
    turns = 0
    #: Loop numbers are per *visit*, not per run. The actor and the verifier each
    #: number their own ReAct loops from 1, and a second attempt starts at 1
    #: again; one run-wide high-water mark makes the verifier inherit the actor's
    #: count and print "loop 25" on its very first turn, which is what it did.
    loop_marks: dict[str, int] = {}
    #: Last prose printed per node, so a closing turn is never shown twice.
    echoed: dict[str, str] = {}
    #: Node whose most recent loop heading is on screen. Inside a node's section the
    #: name is already on the header line above, so repeating it on all 25 loop
    #: headings is 25 lines that say nothing new.
    announced = ""
    tool_calls = 0
    attempts = 0
    max_attempts = 0
    exhausted = False
    saw_error = False
    error_message = ""
    retries = 0

    for event in events:
        kind = event.get("type")

        if kind == "node_start":
            node = str(event.get("node") or "")
            # A revisit numbers its loops from 1 again, so the counter is reset
            # here rather than carried across attempts.
            loop_marks.pop(node, None)
            if interactive:
                announced = node
                # `final` gets no heading: the panel below titles itself, and a
                # "📝 Final" line directly above "✅ final answer" is the same
                # word twice.
                if node != "final":
                    _print_node_header(console, node)
                    layout.last = "header"

        elif kind == "node_output":
            if interactive:
                # The renderer decides whether it has anything to say, and does its
                # own spacing first: a node whose update is empty must not leave a
                # blank line behind.
                _render_node_output(console, event, show_trace=show_trace, layout=layout)

        elif kind == "node_summary":
            if interactive:
                _render_node_summary(console, event, echoed=echoed, layout=layout)

        elif kind == "ai_message":
            turns += 1
            node = str(event.get("node") or "")
            loop = max(loop_marks.get(node, 0), int(event.get("loop") or 0))
            loop_marks[node] = loop
            planned = list(event.get("tool_calls") or [])
            content = str(event.get("content") or "").strip()
            if interactive:
                layout.gap(console)
                head = Text("◆ ", style="magenta")
                if node and node != announced:
                    head.append(f"{node} ", style=NODE_STYLES.get(node, "white"))
                head.append(f"loop {loop}", style="dim")
                if planned:
                    head.append("  plans ", style="dim")
                    head.append(", ".join(planned), style="cyan")
                console.print(head, highlight=False)
                # The verifier's closing turn *is* its JSON report, and the verdict
                # block below renders that same content as findings — echoing the
                # raw object here is what buries the verdict. ``--show-trace``
                # restores it when the raw payload is what you want to read.
                if content and not (node == "verifier" and not planned and not show_trace):
                    if planned:
                        _prose(console, content)
                    else:
                        # No planned tool call means this message *closes* the
                        # loop, and a closing turn is a report: the model writes
                        # headings, bold and bullets. As plain Text they come out
                        # literally — `## Files changed`, `**life.py**` — which is
                        # noise exactly where the run is supposed to be readable.
                        _block(console, Markdown(content), indent=INDENT_BODY)
                    echoed[node] = content
                layout.last = "loop"

        elif kind == "tool_call":
            tool_calls += 1
            if interactive:
                name = str(event.get("name"))
                budget = max(24, console.width - len(name) - INDENT_DETAIL - 4)
                line = Text(" " * INDENT_DETAIL + "▸ ", style="cyan")
                line.append(name, style="bold cyan")
                line.append(f"  {_format_args(event.get('args'), budget=budget)}", style="dim")
                console.print(line, highlight=False)
                layout.last = "tool"

        elif kind == "handoff":
            if interactive:
                # The planner delegating to a specialist. Shown as its own line
                # under the planner section: what was handed over, and to whom —
                # the specialist's own events follow under its own header.
                to_agent = str(event.get("to") or "specialist")
                instruction = " ".join(str(event.get("instruction") or "").split())
                budget = max(24, console.width - len(to_agent) - INDENT_DETAIL - 12)
                if len(instruction) > budget:
                    instruction = instruction[: max(0, budget - 3)] + "..."
                line = Text(" " * INDENT_DETAIL + "⇢ ", style="yellow")
                line.append(f"handoff → {to_agent}", style="bold yellow")
                if instruction:
                    line.append(f"  {instruction}", style="dim")
                console.print(line, highlight=False)
                layout.last = "tool"

        elif kind == "search_results":
            if interactive:
                # One line per finished search: the query and what it yielded.
                ok = bool(event.get("ok"))
                query = " ".join(str(event.get("query") or "").split())
                if ok:
                    count = int(event.get("result_count") or 0)
                    tail = f"{count} source(s)"
                else:
                    tail = str(event.get("error") or "failed")
                line = Text(" " * INDENT_OUTPUT + "⤷ ", style="cyan")
                line.append(query or "(no query)", style="cyan")
                line.append(f"  → {tail}", style="dim" if ok else "red")
                console.print(line, highlight=False)
                layout.last = "result"

        elif kind == "tool_result":
            if not interactive:
                continue
            result = event.get("result") or {}
            ok = bool(event.get("ok"))
            fallback = "" if ok else "unknown error"
            # Two shapes reach this branch: the inner ReAct loop nests the text
            # under ``result``, the delegation tools' specialists emit it flat.
            if "output" in event or "error" in event:
                body = str(event.get("output" if ok else "error", fallback))
            else:
                body = str(result.get("output" if ok else "error", fallback))
            clipped, hidden = _preview(body, lines=None if show_trace else PREVIEW_LINES)
            # No measure here: a tool's stdout is data whose columns carry meaning,
            # so it folds at the terminal rather than at the narrative width.
            _prose(
                console,
                clipped,
                indent=INDENT_OUTPUT,
                style="dim" if ok else "red",
                measure=None,
            )
            _hidden_note(console, hidden, indent=INDENT_OUTPUT)
            layout.last = "result"

        elif kind == "error":
            saw_error = True
            error_message = str(event.get("message") or "")
            errors.print(Text(f"error: {error_message}", style="red"), highlight=False)

        elif kind == "retry":
            retries += 1
            if interactive:
                errors.print(
                    Text(
                        f"transient failure ({event.get('message')}) - "
                        f"retrying {event.get('attempt')}/{event.get('max_retries')} "
                        f"in {event.get('delay')}s",
                        style="yellow",
                    ),
                    highlight=False,
                )

        elif kind == "memory":
            # A node announcing the layered-memory snapshot its next brief
            # carries. One dim line: the payload itself is model food, not
            # viewer food — the shape is the news, not the bytes.
            if interactive:
                layout.gap(console)
                console.print(
                    Text(f"🧠 layered memory ready · {event.get('node') or '?'}", style="dim"),
                    highlight=False,
                )
                layout.last = "body"

        elif kind == "heartbeat":
            # The driver's liveness ping: nothing arrived for a while, but the
            # graph has not ended. A long tool call earns a few of these; a run
            # gone genuinely dark now shows *how* long, instead of hanging
            # silently until the user gives up on it.
            if interactive:
                quiet = float(event.get("quiet_for") or 0.0)
                node = str(event.get("node") or "")
                line = Text(" " * INDENT_DETAIL + "⏳ ", style="dim")
                line.append(f"no events for {quiet:.0f}s", style="dim")
                if node:
                    line.append(f" · {node} still working", style="dim")
                console.print(line, highlight=False)
                layout.last = "tool"

        elif kind == "final_answer":
            final = str(event.get("content") or "").strip()
            passed = bool(event.get("passed"))
            attempts = int(event.get("attempts") or attempts)
            max_attempts = int(event.get("max_attempts") or max_attempts)
            exhausted = bool(event.get("exhausted"))

    if exhausted and final and not saw_error:
        # Measured too, even though it goes to stderr: a warning that runs off the
        # right edge is a warning nobody reads.
        _prose(
            errors,
            f"warning: stopped after {attempts} attempt(s) without meeting the "
            "acceptance criteria. Raise --max-attempts, or narrow the task.",
            indent=0,
            style="yellow",
            measure=None,
        )

    if not interactive:
        # pipe-friendly: the answer alone, unformatted, no wrapping or boxes.
        # Only a *verified* answer counts as one. An aborted run wrote no answer,
        # and a failed run wrote a failure report — letting either reach stdout
        # would let a downstream script mistake it for a deliverable, so the
        # report goes to stderr where a human will still see it.
        stream = getattr(console, "file", None) or sys.stdout
        if final and passed and not saw_error:
            stream.write(f"{final}\n")
        elif final:
            errors.print(Text(final, style="red"), highlight=False)
        return 2 if saw_error else (0 if final and passed and not saw_error else 1)

    console.print()
    if final and saw_error:
        # The run broke mid-flight. `final` is the failure text, not an answer —
        # never dress it in green.
        console.print(
            Panel(
                Markdown(final),
                title=(
                    f"interrupted · {attempts}/{max_attempts} attempt(s) · "
                    f"{_short_reason(error_message, 44)}"
                ),
                title_align="left",
                border_style="red",
                padding=(1, 2),
            )
        )
    elif final and not passed:
        # The workflow finished and reported a failure. Labelling it a "final
        # answer" would present a failure report as a deliverable.
        console.print(
            Panel(
                Markdown(final),
                title=(
                    f"❌ not verified · {attempts}/{max_attempts} attempt(s) · "
                    f"{tool_calls} tool call(s)"
                ),
                title_align="left",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    elif final:
        # Markdown, not Text: the model writes headings, bullets and backticks, and
        # a plain Text renderable prints `**bold**` and the backticks literally.
        console.print(
            Panel(
                Markdown(final),
                title=(
                    f"✅ final answer · {attempts} attempt(s) · "
                    f"{turns} model turn(s) · {tool_calls} tool call(s)"
                ),
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )
    elif not saw_error:
        console.print(
            Panel(
                Text("the workflow produced no final answer", style="yellow"),
                title="final answer",
                title_align="left",
                border_style="yellow",
            )
        )

    return 2 if saw_error else (0 if final and passed and not saw_error else 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"SuuClaw {__version__}")
        raise typer.Exit()


@app.command(epilog=HELP_EPILOG)
def main(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Argument(help="Task for the agent, in plain language."),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help=(
                "Sandbox directory; every tool is confined to it. Created if "
                "missing. Optional with --resume: the run continues in the "
                "checkpoint's workspace. Defaults to ./workspace."
            ),
            show_default=True,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            envvar="OPENAI_MODEL",
            help="Chat model id (default: OPENAI_MODEL or gpt-4o-mini).",
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option("--temperature", "-t", help="Sampling temperature."),
    ] = 0.0,
    max_loops: Annotated[
        int,
        typer.Option(
            "--max-steps",
            "--max-loops",
            help="Maximum ReAct loops the code agent gets per delegation call.",
        ),
    ] = DEFAULT_MAX_LOOPS,
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            help=(
                "How many act → verify rounds to allow. Each failed round sends the "
                "run back to the planner with the verifier's findings."
            ),
        ),
    ] = DEFAULT_MAX_ATTEMPTS,
    approval_mode: Annotated[
        Literal["inline", "auto", "deny"],
        typer.Option(
            "--approval-mode",
            help=(
                "Policy for risky bash commands (pip install, curl, servers): "
                "inline asks before running, auto runs them flagged, deny refuses."
            ),
        ),
    ] = "inline",
    checkpoint_mode: Annotated[
        Literal["light", "strict", "off"],
        typer.Option(
            "--checkpoint-mode",
            help=(
                "How much of the run to snapshot under .suuclaw/checkpoints: "
                "light keeps a resumable snapshot, strict also keeps full state "
                "and every event, off disables it."
            ),
        ),
    ] = "light",
    trace_mode: Annotated[
        Literal["on", "off"],
        typer.Option(
            "--trace-mode",
            help=(
                "Write an execution trace under .suuclaw/traces. on records the "
                "run; off writes nothing."
            ),
        ),
    ] = "on",
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help=(
                "Continue an interrupted run from this workspace's latest "
                "checkpoint instead of starting over. A task argument is only "
                "needed if the checkpoint records none."
            ),
        ),
    ] = None,
    retries: Annotated[
        int,
        typer.Option(
            "--retries",
            help=(
                "Extra attempts when a model call fails for a transient reason "
                "(connection dropout, 5xx, 429)."
            ),
        ),
    ] = DEFAULT_RETRIES,
    command_timeout: Annotated[
        float,
        typer.Option("--command-timeout", help="Default bash timeout in seconds."),
    ] = 120.0,
    shell: Annotated[
        str | None,
        typer.Option("--shell", help="Explicit shell prefix, e.g. '/bin/bash -c'."),
    ] = None,
    list_tools: Annotated[
        bool,
        typer.Option("--list-tools", help="Print the registered tools and exit."),
    ] = False,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help=(
                "Open the interactive TUI (a conversation of session turns) instead "
                "of rendering one task. Needs the 'tui' extra: pip install "
                r"'suuclaw\[tui]'."
            ),
        ),
    ] = False,
    show_trace: Annotated[
        bool,
        typer.Option("--show-trace", help="Print full tool output instead of a preview."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Print only the final answer."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Run one task through SuuClaw's supervisor workflow.

    The planner publishes a checkable plan and delegates the actual work: research
    goes to searchAgent, implementation to codeAgent. The verifier then judges the
    result with read-only tools before the outcome is reported. A run that fails
    verification goes back to the planner with the verifier's findings, up to
    --max-attempts times.
    """
    _harden_console()

    # A resume without --workspace continues the run where its checkpoint lives;
    # the driver enforces the same rule, the CLI just resolves it early so the
    # header and --list-tools have a workspace to print.
    run_workspace = resume if resume is not None else (workspace or Path("workspace"))

    console = _make_console()
    # ``inline`` + a terminal means a human is actually there to answer. A
    # headless run gets None, so a risky command is flagged rather than hanging
    # forever on a prompt nobody can see.
    approval_handler: Any | None = None
    if approval_mode == "inline" and console.is_terminal:
        approval_handler = console_approval_handler(console)

    try:
        state = RuntimeState.create(
            run_workspace,
            command_timeout=command_timeout,
            shell=shell,
            model=model,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode,
            # The sketch's spelling of "the default", mapped at the door so the
            # state never carries a value the core has to guess about.
            trace_mode="light" if trace_mode == "on" else "off",
        )
    except (ValueError, OSError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    tools = build_tools(state)

    if list_tools:
        _print_tools(state)
        raise typer.Exit()

    if tui:
        # Imported here, not at module scope: textual is an optional dependency,
        # and a plain install must stay able to run `suuclaw "task"`. The TUI
        # builds its own model, so a missing key is reported by run_tui with the
        # same exit code this command would have used.
        from .tui.app import run_tui

        raise typer.Exit(
            code=run_tui(
                state=state,
                session_workspace=run_workspace,
                temperature=temperature,
                max_attempts=max_attempts,
                max_loops=max_loops,
            )
        )

    if task is None or not task.strip():
        if resume is None:
            typer.echo(ctx.get_help())
            typer.secho(
                '\nPass a task, e.g.:  suuclaw "list the files here" -w ./sandbox',
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        # The checkpoint supplies the task; the driver falls back to it.
        task = ""

    from ..providers.openai_provider import create_model

    try:
        llm = create_model(model, temperature=temperature)
    except RuntimeError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not quiet:
        _print_header(
            console,
            state=state,
            model_name=llm.model_name,
            tools=tools,
            max_loops=max_loops,
            max_attempts=max_attempts,
            approval_mode=approval_mode,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
            resume_workspace=resume,
        )

    code = run_agent(
        llm=llm,
        state=state,
        task=task,
        max_steps=max_loops,
        max_attempts=max_attempts,
        show_trace=show_trace,
        quiet=quiet,
        console=console,
        retries=retries,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        resume_workspace=resume,
    )
    raise typer.Exit(code=code)


def _print_tools(state: RuntimeState) -> None:
    rows = describe_tools(state)
    console = _make_console()
    console.print(
        Text.assemble(
            ("Registered tools ", "bold"),
            (f"({len(rows)})", "bold"),
            (" - workspace: ", "dim"),
            (str(state.workspace), "dim"),
            (" - shell: ", "dim"),
            (_shell_label(state), "dim"),
        ),
    )
    width = max(len(name) for name, _, _ in rows) if rows else 0
    for name, description, signature in rows:
        console.print()
        console.print(
            Text.assemble(
                (name.ljust(width), "bold cyan"),
                ("  ", ""),
                (description, ""),
            ),
        )
        console.print(Text(f"  {' ' * width}  parameters: {signature}", style="dim"))


def run_agent(
    *,
    llm: Any,
    state: RuntimeState,
    task: str,
    max_steps: int = DEFAULT_MAX_LOOPS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    show_trace: bool = False,
    quiet: bool = False,
    console: Console | None = None,
    retries: int = DEFAULT_RETRIES,
    tools: Sequence[Any] | None = None,
    approval_mode: str = "inline",
    approval_handler: Any | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
    resume_workspace: Path | None = None,
) -> int:
    """Drive the workflow for one task and render it. Returns a process exit code.

    ``tools`` is accepted for backwards compatibility and otherwise unused: the
    workflow gives each node its own tool set (delegation tools for the supervisor,
    read-only ones for the verifier), so a caller cannot inject one here.

    With ``resume_workspace`` the driver builds its own state inside the
    checkpoint's workspace — passing ``state`` alongside it is a hard ``ValueError``
    — so the CLI-built state only carries the flag values across that path.
    """
    common: dict[str, Any] = {
        "task": task,
        "max_attempts": max_attempts,
        "max_loops": max_steps,
        "model": llm,
        "retries": retries,
    }
    if resume_workspace is not None:
        events = stream_agent_events(
            resume_workspace=resume_workspace,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
            command_timeout=state.command_timeout,
            shell=state.shell,
            **common,
        )
    else:
        # Not resuming: ``state`` already carries the approval / checkpoint /
        # trace policy, and the driver prefers it over its own kwargs.
        events = stream_agent_events(workspace=state.workspace, state=state, **common)
    return render_agent_events(events, console=console, show_trace=show_trace, quiet=quiet)
