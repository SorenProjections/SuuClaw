"""The startup splash: an ASCII wordmark, in ink and in motion.

What the app shows for a second or so when it opens, and what it is left showing
underneath::

    /// SuuClaw
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Local agents · Work in view
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three layers, and they are separable on purpose:

* :func:`wordmark` — the art: plain ``str`` rows, ASCII only, always the same
  height.
* :func:`banner` — the *resting* state, the four lines above. The last frame of
  :func:`logo_frames` is exactly this, so a caller that cannot animate (a dumb
  terminal, a test, a pipe) can render this one and be right.
* :func:`logo_frames` — the animation: a generator of rich ``Text`` frames. The
  wordmark prints a row at a time while the ink drifts, the rules draw
  themselves left to right, the stage line types out, and then the art folds
  away from the top and leaves the banner behind.

The ASCII wordmark and three-slash claw mark echo the browser logo's curled S
and three claws. They use only fixed-width characters, so the mark stays aligned
in narrow terminals and copied transcripts.

Why a widget and not a write to the stream: the stream is a ``RichLog``, which
is append-only — two dozen frames would be two dozen lines of history for
something worth reading once. :class:`SplashLogo` is a ``Static`` repainted in
place, and the app drops it when the first turn starts: a logo is worth nine
rows until there is work to show, and not after.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

__all__ = [
    "CLAW_MARK",
    "LOGO_COLORS",
    "LOGO_STAGE",
    "LOGO_TITLE",
    "PAW",
    "RULE_CHAR",
    "RULE_WIDTH",
    "SPLASH_INTERVAL",
    "SplashLogo",
    "banner",
    "logo_frames",
    "wordmark",
]

#: The banner's texts, in the order they are printed.
CLAW_MARK = "///"
# Compatibility for integrations that imported the old decorative constant.
PAW = CLAW_MARK
LOGO_TITLE = "SuuClaw"
LOGO_STAGE = "Local agents · Work in view"
RULE_CHAR = "━"
RULE_WIDTH = 34

#: Seconds between frames. The sequence is a little over two dozen frames, so
#: this is a splash of about a second and a half — long enough to read as an
#: animation, and over before someone who does not care for it finishes typing.
SPLASH_INTERVAL = 0.055

#: The ink. A sea-glass → cyan → cool-blue ramp. :func:`_sample_palette` treats it
#: as a loop rather than a line, so the bands wrap instead of ending in a seam.
LOGO_COLORS: tuple[str, ...] = ("#53e1c6", "#38c8d4", "#68a9ef", "#92d2ed", "#53e1c6")

#: How much of the stage line appears per frame, in characters.
_TYPE_STEP = 6

#: Frames a rule takes to draw itself from the left.
_RULE_STEPS = 3

#: How far the ink band travels per revealed row, and the diagonal's two slopes.
_INK_DRIFT = 0.11
_ROW_DRIFT = 0.09
_COL_DRIFT = 0.45

#: The rules hold one colour. A rule that shimmers with the art reads as noise
#: rather than as motion: a rule is a thing being *drawn*, not a thing inky.
_RULE_PHASE = 0.12

#: Five rows, and every row of a glyph is the same width as its siblings — that
#: is what keeps the letters on a shared baseline. Rows are joined with one space
#: column; each finished row is right-trimmed by :func:`wordmark`, because a
#: line's trailing spaces do nothing but ride along in a copy-paste.
_GLYPH_HEIGHT = 5
_GLYPHS: dict[str, tuple[str, ...]] = {
    "S": ("  ____  ", " / ___| ", " \\___ \\ ", "  ___) |", " |____/ "),
    "u": ("       ", " _   _ ", "| | | |", "| |_| |", " \\__,_|"),
    "C": ("  ____ ", " / ___|", "| |    ", "| |___ ", " \\____|"),
    "l": (" _ ", "| |", "| |", "| |", "|_|"),
    "a": ("       ", "  __ _ ", " / _` |", "| (_| |", " \\__,_|"),
    "w": ("          ", "__      __", "\\ \\ /\\ / /", " \\ V  V / ", "  \\_/\\_/  "),
    " ": (" ", " ", " ", " ", " "),
    # A letter the table does not know is data, not a bug: the splash is a
    # decoration, and a decoration is never worth a traceback.
    "?": (" ___ ", "|__ \\", "  / /", " /_/ ", "(_)  "),
}


def wordmark(word: str = LOGO_TITLE) -> tuple[str, ...]:
    """``word`` as ASCII art: fixed height, right-trimmed, one space between glyphs.

    An unknown character falls back to a boxed ``?`` rather than raising, which
    is what lets a caller put any name in the banner.
    """
    rows: list[str] = []
    for line in range(_GLYPH_HEIGHT):
        cells = [_GLYPHS.get(char) or _GLYPHS.get(char.upper()) or _GLYPHS["?"] for char in word]
        rows.append(" ".join(cell[line] for cell in cells).rstrip())
    return tuple(rows)


def _rgb(color: str) -> tuple[int, int, int]:
    """``"#22d3ee"`` → ``(34, 211, 238)``."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix(start: str, end: str, t: float) -> str:
    """One step along the straight line between two hex colours."""
    left, right = _rgb(start), _rgb(end)
    parts = (round(a + (b - a) * t) for a, b in zip(left, right, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*parts)


def _sample_palette(t: float) -> str:
    """Sample the ramp at ``t``, wrapping: the palette is a loop, not a line."""
    span = len(LOGO_COLORS)
    scaled = (t % 1.0) * span
    index = int(scaled)
    return _mix(LOGO_COLORS[index % span], LOGO_COLORS[(index + 1) % span], scaled - index)


def _ink(rows: Sequence[str], *, phase: float) -> list[Text]:
    """Tint art rows with the ramp — banded diagonally, drifting by ``phase``.

    Whitespace is appended unstyled: a space has no foreground to colour, and
    styling them anyway would only lengthen the span list.
    """
    width = max((len(row) for row in rows), default=1) or 1
    tinted: list[Text] = []
    for row_index, row in enumerate(rows):
        line = Text()
        for column, char in enumerate(row):
            if char == " ":
                line.append(char)
                continue
            t = phase + row_index * _ROW_DRIFT + (column / width) * _COL_DRIFT
            line.append(char, style=_sample_palette(t))
        tinted.append(line)
    return tinted


def _title_line() -> Text:
    """``/// SuuClaw``: the claws in sea glass, the name in cool blue."""
    line = Text()
    line.append(f"{CLAW_MARK} ", style=f"bold {_sample_palette(0.0)}")
    line.append(LOGO_TITLE, style=f"bold {_sample_palette(0.5)}")
    return line


def _rule_line(width: int = RULE_WIDTH, fill: float = 1.0) -> Text:
    """The ━ rule, ``fill`` of the way drawn, coloured across its length."""
    drawn = max(int(round(width * fill)), 0)
    return _ink([RULE_CHAR * drawn], phase=_RULE_PHASE)[0]


def _stage_line(stage: str = LOGO_STAGE) -> Text:
    """The stage line, dim enough that the wordmark stays the loudest thing.

    Right-trimmed because the typewriter passes *prefixes* of the line, and a
    prefix that ends on a space would carry an invisible trailing space into
    every frame — which is the kind of thing that then ambushes a copy-paste of
    the whole splash.
    """
    return Text(stage.rstrip(), style=f"{_sample_palette(0.62)} dim")


def _frame(lines: Sequence[Text]) -> Text:
    """Join prepared lines into one frame."""
    frame = Text()
    for index, line in enumerate(lines):
        if index:
            frame.append("\n")
        frame.append_text(line)
    return frame


def banner(*, rule_width: int = RULE_WIDTH, stage: str = LOGO_STAGE) -> Text:
    """The resting banner — the state the splash settles into.

    Also the last frame of :func:`logo_frames`, and the whole of what a caller
    without a terminal should render.
    """
    return _frame(
        [
            _title_line(),
            _rule_line(rule_width),
            _stage_line(stage),
            _rule_line(rule_width),
        ]
    )


def logo_frames(*, rule_width: int = RULE_WIDTH, stage: str = LOGO_STAGE) -> Iterator[Text]:
    """The animation, one rich ``Text`` frame at a time.

    Five acts, and the last frame is exactly :func:`banner`:

    1. the wordmark prints, top row first, with the ink drifting as it goes;
    2. ``/// SuuClaw`` settles under the art;
    3. the first rule draws itself left to right;
    4. the stage line types out;
    5. the second rule draws, and then the art folds away from the *top*, one row
       per frame, leaving the banner.

    Frame counts are fixed, not timed: the caller owns the clock (the widget's
    timer, a test's loop), so a slow terminal plays the same animation slowly
    instead of skipping frames it could not keep up with.
    """
    art_rows = wordmark()
    # The art is inked once, at the phase the last reveal frame left it in, so
    # the animation does not jump colour when the printing stops.
    art = _ink(art_rows, phase=len(art_rows) * _INK_DRIFT)
    lines: list[Text] = list(art)

    # 1. the wordmark prints row by row, ink drifting with each revealed row
    for drawn in range(1, len(art_rows) + 1):
        yield _frame(_ink(art_rows, phase=drawn * _INK_DRIFT)[:drawn])

    # 2. the name settles under the art
    lines.append(_title_line())
    yield _frame(lines)

    # 3. the first rule draws itself
    for step in range(1, _RULE_STEPS + 1):
        yield _frame(lines + [_rule_line(rule_width, step / _RULE_STEPS)])
    lines.append(_rule_line(rule_width))

    # 4. the stage line types out, in chunks
    for end in range(_TYPE_STEP, len(stage), _TYPE_STEP):
        yield _frame(lines + [_stage_line(stage[:end])])
    lines.append(_stage_line(stage))

    # 5. the second rule draws, and the art folds away from the top
    for step in range(1, _RULE_STEPS + 1):
        yield _frame(lines + [_rule_line(rule_width, step / _RULE_STEPS)])
    lines.append(_rule_line(rule_width))
    tail = lines[len(art) :]
    for drop in range(1, len(art) + 1):
        yield _frame(art[drop:] + tail)


class SplashLogo(Static):
    """The splash widget: plays :func:`logo_frames` in place, once.

    Repainted rather than appended (see the module docstring), and **skippable**:
    clicking it jumps straight to the resting banner, because a decoration that
    cannot be dismissed is a tax on someone who has seen it before.

    Frames are pulled one per timer tick from a generator this widget owns, and
    every tick runs on the app's event loop — so the generator is only ever
    advanced from the thread that created it, which is what a generator needs.
    """

    DEFAULT_CSS = """
    SplashLogo {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, *, interval: float = SPLASH_INTERVAL, id: str | None = None) -> None:
        super().__init__(id=id)
        self.interval = float(interval)
        self._frames = logo_frames()
        self._timer: Timer | None = None
        self._settled = False

    def on_mount(self) -> None:
        # The first frame goes up immediately, so there is never a blank row
        # where the logo is about to be.
        self.update(self._next_frame())
        if not self._settled:
            self._timer = self.set_interval(self.interval, self._advance)

    def on_click(self) -> None:
        self.skip()

    def skip(self) -> None:
        """Stop animating and show the resting banner."""
        if not self._settled:
            self._settle(banner())

    def _next_frame(self) -> Text:
        """The next frame, or the resting banner once the sequence is spent."""
        try:
            return next(self._frames)
        except StopIteration:
            self._settled = True
            return banner()

    def _advance(self) -> None:
        self.update(self._next_frame())
        if self._settled:
            # One tick past the end stops the clock rather than leaving a timer
            # waking up forever to repaint the same banner.
            self._settle(banner())

    def _settle(self, frame: Text) -> None:
        self._settled = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.update(frame)
