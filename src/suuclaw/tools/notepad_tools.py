"""The notepad: durable notes that survive context compression.

Why a file, not a channel
--------------------------
The todo plan is *structured* data and lives in graph state; the notepad is
*unstructured* accumulation — decisions, file paths, blockers, "the test needs
a marker file" — and unstructured accumulation belongs in the workspace, next
to the work it describes. A single ``notepad.md`` file is also the substrate
the layered-memory snapshot (``agents/code_agent``) reads, so "remember this
across steps" has exactly one mechanism: write it to the notepad.

Both tools are ordinary :class:`~suuclaw.tools.registry.SuuClawTool` s:
failures come back as ``"Error: ..."`` text, and output passes through
``state.truncate`` so a bloated notepad cannot blow the prompt on its own.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..core.state import RuntimeState
from .registry import SuuClawTool

__all__ = [
    "NOTEPAD_FILENAME",
    "NotepadAppendInput",
    "NotepadAppendTool",
    "NotepadReadInput",
    "NotepadReadTool",
    "read_notepad",
]

#: The one file both tools (and the memory snapshot) agree on.
NOTEPAD_FILENAME = "notepad.md"

#: Separator between appended entries, so a notepad reads as dated blocks
#: rather than one undifferentiated wall of text.
ENTRY_SEPARATOR = "\n---\n\n"


def _notepad_path(state: RuntimeState) -> Path:
    """The notepad's absolute path, resolved inside the sandbox like any file."""
    return state.resolve(NOTEPAD_FILENAME)


def read_notepad(state: RuntimeState) -> str:
    """The whole notepad as text, or ``""`` when nothing was noted yet.

    The shared read path: :class:`NotepadReadTool` shows it to the model and
    the layered-memory snapshot folds it into the next prompt, so both see the
    same bytes.
    """
    path = _notepad_path(state)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


class NotepadAppendInput(BaseModel):
    """Arguments for :class:`NotepadAppendTool`."""

    text: str = Field(
        description=(
            "The durable note to append: a decision made, an important file, a "
            "blocker, or next-step context. Plain text; one topic per call."
        )
    )


class NotepadAppendTool(SuuClawTool):
    """Append one entry to the workspace notepad."""

    name = "notepad_append"
    description = (
        "Append a durable note to the workspace notepad (notepad.md). Use it for "
        "findings, decisions, important file paths, blockers and next-step context "
        "that must survive later context compression. Notes are read back with "
        "notepad_read."
    )
    args_schema = NotepadAppendInput

    def run(  # type: ignore[override]
        self,
        text: str,
    ) -> str:
        block = str(text or "").strip()
        if not block:
            return "Error: notepad_append needs non-empty text to remember."

        path = _notepad_path(self.state)
        existing = read_notepad(self.state)
        if existing.strip():
            path.write_text(f"{existing}{ENTRY_SEPARATOR}{block}\n", encoding="utf-8")
        else:
            path.write_text(f"{block}\n", encoding="utf-8")
        return f"Appended {len(block)} character(s) to {NOTEPAD_FILENAME}."


class NotepadReadInput(BaseModel):
    """No arguments: the notepad is one file, and you want all of it."""


class NotepadReadTool(SuuClawTool):
    """Read the whole workspace notepad back."""

    name = "notepad_read"
    description = (
        "Read the workspace notepad (notepad.md) — the durable notes written by "
        "notepad_append. Use it to recover context that fell out of the "
        "conversation, e.g. after the history was compressed."
    )
    args_schema = NotepadReadInput

    def run(self, **_kwargs: object) -> str:
        text = read_notepad(self.state)
        if not text.strip():
            return "(the notepad is empty — nothing has been noted yet)"
        return self.state.truncate(text, note="notepad truncated")
