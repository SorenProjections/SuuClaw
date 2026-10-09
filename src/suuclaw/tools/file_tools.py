"""Filesystem tools: read, write, and edit text files inside the workspace.

All three resolve their target through ``RuntimeState.resolve`` first, so a path
traversal attempt (``../../etc/passwd``, an absolute path, a symlink pointing
outside) is rejected before any I/O happens.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.paths import WorkspacePathError
from .registry import SuuClawTool

__all__ = ["FileReadTool", "FileWriteTool", "FileEditTool"]

DEFAULT_READ_LIMIT = 2000
MAX_LINE_CHARS = 2000
_LINE_NUMBER_WIDTH = 6


class FileReadInput(BaseModel):
    """Arguments for :class:`FileReadTool`."""

    file_path: str = Field(description="File to read, absolute or relative to the workspace root.")
    offset: int = Field(default=1, ge=1, description="1-based line number to start reading from.")
    limit: int = Field(
        default=DEFAULT_READ_LIMIT,
        ge=1,
        description="Maximum number of lines to return.",
    )


class FileReadTool(SuuClawTool):
    """Read a UTF-8 text file, returning line-numbered content."""

    name = "file_read"
    description = (
        "Read a text file inside the workspace and return it with 1-based line "
        "numbers formatted as 'N→content'. Use offset/limit to page through long "
        "files. Read before you edit, so old_text can be copied verbatim."
    )
    args_schema = FileReadInput

    def run(
        self,
        file_path: str,
        offset: int = 1,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        try:
            target = self.state.resolve(file_path, must_exist=True)
        except WorkspacePathError as exc:
            return f"Error: blocked by workspace sandbox — {exc}"
        except FileNotFoundError as exc:
            return f"Error: {exc}"

        display = self.state.display(target)
        if target.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            listing = ", ".join(entries[:200]) or "(empty)"
            return f"Error: {display} is a directory, not a file. Contents: {listing}"

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error: cannot read {display}: {exc}"

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            suffix = " (the file has no content)" if text == "" else ""
            return f"{display}: 0 lines{suffix}"

        start = min(offset, total)
        end = min(start + limit - 1, total)
        window = lines[start - 1 : end]

        rendered = []
        for number, line in enumerate(window, start=start):
            shown = line
            if len(shown) > MAX_LINE_CHARS:
                shown = f"{shown[:MAX_LINE_CHARS]}… [line truncated, {len(line)} chars]"
            rendered.append(f"{number:>{_LINE_NUMBER_WIDTH}}→{shown}")

        header = f"{display} — {total} line(s); showing {start}-{end}"
        if start > offset:
            header += f" (offset {offset} is past EOF, clamped)"
        body = self.state.truncate("\n".join(rendered), note="file content truncated")
        return f"{header}\n{body}"


class FileWriteInput(BaseModel):
    """Arguments for :class:`FileWriteTool`."""

    file_path: str = Field(
        description="File to create or overwrite, relative to the workspace root."
    )
    content: str = Field(description="Full file content to write (UTF-8).")


class FileWriteTool(SuuClawTool):
    """Create or overwrite a file, creating parent directories as needed."""

    name = "file_write"
    description = (
        "Create a new file or fully overwrite an existing one inside the workspace. "
        "Missing parent directories are created automatically. Overwriting discards "
        "the previous content — use file_edit for surgical changes."
    )
    args_schema = FileWriteInput

    def run(self, file_path: str, content: str) -> str:
        try:
            target = self.state.resolve(file_path)
        except WorkspacePathError as exc:
            return f"Error: blocked by workspace sandbox — {exc}"

        display = self.state.display(target)
        if target.is_dir():
            return f"Error: {display} is a directory; pass a file path instead."

        existed = target.exists()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Error: cannot write {display}: {exc}"

        verb = "Overwrote" if existed else "Created"
        line_count = len(content.splitlines())
        return f"{verb} {display} — {len(content.encode('utf-8'))} byte(s), {line_count} line(s)."


class FileEditInput(BaseModel):
    """Arguments for :class:`FileEditTool`."""

    file_path: str = Field(description="File to edit, relative to the workspace root.")
    old_text: str = Field(description="Exact text to replace; must occur exactly once.")
    new_text: str = Field(description="Replacement text.")


class FileEditTool(SuuClawTool):
    """Replace one exact, unique snippet inside an existing file."""

    name = "file_edit"
    description = (
        "Replace a snippet of text in an existing file. old_text must match "
        "byte-for-byte (including indentation) and must be unique in the file: zero "
        "matches and multiple matches both fail with an explanatory error, so widen "
        "the snippet with surrounding context when it is ambiguous."
    )
    args_schema = FileEditInput

    def run(self, file_path: str, old_text: str, new_text: str) -> str:
        if not old_text:
            return "Error: old_text must not be empty."

        try:
            target = self.state.resolve(file_path, must_exist=True)
        except WorkspacePathError as exc:
            return f"Error: blocked by workspace sandbox — {exc}"
        except FileNotFoundError as exc:
            return f"Error: {exc}"

        display = self.state.display(target)
        if target.is_dir():
            return f"Error: {display} is a directory, not a file."

        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return f"Error: cannot read {display} as UTF-8 text: {exc}"

        occurrences = text.count(old_text)
        if occurrences == 0:
            return (
                f"Error: old_text not found in {display}. Copy the snippet exactly "
                "(whitespace included) or re-read the file with file_read."
            )
        if occurrences > 1:
            lines = [
                index + 1
                for index, line in enumerate(text.splitlines())
                if old_text.splitlines() and old_text.splitlines()[0] in line
            ]
            near = f" First matching lines: {lines[:10]}." if lines else ""
            return (
                f"Error: old_text matches {occurrences} times in {display} — refusing to "
                f"guess which one to replace.{near} Include more surrounding context so "
                "the snippet is unique."
            )

        index = text.index(old_text)
        start_line = text.count("\n", 0, index) + 1
        updated = f"{text[:index]}{new_text}{text[index + len(old_text) :]}"
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"Error: cannot write {display}: {exc}"

        end_line = start_line + max(len(new_text.splitlines()) - 1, 0)
        span = f"line {start_line}" if end_line == start_line else f"lines {start_line}-{end_line}"
        delta = len(new_text.splitlines()) - len(old_text.splitlines())
        return (
            f"Edited {display} at {span} — replaced {len(old_text)} char(s) with "
            f"{len(new_text)} char(s) ({delta:+d} line(s))."
        )
