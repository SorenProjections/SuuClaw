"""``grep`` tool — regular-expression search across the workspace.

Implemented with the stdlib ``re`` + ``os.walk`` (no ripgrep binary required), so
it behaves identically on Windows, macOS, and Linux. VCS internals, virtualenvs,
and build caches are pruned by default to keep results signal-rich.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ..core.paths import WorkspacePathError
from .registry import SuuClawTool

__all__ = ["GrepTool"]

DEFAULT_HEAD_LIMIT = 100
MAX_MATCH_LINE_CHARS = 400
_SNIPPET_BYTES = 8192

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".tox",
    }
)


class GrepInput(BaseModel):
    """Arguments for :class:`GrepTool`."""

    pattern: str = Field(description="Regular expression (Python 're' syntax) to search for.")
    path: str = Field(
        default=".",
        description="File or directory to search, relative to the workspace root.",
    )
    glob: str | None = Field(
        default=None,
        description="Optional glob filter, e.g. '*.py', '*.md', or 'src/**/*.py'.",
    )
    head_limit: int = Field(
        default=DEFAULT_HEAD_LIMIT,
        ge=1,
        description="Maximum number of matching lines to return.",
    )
    ignore_case: bool = Field(default=False, description="Case-insensitive matching.")


class GrepTool(SuuClawTool):
    """Search workspace files with a regex and report ``path:line:text`` hits."""

    name = "grep"
    description = (
        "Recursively search workspace files with a regular expression. Returns "
        "'relative/path:line:text' for each matching line (line text is trimmed and "
        "truncated). Skips VCS metadata, virtualenvs, node_modules and build caches. "
        "Use it to locate code, then file_read the exact lines before editing."
    )
    args_schema = GrepInput

    def run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = DEFAULT_HEAD_LIMIT,
        ignore_case: bool = False,
    ) -> str:
        if not pattern:
            return "Error: pattern must not be empty."
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return f"Error: invalid regular expression {pattern!r}: {exc}"

        try:
            target = self.state.resolve(path, must_exist=True)
        except WorkspacePathError as exc:
            return f"Error: blocked by workspace sandbox — {exc}"
        except FileNotFoundError as exc:
            return f"Error: {exc}"

        if target.is_file():
            files = [target]
        else:
            files = self._walk(target, glob)

        if not files:
            scope = self.state.display(target)
            suffix = f" matching glob {glob!r}" if glob else ""
            return f"No files to search in {scope}{suffix}."

        matches: list[str] = []
        scanned = 0
        truncated = False

        for candidate in files:
            scanned += 1
            try:
                if candidate.stat().st_size == 0 or _looks_binary(candidate):
                    continue
                content = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            display = self.state.display(candidate)
            for number, line in enumerate(content.splitlines(), start=1):
                if not regex.search(line):
                    continue
                snippet = line.strip()
                if len(snippet) > MAX_MATCH_LINE_CHARS:
                    snippet = f"{snippet[:MAX_MATCH_LINE_CHARS]}…"
                matches.append(f"{display}:{number}:{snippet}")
                if len(matches) >= head_limit:
                    truncated = True
                    break
            if truncated:
                break

        flags = " (case-insensitive)" if ignore_case else ""
        if not matches:
            return (
                f"No matches for pattern {pattern!r}{flags} in {self.state.display(target)} "
                f"across {scanned} file(s)."
            )

        header = (
            f"{len(matches)} match(es) for pattern {pattern!r}{flags} in "
            f"{self.state.display(target)} across {scanned} file(s)"
            + (f" — stopped at head_limit={head_limit}, more may exist" if truncated else "")
        )
        body = self.state.truncate("\n".join(matches), note="grep output truncated")
        return f"{header}\n{body}"

    # -- internals --------------------------------------------------------
    def _walk(self, root: Path, glob: str | None) -> list[Path]:
        """Depth-first walk, pruning :data:`SKIP_DIRS`, honouring ``glob``."""
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for filename in sorted(filenames):
                absolute = Path(dirpath) / filename
                if glob and not _glob_matches(absolute, root, glob):
                    continue
                found.append(absolute)
        return found


def _glob_matches(candidate: Path, root: Path, glob: str) -> bool:
    """Match a glob against both the basename and the workspace-relative path."""
    patterns = [glob]
    if not glob.startswith("**/"):
        patterns.append(f"**/{glob}")
    relative = candidate.as_posix()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        pass
    return any(
        fnmatch.fnmatch(candidate.name, pat) or fnmatch.fnmatch(relative, pat) for pat in patterns
    )


def _looks_binary(candidate: Path) -> bool:
    """Cheap binary sniff: a NUL byte in the first few KB means 'not text'."""
    try:
        with candidate.open("rb") as handle:
            chunk = handle.read(_SNIPPET_BYTES)
    except OSError:
        return True
    return b"\x00" in chunk
