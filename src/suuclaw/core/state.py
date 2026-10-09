"""Shared runtime state.

``RuntimeState`` is the single object handed to every tool. It owns the
workspace root, the default command timeout, and the output-size budget, so a
tool never has to guess (or hardcode) those policy values. It also carries the
run's policies: how risky shell commands are approved, how the run is
checkpointed, and whether it is traced.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .approval import DEFAULT_APPROVAL_MODE, ApprovalHandler, normalize_approval_mode
from .checkpoint import DEFAULT_CHECKPOINT_MODE, normalize_checkpoint_mode
from .paths import ensure_workspace, resolve_in_workspace, to_display_path
from .trace import DEFAULT_TRACE_MODE, normalize_trace_mode

__all__ = ["RuntimeState"]

DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RuntimeState:
    """Mutable per-invocation context shared by all tools.

    Attributes:
        workspace: Sandbox root. Every tool path is resolved against it.
        command_timeout: Default timeout (seconds) for ``bash`` tool calls.
        max_output_chars: Output budget; longer tool output is truncated with a marker.
        shell: Optional explicit shell command, e.g. ``"/bin/bash -c"`` or ``"pwsh -c"``.
        env: Extra environment variables merged over ``os.environ`` for ``bash``.
        created_at: Timestamp of state creation (UTC).
        metadata: Free-form bag for things like the model name or run id.
        approval_mode: Policy for risky ``bash`` commands — ``inline`` (default,
            ask the handler), ``auto`` (run and flag), ``deny`` (refuse).
        approval_handler: Called with an ``ApprovalRequest`` in ``inline`` mode.
            ``None`` means nobody is attached to answer, so the command runs
            flagged rather than hanging on a prompt no one can see.
        checkpoint_mode: How much of a run to write to
            ``<workspace>/.suuclaw/checkpoints`` — ``light`` (default, snapshot
            the workspace on every node switch), ``strict`` (also the graph state
            and the event log), ``off`` (nothing).
        trace_mode: How much of a run to write to
            ``<workspace>/.suuclaw/traces/<trace_id>`` — ``light`` (default,
            the summary and the timeline), ``strict`` (also the event log, so a
            crash still leaves evidence), ``off`` (nothing).
        trace_id: Identity of this run's trace. Empty asks the recorder to mint
            one; set it to record several runs under one id, e.g. a task that was
            resumed after a crash.
    """

    workspace: Path
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    shell: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    approval_mode: str = DEFAULT_APPROVAL_MODE
    approval_handler: ApprovalHandler | None = None
    checkpoint_mode: str = DEFAULT_CHECKPOINT_MODE
    trace_mode: str = DEFAULT_TRACE_MODE
    trace_id: str = ""

    def __post_init__(self) -> None:
        self.workspace = ensure_workspace(self.workspace)
        # Normalised here rather than on every read: the value arrives from a
        # CLI flag or a config file, and a typo should degrade to the default
        # mode at the door instead of reaching a tool as an unknown policy.
        self.approval_mode = normalize_approval_mode(self.approval_mode)
        self.checkpoint_mode = normalize_checkpoint_mode(self.checkpoint_mode)
        self.trace_mode = normalize_trace_mode(self.trace_mode)
        if self.max_output_chars < 1:
            raise ValueError("max_output_chars must be >= 1")
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be > 0")

    # -- constructors -----------------------------------------------------
    @classmethod
    def create(
        cls,
        workspace: str | os.PathLike[str],
        *,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        shell: str | None = None,
        env: Mapping[str, str] | None = None,
        approval_mode: str | None = None,
        approval_handler: ApprovalHandler | None = None,
        checkpoint_mode: str | None = None,
        trace_mode: str | None = None,
        trace_id: str = "",
        **metadata: Any,
    ) -> RuntimeState:
        """Build a state object from loose CLI-ish values (strings are fine)."""
        return cls(
            workspace=Path(os.fspath(workspace)),
            command_timeout=float(command_timeout),
            max_output_chars=int(max_output_chars),
            shell=shell,
            env=dict(env or {}),
            approval_mode=approval_mode or DEFAULT_APPROVAL_MODE,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode or DEFAULT_CHECKPOINT_MODE,
            trace_mode=trace_mode or DEFAULT_TRACE_MODE,
            trace_id=trace_id or "",
            metadata=dict(metadata),
        )

    # -- path helpers -----------------------------------------------------
    def resolve(
        self,
        raw_path: str | os.PathLike[str] | None = None,
        *,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a path inside the workspace, raising on sandbox escapes."""
        return resolve_in_workspace(self.workspace, raw_path, must_exist=must_exist)

    def display(self, path: str | os.PathLike[str]) -> str:
        """Workspace-relative, forward-slash form of ``path`` for tool output."""
        return to_display_path(self.workspace, path)

    # -- output helpers ---------------------------------------------------
    def truncate(self, text: str, *, limit: int | None = None, note: str = "truncated") -> str:
        """Cap ``text`` at the output budget, appending a visible marker."""
        budget = self.max_output_chars if limit is None else limit
        if budget < 1 or len(text) <= budget:
            return text
        dropped = len(text) - budget
        return f"{text[:budget]}\n... [{note}: {dropped} more character(s) withheld]"

    @property
    def name(self) -> str:
        """Basename of the workspace, handy for prompts and logs."""
        return self.workspace.name
