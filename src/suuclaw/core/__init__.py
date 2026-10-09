"""Core runtime primitives: workspace paths, shared state, command approval,
run checkpoints, execution traces, and the event streams.

Two streams live in :mod:`suuclaw.core.agent` and they are not interchangeable:
:func:`~suuclaw.core.agent.stream_agent_events` drives the whole plan → act →
verify workflow and is what a front end subscribes to;
:func:`~suuclaw.core.agent.stream_session_events` is the session-aware front
door that routes each turn through the entry graph first;
:func:`~suuclaw.core.agent.stream_react_events` is the single-model ReAct loop
the graph's nodes run internally.

:mod:`suuclaw.core.approval` is the third primitive: the risk classifier and
the three approval modes that stand between a ``bash`` command and the shell.
:mod:`suuclaw.core.checkpoint` is the fourth: the on-disk snapshot that lets a
dead run be picked back up. :mod:`suuclaw.core.trace` is the fifth: the same
event stream written down as a record of what the run did, which is a different
question from how to get it back. :mod:`suuclaw.core.session` is the sixth:
the cross-run conversation log the entry graph's intent router and chat
responder read as their session context.
"""

from __future__ import annotations

from .agent import (
    ACTOR_PROMPT,
    execute_tool,
    index_tools,
    iter_workflow_events,
    message_text,
    stream_agent_events,
    stream_react_events,
    stream_session_events,
)
from .approval import (
    DEFAULT_APPROVAL_MODE,
    RISK_PATTERNS,
    VALID_APPROVAL_MODES,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    classify_command_risk,
    console_approval_handler,
    normalize_approval_mode,
)
from .checkpoint import (
    DEFAULT_CHECKPOINT_MODE,
    VALID_CHECKPOINT_MODES,
    CheckpointError,
    CheckpointManager,
    build_recovery_markdown,
    normalize_checkpoint_mode,
    resume_command,
    snapshot_workspace_git,
)
from .paths import (
    WorkspacePathError,
    ensure_workspace,
    is_within,
    resolve_in_workspace,
    to_display_path,
)
from .session import (
    MAX_SESSION_CONTEXT,
    MAX_TURN_CONTENT,
    SESSION_FILE,
    SESSION_ROOT,
    SESSION_SUMMARY_FILE,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)
from .state import RuntimeState
from .trace import (
    DEFAULT_TRACE_MODE,
    VALID_TRACE_MODES,
    TraceRecorder,
    build_timeline_markdown,
    new_trace_id,
    normalize_trace_mode,
)

__all__ = [
    "ACTOR_PROMPT",
    "DEFAULT_APPROVAL_MODE",
    "DEFAULT_CHECKPOINT_MODE",
    "DEFAULT_TRACE_MODE",
    "MAX_SESSION_CONTEXT",
    "MAX_TURN_CONTENT",
    "RISK_PATTERNS",
    "SESSION_FILE",
    "SESSION_ROOT",
    "SESSION_SUMMARY_FILE",
    "VALID_APPROVAL_MODES",
    "VALID_CHECKPOINT_MODES",
    "VALID_TRACE_MODES",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalRequest",
    "CheckpointError",
    "CheckpointManager",
    "RuntimeState",
    "TraceRecorder",
    "WorkspacePathError",
    "append_assistant_turn",
    "append_user_turn",
    "build_recovery_markdown",
    "build_session_context",
    "build_timeline_markdown",
    "classify_command_risk",
    "console_approval_handler",
    "ensure_workspace",
    "execute_tool",
    "index_tools",
    "is_within",
    "iter_workflow_events",
    "load_or_create_session",
    "message_text",
    "new_trace_id",
    "normalize_approval_mode",
    "normalize_checkpoint_mode",
    "normalize_trace_mode",
    "resolve_in_workspace",
    "resume_command",
    "save_session",
    "snapshot_workspace_git",
    "stream_agent_events",
    "stream_react_events",
    "stream_session_events",
    "to_display_path",
]
