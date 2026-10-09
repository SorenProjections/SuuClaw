"""Environment-backed configuration for the Feishu gateway."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..core.paths import runtime_path

__all__ = ["FeishuGatewayConfig", "GatewayConfigError"]


class GatewayConfigError(ValueError):
    """Raised when gateway configuration is missing or invalid."""


def _text(env: Mapping[str, str], name: str, default: str = "") -> str:
    return str(env.get(name, default) or "").strip()


def _integer(env: Mapping[str, str], name: str, default: int, *, minimum: int = 1) -> int:
    raw = _text(env, name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise GatewayConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise GatewayConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _number(env: Mapping[str, str], name: str, default: float, *, minimum: float = 0) -> float:
    raw = _text(env, name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise GatewayConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise GatewayConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _text(env, name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise GatewayConfigError(f"{name} must be true or false, got {raw!r}")


def _choice(env: Mapping[str, str], name: str, default: str, choices: set[str]) -> str:
    value = _text(env, name, default).lower()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise GatewayConfigError(f"{name} must be one of {expected}, got {value!r}")
    return value


def _csv(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _text(env, name).split(",") if part.strip())


def _path(raw: str, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class FeishuGatewayConfig:
    """All policy and runtime knobs needed by the gateway."""

    app_id: str
    app_secret: str = field(repr=False)
    workspace: Path = Path("workspace")
    session_root: Path = Path("workspace/.suuclaw/feishu/conversations")
    encrypt_key: str = field(default="", repr=False)
    verification_token: str = field(default="", repr=False)
    model: str | None = None
    temperature: float = 0.0
    max_attempts: int = 3
    max_loops: int = 25
    retries: int = 2
    command_timeout: float = 120.0
    approval_mode: str = "deny"
    checkpoint_mode: str = "light"
    trace_mode: str = "light"
    max_concurrency: int = 4
    connect_timeout: float = 120.0
    ws_proxy: str = "direct"
    ws_open_timeout: float = 30.0
    processing_text: str = "收到，正在判断任务类型…"
    workflow_progress: bool = True
    workflow_heartbeat_seconds: float = 60.0
    entry_timeout: float = 30.0
    deliver_artifacts: bool = True
    artifact_max_files: int = 8
    artifact_max_megabytes: float = 25.0
    unsupported_text: str = "目前只支持文本和富文本消息。"
    error_text: str = "处理失败，请稍后重试。"
    reply_in_thread: bool = False
    dm_policy: str = "open"
    group_policy: str = "open"
    require_mention: bool = True
    respond_to_mention_all: bool = False
    allow_from: tuple[str, ...] = ()
    deny_from: tuple[str, ...] = ()
    group_allowlist: tuple[str, ...] = ()
    group_blocklist: tuple[str, ...] = ()
    admins: tuple[str, ...] = ()
    dedup_ttl_seconds: int = 12 * 3600
    stale_window_ms: int = 30 * 60 * 1000
    text_batch_delay_ms: int = 600
    text_chunk_limit: int = 3500
    outbound_attempts: int = 3

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        base: str | os.PathLike[str] | None = None,
    ) -> FeishuGatewayConfig:
        """Build and validate a config without mutating the environment."""
        values = os.environ if env is None else env
        base_path = Path.cwd() if base is None else Path(base)
        base_path = base_path.expanduser().resolve()

        app_id = _text(values, "FEISHU_APP_ID")
        app_secret = _text(values, "FEISHU_APP_SECRET")
        missing = [
            name
            for name, value in (
                ("FEISHU_APP_ID", app_id),
                ("FEISHU_APP_SECRET", app_secret),
            )
            if not value
        ]
        if missing:
            raise GatewayConfigError(f"missing required setting(s): {', '.join(missing)}")

        workspace = _path(_text(values, "SUUCLAW_FEISHU_WORKSPACE", "workspace"), base=base_path)
        session_raw = _text(values, "SUUCLAW_FEISHU_SESSION_ROOT")
        session_root = (
            _path(session_raw, base=base_path)
            if session_raw
            else runtime_path(workspace, "feishu", "conversations")
        )

        return cls(
            app_id=app_id,
            app_secret=app_secret,
            workspace=workspace,
            session_root=session_root,
            encrypt_key=_text(values, "FEISHU_ENCRYPT_KEY"),
            verification_token=_text(values, "FEISHU_VERIFICATION_TOKEN"),
            model=_text(values, "SUUCLAW_FEISHU_MODEL") or None,
            temperature=_number(values, "SUUCLAW_FEISHU_TEMPERATURE", 0.0),
            max_attempts=_integer(values, "SUUCLAW_FEISHU_MAX_ATTEMPTS", 3),
            max_loops=_integer(values, "SUUCLAW_FEISHU_MAX_LOOPS", 25),
            retries=_integer(values, "SUUCLAW_FEISHU_RETRIES", 2, minimum=0),
            command_timeout=_number(values, "SUUCLAW_FEISHU_COMMAND_TIMEOUT", 120.0, minimum=0.1),
            approval_mode=_choice(values, "SUUCLAW_FEISHU_APPROVAL_MODE", "deny", {"deny", "auto"}),
            checkpoint_mode=_choice(
                values, "SUUCLAW_FEISHU_CHECKPOINT_MODE", "light", {"light", "strict", "off"}
            ),
            trace_mode=_choice(
                values, "SUUCLAW_FEISHU_TRACE_MODE", "light", {"light", "strict", "off"}
            ),
            max_concurrency=_integer(values, "SUUCLAW_FEISHU_MAX_CONCURRENCY", 4),
            connect_timeout=_number(values, "SUUCLAW_FEISHU_CONNECT_TIMEOUT", 120.0, minimum=0.1),
            ws_proxy=_text(values, "SUUCLAW_FEISHU_WS_PROXY", "direct"),
            ws_open_timeout=_number(values, "SUUCLAW_FEISHU_WS_OPEN_TIMEOUT", 30.0, minimum=0.1),
            processing_text=_text(
                values, "SUUCLAW_FEISHU_PROCESSING_TEXT", "收到，正在判断任务类型…"
            ),
            workflow_progress=_boolean(values, "SUUCLAW_FEISHU_WORKFLOW_PROGRESS", True),
            workflow_heartbeat_seconds=_number(
                values,
                "SUUCLAW_FEISHU_WORKFLOW_HEARTBEAT_SECONDS",
                60.0,
                minimum=10.0,
            ),
            entry_timeout=_number(
                values,
                "SUUCLAW_FEISHU_ENTRY_TIMEOUT",
                30.0,
                minimum=1.0,
            ),
            deliver_artifacts=_boolean(values, "SUUCLAW_FEISHU_DELIVER_ARTIFACTS", True),
            artifact_max_files=_integer(values, "SUUCLAW_FEISHU_ARTIFACT_MAX_FILES", 8),
            artifact_max_megabytes=_number(
                values,
                "SUUCLAW_FEISHU_ARTIFACT_MAX_MB",
                25.0,
                minimum=0.1,
            ),
            unsupported_text=_text(
                values, "SUUCLAW_FEISHU_UNSUPPORTED_TEXT", "目前只支持文本和富文本消息。"
            ),
            error_text=_text(values, "SUUCLAW_FEISHU_ERROR_TEXT", "处理失败，请稍后重试。"),
            reply_in_thread=_boolean(values, "SUUCLAW_FEISHU_REPLY_IN_THREAD", False),
            dm_policy=_choice(
                values,
                "SUUCLAW_FEISHU_DM_POLICY",
                "open",
                {"open", "allowlist", "blocklist", "disabled"},
            ),
            group_policy=_choice(
                values,
                "SUUCLAW_FEISHU_GROUP_POLICY",
                "open",
                {"open", "allowlist", "blocklist", "admin_only", "disabled"},
            ),
            require_mention=_boolean(values, "SUUCLAW_FEISHU_REQUIRE_MENTION", True),
            respond_to_mention_all=_boolean(values, "SUUCLAW_FEISHU_RESPOND_TO_MENTION_ALL", False),
            allow_from=_csv(values, "SUUCLAW_FEISHU_ALLOW_FROM"),
            deny_from=_csv(values, "SUUCLAW_FEISHU_DENY_FROM"),
            group_allowlist=_csv(values, "SUUCLAW_FEISHU_GROUP_ALLOWLIST"),
            group_blocklist=_csv(values, "SUUCLAW_FEISHU_GROUP_BLOCKLIST"),
            admins=_csv(values, "SUUCLAW_FEISHU_ADMINS"),
            dedup_ttl_seconds=_integer(values, "SUUCLAW_FEISHU_DEDUP_TTL", 12 * 3600),
            stale_window_ms=_integer(values, "SUUCLAW_FEISHU_STALE_WINDOW_MS", 30 * 60 * 1000),
            text_batch_delay_ms=_integer(values, "SUUCLAW_FEISHU_BATCH_DELAY_MS", 600, minimum=0),
            text_chunk_limit=_integer(values, "SUUCLAW_FEISHU_TEXT_CHUNK_LIMIT", 3500),
            outbound_attempts=_integer(values, "SUUCLAW_FEISHU_OUTBOUND_ATTEMPTS", 3),
        )
