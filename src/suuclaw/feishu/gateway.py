"""Feishu transport adapter for SuuClaw's session-aware agent driver."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.agent import stream_session_events
from ..core.paths import runtime_path
from ..core.session import load_or_create_session, session_path
from ..core.state import RuntimeState
from .artifacts import ArtifactTracker, build_artifact_archive
from .config import FeishuGatewayConfig
from .isolation import (
    activate_conversation_generation,
    active_conversation_generation,
    active_conversation_root,
    new_conversation_generation,
    prepare_conversation_workspace,
)
from .render import WorkflowPanel

__all__ = [
    "AgentTurnResult",
    "FeishuGateway",
    "SuuClawAgentRunner",
    "build_lark_channel",
    "create_gateway",
]

logger = logging.getLogger(__name__)
_PROGRESS_REPLACEMENT_SECONDS = 30.0
_NEW_SESSION_COMMANDS = {
    "/new",
    "/newsession",
    "new session",
    "开启新对话",
    "开启新会话",
    "新对话",
    "新会话",
}
_STATUS_COMMANDS = {"/status", "当前对话", "会话状态"}


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """The small result shape the transport needs from a complete agent turn."""

    content: str
    passed: bool
    route: str = ""
    error: str = ""
    elapsed: float = 0.0
    artifacts: tuple[Path, ...] = ()
    artifact_root: Path | None = None
    artifact_failures: tuple[str, ...] = ()
    artifact_omitted: int = 0


class SuuClawAgentRunner:
    """Synchronous adapter around ``stream_session_events`` for ``to_thread``."""

    def __init__(self, config: FeishuGatewayConfig, model: Any) -> None:
        self.config = config
        self.model = model

    def __call__(self, task: str, session_workspace: Path) -> AgentTurnResult:
        return self.run_with_events(task, session_workspace)

    def run_with_events(
        self,
        task: str,
        session_workspace: Path,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentTurnResult:
        """Run one turn and optionally expose its existing event stream.

        The callback stays synchronous because this method runs in
        ``asyncio.to_thread``.  The gateway bridges it back to its event loop;
        the core workflow remains transport-agnostic.
        """
        started = time.monotonic()
        state = RuntimeState.create(
            session_workspace,
            command_timeout=self.config.command_timeout,
            approval_mode=self.config.approval_mode,
            approval_handler=None,
            checkpoint_mode=self.config.checkpoint_mode,
            trace_mode=self.config.trace_mode,
            transport="feishu",
        )
        final: dict[str, Any] | None = None
        for event in stream_session_events(
            task,
            session_workspace=session_workspace,
            state=state,
            model=self.model,
            max_attempts=self.config.max_attempts,
            max_loops=self.config.max_loops,
            retries=self.config.retries,
            entry_timeout=self.config.entry_timeout,
        ):
            if on_event is not None:
                try:
                    on_event(dict(event))
                except Exception:
                    # Progress reporting is best-effort.  A closing transport
                    # must never abort file work that is already in flight.
                    logger.debug("Feishu progress callback failed", exc_info=True)
            if str(event.get("type") or "") == "final_answer":
                final = event

        elapsed = time.monotonic() - started
        if final is None:
            raise RuntimeError("agent stream ended without a final_answer event")
        return AgentTurnResult(
            content=str(final.get("content") or "").strip(),
            passed=bool(final.get("passed")),
            route=str(final.get("route") or ""),
            error=str(final.get("error") or "").strip(),
            elapsed=elapsed,
        )


def _optional_list(values: tuple[str, ...]) -> list[str] | None:
    return list(values) if values else None


def _configure_lark_ws_proxy(mode: str, open_timeout: float) -> None:
    """Bridge the SDK's forced-direct WS behavior to websockets 15 proxy support.

    ``lark-oapi==1.7.3`` passes ``proxy=None`` unconditionally even though its
    endpoint discovery HTTP request honors proxy environment variables.  That
    creates a split route where discovery works and the WSS handshake times
    out.  Keep this compatibility shim in our process rather than patching the
    installed package.  An empty/``auto`` mode restores websockets' own proxy
    selection (including ``NO_PROXY``); ``direct`` preserves SDK behavior; any
    other value is treated as an explicit proxy URL.
    """
    from lark_oapi.ws import client as ws_client_module

    if "proxy" not in inspect.signature(ws_client_module.websockets.connect).parameters:
        return
    normalized = str(mode or "auto").strip()
    if normalized.lower() in {"auto", "env"}:
        ws_client_module._ws_connect_kwargs = lambda: {"open_timeout": open_timeout}
        logger.info("Feishu WebSocket proxy mode: environment auto-detection")
    elif normalized.lower() in {"direct", "none", "off"}:
        ws_client_module._ws_connect_kwargs = lambda: {
            "proxy": None,
            "open_timeout": open_timeout,
        }
        logger.info("Feishu WebSocket proxy mode: direct")
    else:
        ws_client_module._ws_connect_kwargs = lambda: {
            "proxy": normalized,
            "open_timeout": open_timeout,
        }
        logger.info("Feishu WebSocket proxy mode: explicit")


def build_lark_channel(config: FeishuGatewayConfig) -> Any:
    """Build the SDK facade lazily so the core package keeps Feishu optional."""
    try:
        from lark_oapi.channel import (
            ChatQueueConfig,
            DedupConfig,
            FeishuChannel,
            OutboundConfig,
            PolicyConfig,
            RetryConfig,
            SafetyConfig,
            TextBatchConfig,
        )
        from lark_oapi.core.enum import LogLevel
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            'Feishu support is not installed. Run: pip install -e ".[feishu]"'
        ) from exc

    _configure_lark_ws_proxy(config.ws_proxy, config.ws_open_timeout)

    policy = PolicyConfig(
        dm_policy=config.dm_policy,
        group_policy=config.group_policy,
        require_mention=config.require_mention,
        respond_to_mention_all=config.respond_to_mention_all,
        allow_from=_optional_list(config.allow_from),
        deny_from=_optional_list(config.deny_from),
        group_allowlist=_optional_list(config.group_allowlist),
        group_blocklist=_optional_list(config.group_blocklist),
        admins=_optional_list(config.admins),
    )
    safety = SafetyConfig(
        dedup=DedupConfig(ttl_seconds=config.dedup_ttl_seconds),
        text_batch=TextBatchConfig(delay_ms=config.text_batch_delay_ms),
        chat_queue=ChatQueueConfig(enabled=True),
        stale_message_window_ms=config.stale_window_ms,
    )
    outbound = OutboundConfig(
        text_chunk_limit=config.text_chunk_limit,
        retry=RetryConfig(max_attempts=config.outbound_attempts),
    )
    return FeishuChannel(
        app_id=config.app_id,
        app_secret=config.app_secret,
        encrypt_key=config.encrypt_key or None,
        verification_token=config.verification_token or None,
        # The SDK logs its complete WSS URL (including a temporary access_key
        # and ticket) at INFO.  Keep vendor logs at WARNING and emit our own
        # credential-free readiness/reconnect messages instead.
        log_level=LogLevel.WARNING,
        transport="ws",
        policy=policy,
        safety=safety,
        outbound=outbound,
    )


class FeishuGateway:
    """Route normalized Feishu messages into isolated SuuClaw sessions."""

    def __init__(
        self,
        config: FeishuGatewayConfig,
        channel: Any,
        runner: Callable[[str, Path], AgentTurnResult],
    ) -> None:
        self.config = config
        self.channel = channel
        self.runner = runner
        self._slots = asyncio.Semaphore(config.max_concurrency)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._registered = False

    @staticmethod
    def conversation_key(message: Any) -> str:
        chat_id = str(getattr(message, "chat_id", "") or "").strip()
        conversation = getattr(message, "conversation", None)
        thread_id = str(getattr(conversation, "thread_id", "") or "").strip()
        if not chat_id:
            chat_id = str(getattr(message, "sender_id", "") or "unknown").strip()
        return f"{chat_id}:{thread_id or 'root'}"

    def conversation_namespace(self, message: Any) -> Path:
        """Return the stable storage namespace for one Feishu chat/thread."""
        key = self.conversation_key(message)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.config.session_root / f"conversation-{digest}"

    def conversation_root(self, message: Any) -> Path:
        """Return the currently active session generation for a chat/thread."""
        return active_conversation_root(self.conversation_namespace(message))

    def session_workspace(self, message: Any) -> Path:
        """Return the actual isolated Agent workspace for this conversation."""
        return self.conversation_root(message) / "workdir"

    def prepare_session_workspace(self, message: Any) -> Path:
        """Seed an isolated workdir and copy legacy session history once."""
        namespace = self.conversation_namespace(message)
        root = self.conversation_root(message)
        legacy = None
        if root == namespace:
            legacy = runtime_path(self.config.workspace, "feishu", "sessions", root.name)
        return prepare_conversation_workspace(
            self.config.workspace,
            root / "workdir",
            legacy_workspace=legacy,
        )

    def start_new_session(self, message: Any) -> tuple[str, str, Path]:
        """Create and activate a new persistent session without deleting the old one."""
        namespace = self.conversation_namespace(message)
        generation, root = new_conversation_generation(namespace)
        workspace = prepare_conversation_workspace(self.config.workspace, root / "workdir")
        session = load_or_create_session(workspace)
        if not session_path(workspace).is_file():
            raise RuntimeError("new session metadata could not be saved")
        activate_conversation_generation(namespace, generation)
        return generation, str(session.get("session_id") or ""), workspace

    def session_status(self, message: Any) -> tuple[str, str, int, Path]:
        """Read the selected generation and core session metadata without creating it."""
        namespace = self.conversation_namespace(message)
        generation = active_conversation_generation(namespace)
        workspace = active_conversation_root(namespace) / "workdir"
        session_id = ""
        turn_index = 0
        try:
            payload = json.loads(session_path(workspace).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                session_id = str(payload.get("session_id") or "")
                turn_index = max(0, int(payload.get("turn_index") or 0))
        except (OSError, TypeError, ValueError):
            pass
        return generation, session_id, turn_index, workspace

    @staticmethod
    def _command(task: str) -> str:
        normalized = " ".join(str(task or "").strip().lower().split()).rstrip("。！？!")
        if normalized in _NEW_SESSION_COMMANDS:
            return "new"
        if normalized in _STATUS_COMMANDS:
            return "status"
        return ""

    def register_handlers(self) -> None:
        if self._registered:
            return
        self.channel.on("message", self.handle_message)
        self.channel.on("error", self.handle_channel_error)
        self.channel.on("reject", self.handle_reject)
        self.channel.on("reconnecting", self.handle_reconnecting)
        self.channel.on("reconnected", self.handle_reconnected)
        self._registered = True

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Connect, serve until cancellation/stop, and always disconnect cleanly."""
        self.register_handlers()
        waiter = stop_event if stop_event is not None else asyncio.Event()
        try:
            await self.channel.connect_until_ready(timeout=self.config.connect_timeout)
            logger.info(
                "Feishu gateway ready (workspace=%s, concurrency=%s, approval=%s)",
                self.config.workspace,
                self.config.max_concurrency,
                self.config.approval_mode,
            )
            await waiter.wait()
        finally:
            await self.channel.disconnect()

    async def handle_message(self, message: Any) -> None:
        message_id = str(getattr(message, "message_id", "") or "").strip()
        chat_id = str(getattr(message, "chat_id", "") or "").strip()
        if not chat_id:
            logger.warning("dropping Feishu message without chat_id (message=%s)", message_id)
            return

        kind = str(
            getattr(message, "raw_content_type", "")
            or getattr(getattr(message, "content", None), "kind", "")
            or ""
        ).lower()
        task = str(getattr(message, "content_text", "") or "").strip()
        if kind not in {"text", "post"} or not task:
            await self._send_reply(
                chat_id, message_id, self.config.unsupported_text, markdown=False
            )
            return

        key = self.conversation_key(message)
        lock = self._session_locks.setdefault(key, asyncio.Lock())
        async with lock:
            command = self._command(task)
            if command == "new":
                try:
                    generation, session_id, workspace = await asyncio.to_thread(
                        self.start_new_session, message
                    )
                except Exception:
                    logger.exception(
                        "could not start a new Feishu session (chat=%s)",
                        _short_id(chat_id),
                    )
                    await self._send_reply(
                        chat_id,
                        message_id,
                        "新对话创建失败，请稍后重试。",
                        markdown=False,
                    )
                    return
                logger.info(
                    "new Feishu session activated (chat=%s, generation=%s, session=%s)",
                    _short_id(chat_id),
                    generation[:8],
                    _short_id(session_id),
                )
                await self._send_reply(
                    chat_id,
                    message_id,
                    "\n".join(
                        [
                            "✅ **已开启新对话**",
                            f"- 会话：`{session_id[:8] or generation[:8]}`",
                            f"- 工作区：`{workspace}`",
                            "- 旧对话、文件和产物均已保留。",
                            "",
                            "请直接发送你的新任务。",
                        ]
                    ),
                    markdown=True,
                )
                return

            if command == "status":
                generation, session_id, turn_index, workspace = await asyncio.to_thread(
                    self.session_status, message
                )
                await self._send_reply(
                    chat_id,
                    message_id,
                    "\n".join(
                        [
                            "📍 **当前对话状态**",
                            f"- 会话：`{session_id[:8] or '(尚未建立)'}`",
                            f"- 代次：`{generation[:8] if generation else '初始会话'}`",
                            f"- 已处理任务：{turn_index}",
                            f"- 工作区：`{workspace}`",
                        ]
                    ),
                    markdown=True,
                )
                return

            async with self._slots:
                if self.config.processing_text:
                    await self._send_reply(
                        chat_id,
                        message_id,
                        self.config.processing_text,
                        markdown=False,
                        required=False,
                    )
                logger.info(
                    "agent turn started (chat=%s, message=%s)",
                    _short_id(chat_id),
                    _short_id(message_id),
                )
                try:
                    conversation_workspace = await asyncio.to_thread(
                        self.prepare_session_workspace, message
                    )
                    result = await self._run_agent_turn(
                        task,
                        conversation_workspace,
                        chat_id=chat_id,
                        message_id=message_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "agent turn failed (chat=%s, message=%s)",
                        _short_id(chat_id),
                        _short_id(message_id),
                    )
                    await self._send_reply(
                        chat_id, message_id, self.config.error_text, markdown=False
                    )
                    return

                answer = result.content
                if not answer:
                    answer = self.config.error_text
                    if result.error:
                        logger.error(
                            "agent returned no content (chat=%s, error=%s)",
                            _short_id(chat_id),
                            result.error,
                        )
                await self._send_reply(chat_id, message_id, answer, markdown=True)
                if self.config.deliver_artifacts and (
                    result.artifacts or result.artifact_failures or result.artifact_omitted
                ):
                    await self._deliver_artifacts(
                        chat_id,
                        message_id,
                        result.artifacts,
                        workspace=result.artifact_root or conversation_workspace,
                        initial_failures=result.artifact_failures,
                        omitted=result.artifact_omitted,
                    )
                logger.info(
                    "agent turn finished (chat=%s, passed=%s, route=%s, elapsed=%.2fs)",
                    _short_id(chat_id),
                    result.passed,
                    result.route or "unknown",
                    result.elapsed,
                )

    async def _run_agent_turn(
        self,
        task: str,
        session_workspace: Path,
        *,
        chat_id: str,
        message_id: str,
    ) -> AgentTurnResult:
        """Run the agent and relay selected workflow milestones to Feishu."""
        streaming_runner = getattr(self.runner, "run_with_events", None)
        if not callable(streaming_runner):
            return await asyncio.to_thread(self.runner, task, session_workspace)

        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def publish(event: dict[str, Any] | None) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        def invoke() -> AgentTurnResult:
            try:
                return streaming_runner(task, session_workspace, publish)
            finally:
                # Queue the sentinel from the same thread as every event. This
                # preserves call_soon_threadsafe FIFO order and prevents the
                # consumer from seeing completion before pending milestones.
                publish(None)

        agent_task = asyncio.create_task(asyncio.to_thread(invoke))
        progress = WorkflowPanel(task, self.config.workflow_heartbeat_seconds)
        artifacts = ArtifactTracker(session_workspace)
        progress_message_id = ""
        last_progress_replacement = 0.0
        latest_panel = ""
        events_exhausted = False

        async def consume_events(stream_controller: Any | None = None) -> None:
            nonlocal events_exhausted
            nonlocal last_progress_replacement
            nonlocal latest_panel
            nonlocal progress_message_id

            while True:
                event = await events.get()
                if event is None:
                    events_exhausted = True
                    break
                artifacts.observe(event)
                panel = progress.render_for(event)
                if not panel or not self.config.workflow_progress:
                    continue
                latest_panel = panel
                if stream_controller is not None:
                    await stream_controller.set_content(panel)
                    continue

                now = time.monotonic()
                old_message_id = progress_message_id
                allow_replacement = (
                    not progress_message_id
                    or str(event.get("type") or "") == "final_answer"
                    or now - last_progress_replacement >= _PROGRESS_REPLACEMENT_SECONDS
                )
                progress_message_id, delivered = await self._upsert_progress_panel(
                    chat_id,
                    message_id,
                    progress_message_id,
                    panel,
                    allow_replacement=allow_replacement,
                )
                if delivered and progress_message_id != old_message_id:
                    last_progress_replacement = now

        stream = getattr(self.channel, "stream", None)
        if self.config.workflow_progress and callable(stream):
            opts: dict[str, Any] = {"reply_to": message_id} if message_id else {}
            if message_id and self.config.reply_in_thread:
                opts["reply_in_thread"] = True

            async def produce_progress(controller: Any) -> None:
                await consume_events(controller)

            try:
                stream_result = await stream(
                    chat_id,
                    {"markdown": produce_progress},
                    opts or None,
                )
                if stream_result is not None and getattr(stream_result, "success", True) is False:
                    raise RuntimeError(f"stream rejected: {getattr(stream_result, 'error', None)}")
                if latest_panel:
                    logger.info(
                        "Feishu native progress stream completed (chat=%s, message=%s)",
                        _short_id(chat_id),
                        _short_id(str(getattr(stream_result, "message_id", "") or "")),
                    )
            except Exception:
                logger.warning(
                    "Feishu native progress stream failed; using message fallback (chat=%s)",
                    _short_id(chat_id),
                    exc_info=True,
                )
                if latest_panel:
                    progress_message_id, _ = await self._upsert_progress_panel(
                        chat_id,
                        message_id,
                        "",
                        latest_panel,
                        allow_replacement=True,
                    )
                    last_progress_replacement = time.monotonic()
                if not events_exhausted:
                    await consume_events()
        else:
            await consume_events()
        result = await agent_task
        max_bytes = int(self.config.artifact_max_megabytes * 1024 * 1024)
        turn_digest = hashlib.sha256((message_id or task).encode("utf-8")).hexdigest()[:24]
        snapshot = artifacts.snapshot(
            session_workspace.parent / "artifacts" / turn_digest,
            max_files=self.config.artifact_max_files,
            max_bytes=max_bytes,
        )
        return AgentTurnResult(
            content=result.content,
            passed=result.passed,
            route=result.route,
            error=result.error,
            elapsed=result.elapsed,
            artifacts=snapshot.paths,
            artifact_root=snapshot.root,
            artifact_failures=snapshot.failures,
            artifact_omitted=snapshot.omitted,
        )

    async def _deliver_artifacts(
        self,
        chat_id: str,
        reply_to: str,
        artifacts: tuple[Path, ...],
        *,
        workspace: Path | None = None,
        initial_failures: tuple[str, ...] = (),
        omitted: int = 0,
    ) -> None:
        """Upload bounded, workspace-confined files as Feishu attachments."""
        limit = self.config.artifact_max_files
        max_bytes = int(self.config.artifact_max_megabytes * 1024 * 1024)
        failures = list(initial_failures)
        delivered = 0
        selected: list[Path] = []

        for path in artifacts[:limit]:
            try:
                size = path.stat().st_size
            except OSError:
                failures.append(path.name)
                continue
            if size <= 0 or size > max_bytes:
                failures.append(path.name)
                continue
            selected.append(path)
            if not await self._send_artifact_file(
                chat_id, reply_to, source=str(path), file_name=path.name
            ):
                failures.append(path.name)
                continue
            delivered += 1

        if len(selected) > 1 and any(path.suffix.lower() in {".htm", ".html"} for path in selected):
            archive_name = "suuclaw-web-artifacts.zip"
            archive = build_artifact_archive(
                tuple(selected),
                workspace=workspace or self.config.workspace,
                max_bytes=max_bytes,
            )
            if archive is None or not await self._send_artifact_file(
                chat_id,
                reply_to,
                source=archive,
                file_name=archive_name,
            ):
                failures.append(archive_name)
            else:
                delivered += 1

        omitted += max(0, len(artifacts) - limit)
        if omitted:
            failures.append(f"另有 {omitted} 个文件超过数量上限")
        if failures:
            names = "、".join(failures[:5])
            suffix = "等" if len(failures) > 5 else ""
            await self._send_reply(
                chat_id,
                reply_to,
                f"有些生成文件未自动发送：{names}{suffix}。已成功发送 {delivered} 个文件。",
                markdown=False,
                required=False,
            )

    async def _send_artifact_file(
        self,
        chat_id: str,
        reply_to: str,
        *,
        source: str | bytes,
        file_name: str,
    ) -> bool:
        opts: dict[str, Any] = {"reply_to": reply_to} if reply_to else {}
        if reply_to and self.config.reply_in_thread:
            opts["reply_in_thread"] = True
        payload = {"file": {"source": source, "file_name": file_name}}
        try:
            result = await self.channel.send(chat_id, payload, opts or None)
        except Exception:
            logger.warning(
                "Feishu artifact upload failed (chat=%s, file=%s)",
                _short_id(chat_id),
                file_name,
                exc_info=True,
            )
            return False
        if result is not None and getattr(result, "success", True) is False:
            logger.warning(
                "Feishu artifact upload rejected (chat=%s, file=%s, error=%s)",
                _short_id(chat_id),
                file_name,
                getattr(result, "error", None),
            )
            return False
        return True

    async def _upsert_progress_panel(
        self,
        chat_id: str,
        reply_to: str,
        progress_message_id: str,
        markdown: str,
        *,
        allow_replacement: bool,
    ) -> tuple[str, bool]:
        """Create or update one Markdown progress panel without message spam."""
        previous_message_id = progress_message_id
        if progress_message_id:
            edit = getattr(self.channel, "edit_message", None)
            if callable(edit):
                try:
                    result = await edit(progress_message_id, {"markdown": markdown})
                    if result is None or getattr(result, "success", True):
                        return progress_message_id, True
                    logger.warning(
                        "Feishu progress edit rejected (message=%s, error=%s)",
                        _short_id(progress_message_id),
                        getattr(result, "error", None),
                    )
                except Exception:
                    logger.warning(
                        "Feishu progress edit failed (message=%s)",
                        _short_id(progress_message_id),
                        exc_info=True,
                    )
                if not allow_replacement:
                    return progress_message_id, False
                progress_message_id = ""

        opts: dict[str, Any] = {"reply_to": reply_to} if reply_to else {}
        if reply_to and self.config.reply_in_thread:
            opts["reply_in_thread"] = True
        try:
            result = await self.channel.send(chat_id, {"markdown": markdown}, opts or None)
        except Exception:
            logger.warning("Feishu progress panel failed (chat=%s)", _short_id(chat_id))
            return previous_message_id, False
        if result is not None and getattr(result, "success", True) is False:
            logger.warning(
                "Feishu progress panel rejected (chat=%s, error=%s)",
                _short_id(chat_id),
                getattr(result, "error", None),
            )
            return previous_message_id, False
        message_id = str(getattr(result, "message_id", "") or "")
        return message_id or previous_message_id, bool(message_id)

    async def _send_reply(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        markdown: bool,
        required: bool = True,
    ) -> bool:
        payload = {"markdown" if markdown else "text": text}
        opts: dict[str, Any] = {}
        if message_id:
            opts["reply_to"] = message_id
            if self.config.reply_in_thread:
                opts["reply_in_thread"] = True
        try:
            result = await self.channel.send(chat_id, payload, opts or None)
        except Exception:
            if required:
                logger.exception("Feishu reply raised (chat=%s)", _short_id(chat_id))
            else:
                logger.warning("Feishu progress reply failed (chat=%s)", _short_id(chat_id))
            return False
        if result is not None and getattr(result, "success", True) is False:
            logger.error(
                "Feishu reply rejected (chat=%s, error=%s)",
                _short_id(chat_id),
                getattr(result, "error", None),
            )
            return False
        return True

    def handle_channel_error(self, error: Any) -> None:
        logger.error("Feishu channel error: %s", error)

    def handle_reject(self, event: Any) -> None:
        logger.info(
            "Feishu message rejected (message=%s, reason=%s)",
            _short_id(str(getattr(event, "message_id", "") or "")),
            getattr(event, "reason", "unknown"),
        )

    def handle_reconnecting(self) -> None:
        logger.warning("Feishu channel reconnecting")

    def handle_reconnected(self) -> None:
        logger.info("Feishu channel reconnected")


def _short_id(value: str) -> str:
    if len(value) <= 12:
        return value or "-"
    return f"{value[:6]}…{value[-4:]}"


def create_gateway(
    config: FeishuGatewayConfig,
    *,
    channel: Any | None = None,
    model: Any | None = None,
) -> FeishuGateway:
    """Construct the production channel and shared model, with test seams."""
    if model is None:
        from ..providers.openai_provider import create_model

        # The graph has its own visible retry policy. Disable ChatOpenAI's
        # hidden retries here so one 120s request cannot multiply into a long,
        # silent stall before the workflow event stream starts.
        model = create_model(config.model, temperature=config.temperature, max_retries=0)
    resolved_channel = channel if channel is not None else build_lark_channel(config)
    return FeishuGateway(config, resolved_channel, SuuClawAgentRunner(config, model))
