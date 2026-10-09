"""Run coordination, browser approvals, and the existing Agent stream adapter."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..core.approval import ApprovalDecision, ApprovalHandler, ApprovalRequest
from .store import DashboardError, DashboardStore, _hidden_or_sensitive, utcnow

EventRunner = Callable[[str, Path, ApprovalHandler], Iterable[dict[str, Any]]]
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|password|passwd|secret|access[_-]?token|refresh[_-]?token|authorization)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?im)\b([A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|ACCESS_TOKEN|PRIVATE_KEY)[A-Z0-9_]*)"
    r"\s*[:=]\s*([^\s,;]{8,})"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")


class Redactor:
    """Never serialize runtime objects or configured credentials into browser data."""

    def __init__(self) -> None:
        self.secrets = tuple(
            sorted(
                {
                    value
                    for key, value in os.environ.items()
                    if _SECRET_KEY.search(key) and len(value) >= 8
                },
                key=len,
                reverse=True,
            )
        )

    def text(self, value: str) -> str:
        for secret in self.secrets:
            value = value.replace(secret, "[redacted]")
        value = _ASSIGNMENT.sub(r"\1=[redacted]", value)
        value = _BEARER.sub("Bearer [redacted]", value)
        return value

    def value(self, value: Any, depth: int = 0) -> Any:
        if depth > 10:
            return "[nested data omitted]"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            result = self.text(value)
            return result if len(result) <= 32_000 else result[:32_000] + "\n[truncated]"
        if isinstance(value, Path):
            return self.text(str(value))
        if isinstance(value, Mapping):
            result = {}
            for key, item in list(value.items())[:200]:
                label = str(key)
                if _SECRET_KEY.search(label) or label.lower() in {"env", "runtime"}:
                    result[label] = "[redacted]"
                else:
                    result[label] = self.value(item, depth + 1)
            return result
        if isinstance(value, (list, tuple)):
            return [self.value(item, depth + 1) for item in value[:200]]
        # LangChain messages in node updates contain useful content, while
        # generic repr()/model_dump() can expose credentials held by a runtime.
        if hasattr(value, "content") and hasattr(value, "type"):
            return self.value({"type": value.type, "content": value.content}, depth + 1)
        return f"<{type(value).__name__}>"

    def contains_secret(self, content: bytes) -> bool:
        if any(secret.encode("utf-8") in content for secret in self.secrets):
            return True
        return bool(re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", content))


def agent_runner(model_name: str | None = None) -> EventRunner:
    """Use the production session/graph engine; constructing this makes no API call."""

    def run(message: str, workspace: Path, approval: ApprovalHandler) -> Iterable[dict[str, Any]]:
        from ..core.agent import stream_session_events
        from ..core.state import RuntimeState
        from ..providers.openai_provider import create_model

        state = RuntimeState.create(
            workspace,
            model=model_name,
            approval_mode="inline",
            approval_handler=approval,
            checkpoint_mode="light",
            trace_mode="light",
        )
        model = create_model(model_name)
        yield from stream_session_events(
            message,
            session_workspace=workspace,
            state=state,
            model=model,
        )

    return run


@dataclass
class _PendingApproval:
    request: ApprovalRequest
    ready: threading.Event = field(default_factory=threading.Event)
    decision: ApprovalDecision | None = None


class DashboardService:
    """Own the local workers; HTTP connections merely subscribe to durable events.

    A custom ``runner`` is a test seam, not a user-selectable simulated mode.
    It takes ``(message, session_workspace, approval_handler)`` and yields the
    production event catalogue. Tests can exercise every transport offline.
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        model: str | None = None,
        runner: EventRunner | None = None,
        configured: bool | None = None,
        approval_timeout: float = 300.0,
        max_concurrent_runs: int = 4,
    ) -> None:
        from ..providers.openai_provider import DEFAULT_MODEL, load_environment

        if runner is None:
            load_environment()
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.configured = (
            bool(configured)
            if configured is not None
            else runner is not None or bool(os.getenv("OPENAI_API_KEY", "").strip())
        )
        self.redactor = Redactor()
        self.store = DashboardStore(workspace)
        self.store.recover_interrupted()
        self.runner = runner or agent_runner(self.model)
        self.approval_timeout = max(0.01, float(approval_timeout))
        self.max_concurrent_runs = max(1, int(max_concurrent_runs))
        self.changed = threading.Condition(threading.RLock())
        self.stopped = threading.Event()
        self._workers: dict[str, threading.Thread] = {}
        self._approvals: dict[tuple[str, str], _PendingApproval] = {}
        self._started: dict[str, float] = {}
        self._private_calls: dict[str, set[str]] = {}

    def bootstrap(self) -> dict[str, Any]:
        from ..core.state import RuntimeState
        from ..graph.topology import describe_workflow
        from .catalog import describe_workflow_tools

        # Tool descriptions need policy only. No model, tool, or graph executes.
        state = RuntimeState.create(self.store.workspace, approval_mode="deny")
        return {
            "brand": "SuuClaw",
            "workspace": str(self.store.workspace),
            "session_storage": str(self.store.sessions_root),
            "model": {"name": self.model, "configured": self.configured},
            "configured": self.configured,
            "configuration_message": (
                "Model is configured."
                if self.configured
                else "Set OPENAI_API_KEY in your environment or .env, then restart the dashboard."
            ),
            "tools": describe_workflow_tools(state),
            "topology": describe_workflow(),
            "approval": {"mode": "inline", "timeout_seconds": self.approval_timeout},
        }

    def start_run(self, session_id: str, message: str) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise DashboardError("invalid_message", "Enter a non-empty message.")
        if len(message) > 16_000:
            raise DashboardError(
                "message_too_long", "Messages are limited to 16,000 characters.", 413
            )
        self.store.get_session(session_id)
        if not self.configured:
            raise DashboardError(
                "model_not_configured",
                "OPENAI_API_KEY is not configured. Set it in your environment or .env, "
                "then restart the dashboard.",
                503,
            )
        with self.changed:
            if self.stopped.is_set():
                raise DashboardError("shutting_down", "The dashboard is shutting down.", 503)
            if len(self.store.active_runs()) >= self.max_concurrent_runs:
                raise DashboardError("server_busy", "All run slots are busy. Try again later.", 429)
            run = self.store.create_run(session_id, self.redactor.text(message.strip()))
            self._started[run["id"]] = time.monotonic()
            worker = threading.Thread(
                target=self._run,
                args=(run["id"], message.strip()),
                name=f"suuclaw-run-{run['id'][:8]}",
                daemon=True,
            )
            self._workers[run["id"]] = worker
            worker.start()
            return run

    def _emit(self, run_id: str, event: Mapping[str, Any]) -> None:
        elapsed_ms = int((time.monotonic() - self._started.get(run_id, time.monotonic())) * 1000)
        # Dashboard-owned envelope fields always win over engine payload fields.
        payload = self.redactor.value(dict(event))
        # Private-file tool results must not become a second artifact download
        # path through SSE. Preserve the operation identity, omit its contents.
        call_id = str(event.get("id") or "")
        if event.get("type") == "tool_call" and isinstance(event.get("args"), Mapping):
            arguments = event["args"]
            raw_path = str(arguments.get("file_path") or arguments.get("path") or "")
            if raw_path and _hidden_or_sensitive(tuple(raw_path.replace("\\", "/").split("/"))):
                self._private_calls.setdefault(run_id, set()).add(call_id)
                payload["args"] = {"path": raw_path, "content": "[private file content omitted]"}
        elif event.get("type") == "tool_result":
            private_calls = self._private_calls.get(run_id, set())
            if call_id in private_calls:
                private_calls.discard(call_id)
                payload["result"] = "[private file content omitted]"
        payload["elapsed_ms"] = max(0, elapsed_ms)
        self.store.append_event(run_id, payload)
        with self.changed:
            self.changed.notify_all()

    def _run(self, run_id: str, message: str) -> None:
        run = self.store.get_run(run_id)
        final: dict[str, Any] | None = None
        status, answer, error = "failed", "", None
        try:
            self._emit(run_id, {"type": "run_start", "status": "running", "model": self.model})
            workspace = self.store.session_workspace(run["session_id"])
            for event in self.runner(message, workspace, self._approval_handler(run_id)):
                if self.stopped.is_set():
                    break
                if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
                    raise RuntimeError("Agent returned an invalid event.")
                self._emit(run_id, event)
                if event.get("type") == "final_answer":
                    final = dict(event)
            if self.stopped.is_set():
                status, error = "interrupted", "The dashboard stopped before this run finished."
            elif final is None:
                error = "Agent stream ended without a final answer."
            else:
                answer = self.redactor.text(str(final.get("content") or ""))
                error = self.redactor.text(str(final.get("error") or "")) or None
                if final.get("interrupted"):
                    status = "interrupted"
                elif error or final.get("passed") is False or final.get("exhausted"):
                    status = "failed"
                    error = error or "The workflow did not pass verification."
                else:
                    status = "completed"
        except Exception as exc:  # noqa: BLE001 - propagate failures to the event stream
            error = self.redactor.text(f"{type(exc).__name__}: {exc}")
            self._emit(run_id, {"type": "error", "message": error, "node": ""})
        finally:
            # Serialize terminal events with shutdown; neither path can finish
            # a run twice, and a late worker cannot change an interrupted run.
            with self.changed:
                if self.store.get_run(run_id)["status"] == "running":
                    self._emit(run_id, {"type": "run_end", "status": status, "error": error})
                    self.store.finish_run(run_id, status, answer=answer, error=error)
                self._workers.pop(run_id, None)
                self._started.pop(run_id, None)
                self._private_calls.pop(run_id, None)
                self.changed.notify_all()

    def _approval_handler(self, run_id: str) -> ApprovalHandler:
        def approve(request: ApprovalRequest) -> ApprovalDecision:
            if self.stopped.is_set():
                return ApprovalDecision(False, "Dashboard is shutting down.")
            pending = _PendingApproval(request)
            key = (run_id, request.id)
            with self.changed:
                # Shutdown may win between the initial check and registration.
                if self.stopped.is_set():
                    return ApprovalDecision(False, "Dashboard is shutting down.")
                self._approvals[key] = pending
            self._emit(
                run_id,
                {
                    "type": "approval_requested",
                    "approval_id": request.id,
                    "command": request.command,
                    "risk_reason": request.risk_reason,
                    "tool_name": request.tool_name,
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=self.approval_timeout)
                    ).isoformat(timespec="milliseconds"),
                },
            )
            pending.ready.wait(self.approval_timeout)
            with self.changed:
                decision = pending.decision or ApprovalDecision(
                    False, "Approval timed out; the command was not approved."
                )
                self._approvals.pop(key, None)
            self._emit(
                run_id,
                {
                    "type": "approval_resolved",
                    "approval_id": request.id,
                    "approved": decision.approved,
                    "reason": decision.reason,
                },
            )
            return decision

        return approve

    def resolve_approval(self, run_id: str, approval_id: str, approved: bool) -> dict[str, Any]:
        self.store.get_run(run_id)
        if not isinstance(approved, bool):
            raise DashboardError("invalid_approval", "approved must be true or false.")
        with self.changed:
            pending = self._approvals.get((run_id, approval_id))
            if pending is None or pending.decision is not None or self.stopped.is_set():
                raise DashboardError("approval_closed", "This approval is no longer pending.", 409)
            pending.decision = ApprovalDecision(
                approved, "Approved in the dashboard." if approved else "Declined in the dashboard."
            )
            pending.ready.set()
            return {"approval_id": approval_id, "approved": approved, "at": utcnow()}

    def read_file(self, session_id: str, path: str, *, download: bool = False) -> Any:
        result = self.store.read_file(session_id, path, download=download)
        if download:
            _, data = result
            if self.redactor.contains_secret(data):
                raise DashboardError(
                    "sensitive_file", "This file contains private credentials.", 403
                )
        elif result.get("content"):
            if self.redactor.contains_secret(result["content"].encode("utf-8")):
                raise DashboardError(
                    "sensitive_file", "This file contains private credentials.", 403
                )
            result["content"] = self.redactor.text(result["content"])
        return result

    def shutdown(self) -> None:
        with self.changed:
            self.stopped.set()
            for pending in self._approvals.values():
                pending.decision = ApprovalDecision(False, "Dashboard is shutting down.")
                pending.ready.set()
            for run in self.store.active_runs():
                message = "The dashboard stopped before this run finished."
                self._emit(
                    run["id"], {"type": "run_end", "status": "interrupted", "error": message}
                )
                self.store.finish_run(run["id"], "interrupted", error=message)
            self.changed.notify_all()
