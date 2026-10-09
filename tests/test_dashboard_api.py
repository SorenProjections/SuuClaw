"""Exercise real HTTP/SSE against deterministic, offline Agent event runners."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import pytest

from suuclaw.core.approval import ApprovalRequest
from suuclaw.dashboard.server import MAX_BODY_BYTES, create_server
from suuclaw.dashboard.service import DashboardService, agent_runner


def fake_runner(message, workspace, approval):
    yield {"type": "node_start", "node": "planner"}
    yield {
        "type": "tool_call",
        "node": "code_agent",
        "id": "tool-1",
        "name": "file_write",
        "args": {"file_path": "report.md", "content": message},
    }
    (workspace / "report.md").write_text(message, encoding="utf-8")
    yield {
        "type": "tool_result",
        "node": "code_agent",
        "id": "tool-1",
        "name": "file_write",
        "ok": True,
        "result": {"output": "Created report.md"},
    }
    yield {"type": "node_output", "node": "planner", "data": {"done": True}}
    yield {"type": "final_answer", "content": "报告已生成。", "passed": True, "error": None}


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, body=data, headers=request_headers)
        response = connection.getresponse()
        raw = response.read()
        result_headers = dict(response.getheaders())
        status = response.status
        connection.close()
        if result_headers.get("Content-Type", "").startswith("application/json"):
            raw = json.loads(raw)
        return status, raw, result_headers

    def events(self, run_id, after=0, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", f"/api/runs/{run_id}/events?after={after}", headers=headers or {})
        response = connection.getresponse()
        assert response.status == 200
        raw = response.read().decode("utf-8")
        connection.close()
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]

    def session(self):
        status, body, _ = self.request("POST", "/api/sessions", {})
        assert status == 201
        return body["id"]

    def start(self, session_id, message="Write a report"):
        status, body, _ = self.request(
            "POST", f"/api/sessions/{session_id}/runs", {"message": message}
        )
        assert status == 202
        return body["id"]


@contextmanager
def running_server(tmp_path: Path, **options) -> Iterator[tuple[Client, DashboardService]]:
    options.setdefault("runner", fake_runner)
    service = DashboardService(tmp_path, **options)
    server = create_server(tmp_path, port=0, service=service)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield Client(server.server_port), service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def await_event(client: Client, run_id: str, kind: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, run, _ = client.request("GET", f"/api/runs/{run_id}")
        for event in run["events"]:
            if event["type"] == kind:
                return event
        time.sleep(0.01)
    raise AssertionError(f"Did not see {kind}")


def test_complete_http_run_files_sse_replay_and_restart(tmp_path: Path) -> None:
    with running_server(tmp_path) as (client, service):
        status, bootstrap, _ = client.request("GET", "/api/bootstrap")
        assert status == 200
        assert bootstrap["brand"] == "SuuClaw"
        assert bootstrap["configured"] is True
        assert {"bash", "todo_write", "call_code_agent", "web_search", "notepad_read"} <= {
            tool["name"] for tool in bootstrap["tools"]
        }
        assert "planner" in {node["id"] for node in bootstrap["topology"]["nodes"]}
        session = client.session()
        run = client.start(session, "hello 世界")
        events = client.events(run)
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        assert events[0]["type"] == "run_start"
        assert events[-1]["type"] == "run_end"
        assert events[-1]["status"] == "completed"
        assert all(event["run_id"] == run and event["session_id"] == session for event in events)
        assert all(event["elapsed_ms"] >= 0 and event["at"] for event in events)
        replay = client.events(run, after=2)
        assert replay == events[2:]
        # Native EventSource reconnect must honor the newest delivered sequence.
        replay = client.events(run, after=0, headers={"Last-Event-ID": "4"})
        assert replay == events[4:]
        _, detail, _ = client.request("GET", f"/api/sessions/{session}")
        assert detail["runs"][0]["status"] == "completed"
        assert detail["messages"][-1]["content"] == "报告已生成。"
        _, files, _ = client.request("GET", f"/api/sessions/{session}/files")
        assert [file["path"] for file in files["files"]] == ["report.md"]
        _, preview, _ = client.request("GET", f"/api/sessions/{session}/file?path=report.md")
        assert preview["content"] == "hello 世界"
        _, download, headers = client.request(
            "GET", f"/api/sessions/{session}/file?path=report.md&download=1"
        )
        assert download.decode() == "hello 世界"
        assert headers["Content-Disposition"].startswith("attachment;")
        assert service.store.workspace == tmp_path.resolve()
    with running_server(tmp_path) as (client, _):
        _, sessions, _ = client.request("GET", "/api/sessions")
        assert sessions["sessions"][0]["id"] == session
        assert client.events(run) == events


def test_events_reach_browser_while_worker_is_running_and_busy_is_409(tmp_path: Path) -> None:
    release = threading.Event()

    def slow_runner(message, workspace, approval):
        yield {"type": "node_start", "node": "planner"}
        release.wait(timeout=5)
        yield {"type": "final_answer", "content": "done", "passed": True}

    with running_server(tmp_path, runner=slow_runner) as (client, _):
        session = client.session()
        started = time.monotonic()
        run = client.start(session)
        assert time.monotonic() - started < 2
        connection = http.client.HTTPConnection("127.0.0.1", client.port, timeout=3)
        connection.request("GET", f"/api/runs/{run}/events")
        response = connection.getresponse()
        seen = []
        try:
            while not any(event["type"] == "node_start" for event in seen):
                line = response.readline().decode()
                if line.startswith("data: "):
                    seen.append(json.loads(line[6:]))
            status, body, _ = client.request(
                "POST", f"/api/sessions/{session}/runs", {"message": "another"}
            )
            assert status == 409 and body["error"]["code"] == "session_busy"
            _, metadata, _ = client.request("GET", f"/api/runs/{run}")
            assert metadata["status"] == "running"
        finally:
            release.set()
            response.read()
            connection.close()


def test_unconfigured_model_allows_browsing_and_does_not_start_run(tmp_path: Path) -> None:
    with running_server(tmp_path, configured=False) as (client, _):
        status, body, _ = client.request("GET", "/api/bootstrap")
        assert status == 200 and body["model"]["configured"] is False
        session = client.session()
        status, body, _ = client.request(
            "POST", f"/api/sessions/{session}/runs", {"message": "hello"}
        )
        assert status == 503 and body["error"]["code"] == "model_not_configured"
        _, detail, _ = client.request("GET", f"/api/sessions/{session}")
        assert detail["run_count"] == 0
        assert detail["messages"] == []


@pytest.mark.parametrize("kind", ["exception", "unfinished", "verification"])
def test_failures_are_terminal_and_honest(tmp_path: Path, kind: str) -> None:
    def broken_runner(message, workspace, approval):
        yield {"type": "node_start", "node": "planner"}
        if kind == "exception":
            raise RuntimeError("Provider is unavailable")
        if kind == "verification":
            yield {"type": "final_answer", "content": "Check failed", "passed": False}

    with running_server(tmp_path, runner=broken_runner) as (client, _):
        session = client.session()
        run = client.start(session)
        events = client.events(run)
        assert events[-1]["status"] == "failed"
        _, metadata, _ = client.request("GET", f"/api/runs/{run}")
        assert metadata["status"] == "failed" and metadata["error"]
        _, detail, _ = client.request("GET", f"/api/sessions/{session}")
        assert detail["active_run_id"] is None


@pytest.mark.parametrize("approved", [True, False])
def test_browser_approval_resolves_only_pending_request(tmp_path: Path, approved: bool) -> None:
    decisions = []

    def approval_runner(message, workspace, approval):
        decisions.append(approval(ApprovalRequest("pip install example", "Package installation")))
        yield {"type": "final_answer", "content": "Decision recorded", "passed": True}

    with running_server(tmp_path, runner=approval_runner) as (client, _):
        session = client.session()
        run = client.start(session)
        request = await_event(client, run, "approval_requested")
        endpoint = f"/api/runs/{run}/approvals/{request['approval_id']}"
        status, _, _ = client.request("POST", endpoint, {"approved": "yes"})
        assert status == 400
        status, _, _ = client.request("POST", endpoint, {"approved": approved})
        assert status == 200
        events = client.events(run)
        resolved = next(event for event in events if event["type"] == "approval_resolved")
        assert resolved["approved"] is approved
        assert decisions[0].approved is approved
        assert client.request("POST", endpoint, {"approved": True})[0] == 409


def test_approval_timeout_and_shutdown_deny(tmp_path: Path) -> None:
    decisions = []

    def approval_runner(message, workspace, approval):
        decisions.append(approval(ApprovalRequest("pip install example", "Package installation")))
        yield {"type": "final_answer", "content": "Decision recorded", "passed": True}

    with running_server(tmp_path, runner=approval_runner, approval_timeout=0.05) as (client, _):
        run = client.start(client.session())
        events = client.events(run)
        assert any(
            event["type"] == "approval_resolved" and not event["approved"] for event in events
        )
        assert decisions[-1].approved is False
        assert "timed out" in decisions[-1].reason
    with running_server(tmp_path, runner=approval_runner, approval_timeout=30) as (client, service):
        run = client.start(client.session())
        await_event(client, run, "approval_requested")
        service.shutdown()
        events = client.events(run)
        assert events[-1]["status"] == "interrupted"
        deadline = time.monotonic() + 2
        while len(decisions) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert decisions[-1].approved is False
        assert "shutting down" in decisions[-1].reason


def test_secret_redaction_runtime_objects_and_private_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "test-credential-not-for-browser-12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def secret_runner(message, workspace, approval):
        (workspace / "report.txt").write_text(secret, encoding="utf-8")
        (workspace / ".env").write_text("PRIVATE=value", encoding="utf-8")
        yield {
            "type": "tool_result",
            "name": "bash",
            "result": {"output": secret},
            "data": {"runtime": object(), "api_key": secret},
        }
        yield {
            "type": "tool_call",
            "name": "file_read",
            "id": "private-file",
            "args": {"file_path": ".env"},
        }
        yield {
            "type": "tool_result",
            "name": "file_read",
            "id": "private-file",
            "result": {"output": "PRIVATE=value"},
        }
        yield {"type": "final_answer", "content": f"Key: {secret}", "passed": True}

    with running_server(tmp_path, runner=secret_runner) as (client, service):
        session = client.session()
        run = client.start(session)
        assert secret not in json.dumps(client.events(run))
        _, metadata, _ = client.request("GET", f"/api/runs/{run}")
        assert secret not in json.dumps(metadata)
        assert "PRIVATE=value" not in json.dumps(metadata)
        _, bootstrap, _ = client.request("GET", "/api/bootstrap")
        assert secret not in json.dumps(bootstrap)
        assert client.request("GET", f"/api/sessions/{session}/file?path=.env")[0] == 403
        assert client.request("GET", f"/api/sessions/{session}/file?path=report.txt")[0] == 403
        assert (
            client.request("GET", f"/api/sessions/{session}/file?path=report.txt&download=1")[0]
            == 403
        )
        for path in (service.store.sessions_root / session / "runs" / run).iterdir():
            assert secret not in path.read_text(encoding="utf-8")


def test_host_origin_payload_and_traversal_guards(tmp_path: Path) -> None:
    with running_server(tmp_path) as (client, _):
        assert client.request("GET", "/api/bootstrap", headers={"Host": "evil.example"})[0] == 403
        assert (
            client.request("POST", "/api/sessions", {}, {"Origin": "https://evil.example"})[0]
            == 403
        )
        assert (
            client.request("POST", "/api/sessions", {}, {"Sec-Fetch-Site": "cross-site"})[0] == 403
        )
        assert client.request("POST", "/api/sessions", {}, {"Content-Type": "text/plain"})[0] == 415
        assert client.request("POST", "/api/sessions", {"title": "x" * MAX_BODY_BYTES})[0] == 413
        assert client.request("POST", "/api/sessions", {"title": []})[0] == 400
        session = client.session()
        bad = quote("../session.json", safe="")
        assert client.request("GET", f"/api/sessions/{session}/file?path={bad}")[0] == 403
        assert client.request("GET", "/static/../server.py")[0] == 404
        assert client.request("POST", f"/api/sessions/{session}/runs", {"message": ""})[0] == 400
        assert client.request("GET", "/api/runs/not-a-run/events?after=-1")[0] == 400


def test_public_binding_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_server(tmp_path, host="0.0.0.0")


def test_production_adapter_uses_existing_session_engine(tmp_path: Path, monkeypatch) -> None:
    import suuclaw.core.agent
    import suuclaw.providers.openai_provider

    model = object()
    calls = []
    monkeypatch.setattr(suuclaw.providers.openai_provider, "create_model", lambda name: model)

    def existing_stream(message, **kwargs):
        calls.append((message, kwargs))
        yield {"type": "final_answer", "content": "existing engine", "passed": True}

    monkeypatch.setattr(suuclaw.core.agent, "stream_session_events", existing_stream)

    def approval(request):
        return None

    events = list(agent_runner("configured-model")("task", tmp_path, approval))
    assert events[-1]["content"] == "existing engine"
    assert calls[0][0] == "task"
    assert calls[0][1]["model"] is model
    state = calls[0][1]["state"]
    assert state.workspace == tmp_path.resolve()
    assert state.approval_handler is approval and state.approval_mode == "inline"
    assert calls[0][1]["session_workspace"] == tmp_path
