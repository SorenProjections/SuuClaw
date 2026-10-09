"""Exercise the system overview with the real graph events and browser reducer."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from test_workflow_events import actual_workflow_events

from suuclaw.core.agent import stream_session_events
from suuclaw.core.state import RuntimeState
from suuclaw.dashboard.catalog import describe_workflow_tools
from suuclaw.dashboard.service import Redactor
from suuclaw.graph.topology import describe_workflow
from suuclaw.tools.notepad_tools import NOTEPAD_FILENAME

_STATIC = Path(__file__).parents[1] / "src" / "suuclaw" / "dashboard" / "static"

# Import the production ES module before supplying any DOM stubs. The stubs only
# let app.js's actual reducer run; neither event reduction nor overview derivation
# is reimplemented in this harness.
_NODE_ARCHITECTURE = r"""
const fs = require('node:fs');
const vm = require('node:vm');
(async () => {
  const moduleSource = fs.readFileSync(process.argv[1]);
  const {deriveArchitecture} = await import(
    'data:text/javascript;base64,' + moduleSource.toString('base64')
  );
  const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
  const source = fs.readFileSync(process.argv[2], 'utf8')
    .replace(/^import .*?;\r?$/gm, '');
  const marker = source.indexOf('// UI wiring');
  if (marker < 0) throw new Error('Dashboard UI wiring marker changed');
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {innerHTML:'', hidden:false,
      classList:{add(){},remove(){},toggle(){},contains(){return false}},
      setAttribute(){},querySelector(){return null},querySelectorAll(){return []}});
    return elements.get(id);
  };
  let result;
  const sandbox = {payload, console, performance, Date, URL,
    setTimeout(){},clearTimeout(){},requestAnimationFrame(){},
    matchMedia(){return {matches:true}},
    document:{getElementById:element,querySelectorAll(){return []},
      documentElement:{dataset:{},style:{setProperty(){}}}},
    inspect(state) {
      // The production module and VM reducer have separate JS realms. Preserve
      // all values while giving the module ordinary Maps from its own realm.
      return deriveArchitecture({...state, nodes:new Map(state.nodes),
        tools:new Map(state.tools), approvals:new Map(state.approvals)});
    },
    report(value){result=value}
  };
  vm.createContext(sandbox);
  vm.runInContext(source.slice(0, marker) + `
state.bootstrap = payload.bootstrap;
state.session = {id:'session-test',messages:[],active_run_id:payload.run?.id || null};
resetRun(payload.run);
const snapshots=[{event:null,overview:inspect(state)}];
for (const event of payload.events) {
  applyEvent(event, true);
  snapshots.push({event,overview:inspect(state)});
}
report(snapshots);
`, sandbox);
  process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def replay_architecture(events, *, bootstrap=None, running=True):
    """Read the real browser modules; no server or model connection is involved."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the system overview regression checks")
    redactor = Redactor()
    normalized = [
        {
            **redactor.value(event),
            "seq": index,
            "elapsed_ms": index * 10,
            "run_id": "run-test",
            "session_id": "session-test",
            "at": "2026-10-09T00:00:00.000Z",
        }
        for index, event in enumerate(events, start=1)
    ]
    result = subprocess.run(
        [node, "-e", _NODE_ARCHITECTURE, str(_STATIC / "architecture.js"), str(_STATIC / "app.js")],
        input=json.dumps(
            {
                "bootstrap": bootstrap or {"topology": describe_workflow()},
                "run": {"id": "run-test", "status": "running"} if running else None,
                "events": normalized,
            }
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def finished_events(events):
    final = events[-1]
    assert final["type"] == "final_answer"
    status = (
        "interrupted"
        if final.get("interrupted")
        else ("completed" if final.get("passed") else "failed")
    )
    return [
        {"type": "run_start"},
        *events,
        {
            "type": "run_end",
            "status": status,
            "error": final.get("error"),
        },
    ]


def assert_no_live_activity(overview):
    assert not overview["running"]
    assert not any(node["active"] for node in overview["nodes"])
    assert not any(node["status"] in {"active", "waiting"} for node in overview["nodes"])
    assert not any(edge["active"] for edge in overview["edges"])
    assert not any(edge["status"] in {"active", "waiting"} for edge in overview["edges"])
    assert overview["tools"]["pending"] == 0
    assert overview["approvals"]["pending"] == 0


def test_overview_exposes_actual_three_layer_memory_and_tool_observations(tmp_path, monkeypatch):
    (tmp_path / NOTEPAD_FILENAME).write_text("Retain the offline finding.", encoding="utf-8")
    (tmp_path / "HISTORY_SUMMARY.md").write_text("Prior research is complete.", encoding="utf-8")
    events = actual_workflow_events(tmp_path, monkeypatch)
    memories = [event for event in events if event["type"] == "memory"]
    assert memories, "The real workflow must supply memory events for this regression"
    assert not any(event["type"] == "checkpoint_saved" for event in events)

    snapshots = replay_architecture(finished_events(events))
    overview = snapshots[-1]["overview"]
    last_memory = memories[-1]
    assert overview["memory"]["snapshot"] == last_memory["memory"]
    assert overview["memory"]["sourceNode"] == last_memory["node"]
    assert overview["memory"]["count"] == len(memories)
    snapshot = overview["memory"]["snapshot"]
    assert snapshot["rules"]["rules"]
    assert snapshot["working_memory"]["task"] == "Create report"
    assert snapshot["history_summary_store"]["notepad"] == "Retain the offline finding."
    assert snapshot["history_summary_store"]["history_summary"] == "Prior research is complete."
    assert all(overview["nodeById"][node]["observed"] for node in ("rules", "working", "history"))
    tool_calls = [event for event in events if event["type"] == "tool_call"]
    assert overview["tools"]["total"] == len(tool_calls)
    assert {row["name"] for row in overview["tools"]["rows"]} == {
        event["name"] for event in tool_calls
    }
    assert overview["nodeById"]["tools"]["observed"]
    assert not overview["nodeById"]["checkpoints"]["observed"]
    assert overview["nodeById"]["checkpoints"]["status"] == "static"
    assert_no_live_activity(overview)


def test_available_capabilities_do_not_imply_tools_ran_or_checkpoint_was_saved(tmp_path):
    runtime = RuntimeState.create(tmp_path, checkpoint_mode="off", trace_mode="off")
    catalogue = describe_workflow_tools(runtime)
    assert catalogue
    overview = replay_architecture(
        [],
        bootstrap={
            "topology": describe_workflow(),
            "tools": catalogue,
            "model": {"name": "offline-model", "configured": True},
        },
        running=False,
    )[0]["overview"]

    assert overview["tools"]["available"] == len(catalogue)
    assert overview["tools"]["total"] == overview["tools"]["observed"] == 0
    assert overview["tools"]["succeeded"] == overview["tools"]["failed"] == 0
    assert overview["tools"]["rows"] == []
    assert not overview["nodeById"]["tools"]["observed"]
    assert not overview["nodeById"]["checkpoints"]["observed"]
    assert overview["nodeById"]["checkpoints"]["status"] == "static"
    assert_no_live_activity(overview)


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_terminal_runs_close_live_nodes_tools_and_approval_waits(status):
    snapshots = replay_architecture(
        [
            {"type": "run_start"},
            {"type": "node_start", "node": "code"},
            {
                "type": "tool_call",
                "node": "code",
                "id": "pending-tool",
                "name": "bash",
                "args": {"command": "echo offline"},
            },
            {
                "type": "approval_requested",
                "approval_id": "pending-approval",
                "tool_name": "bash",
                "command": "echo offline",
                "risk_reason": "Review required",
            },
            {"type": "run_end", "status": status},
        ]
    )
    waiting = snapshots[-2]["overview"]
    assert waiting["running"]
    assert waiting["tools"]["pending"] == 1
    assert waiting["approvals"]["pending"] == 1
    assert waiting["nodeById"]["approval"]["status"] == "waiting"
    assert_no_live_activity(snapshots[-1]["overview"])
    # An unfinished call does not become a successful tool just because the run ends.
    assert snapshots[-1]["overview"]["tools"]["succeeded"] == 0


def test_actual_search_failure_remains_failed_when_final_verification_passes(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch, search_ok=False)
    failure = next(event for event in events if event["type"] == "search_results")
    assert failure["ok"] is False and failure["error"]
    assert events[-1]["passed"] is True
    overview = replay_architecture(finished_events(events))[-1]["overview"]

    searches = [row for row in overview["tools"]["rows"] if row["name"] == "web_search"]
    assert len(searches) == 1
    assert searches[0]["status"] == "failed"
    assert searches[0]["result"]["error"] == failure["error"]
    assert overview["tools"]["failed"] == 1
    assert overview["tools"]["succeeded"] == overview["tools"]["total"] - 1
    assert overview["nodeById"]["tools"]["status"] == "failed"
    assert_no_live_activity(overview)


def test_denied_approval_and_failed_command_never_count_as_success():
    reason = "Declined in the dashboard."
    snapshots = replay_architecture(
        [
            {"type": "run_start"},
            {"type": "node_start", "node": "code"},
            {
                "type": "tool_call",
                "node": "code",
                "id": "denied-tool",
                "name": "bash",
                "args": {"command": "echo offline"},
            },
            {
                "type": "approval_requested",
                "approval_id": "denied-approval",
                "tool_name": "bash",
                "command": "echo offline",
            },
            {
                "type": "approval_resolved",
                "approval_id": "denied-approval",
                "approved": False,
                "reason": reason,
            },
            {
                "type": "tool_result",
                "node": "code",
                "id": "denied-tool",
                "name": "bash",
                "ok": False,
                "error": reason,
                "result": {"ok": False, "error": reason},
            },
            {"type": "run_end", "status": "failed"},
        ]
    )
    overview = snapshots[-1]["overview"]
    assert overview["approvals"]["denied"] == 1
    assert overview["approvals"]["approved"] == 0
    assert overview["approvals"]["rows"][0]["reason"] == reason
    assert overview["nodeById"]["approval"]["status"] == "failed"
    assert overview["tools"]["failed"] == overview["tools"]["total"] == 1
    assert overview["tools"]["succeeded"] == 0
    assert overview["tools"]["rows"][0]["result"]["error"] == reason
    assert overview["nodeById"]["tools"]["status"] == "failed"
    assert_no_live_activity(overview)


def test_chat_completion_is_not_reported_as_independent_verification(tmp_path, monkeypatch):
    class ChatModel:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content='{"route":"chat","confidence":1,"reason":"greeting"}')
            return AIMessage(content="Hello from the real chat branch")

    runtime = RuntimeState.create(tmp_path / "chat", checkpoint_mode="off", trace_mode="off")
    chat_events = list(
        stream_session_events("Hello", state=runtime, model=ChatModel(), entry_timeout=3)
    )
    workflow_events = actual_workflow_events(tmp_path / "workflow", monkeypatch)
    assert chat_events[-1]["route"] == "chat"
    assert chat_events[-1]["passed"] is workflow_events[-1]["passed"] is True
    assert not any(event.get("node") == "verifier" for event in chat_events)
    assert any(
        event["type"] == "node_start" and event.get("node") == "verifier"
        for event in workflow_events
    )

    def verification_fact(events):
        overview = replay_architecture(finished_events(events))[-1]["overview"]
        return next(
            fact["value"]
            for fact in overview["nodeById"]["response"]["facts"]
            if fact["label"] == "验证结果"
        )

    assert verification_fact(chat_events) == "本轮未经过独立验证"
    assert verification_fact(workflow_events) == "通过"
