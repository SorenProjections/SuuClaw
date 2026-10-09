"""Use the real graph and specialists to verify the browser's event contract offline."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from suuclaw.core.agent import iter_workflow_events
from suuclaw.core.state import RuntimeState
from suuclaw.graph.nodes import make_context_compressor_node, make_final_node
from suuclaw.graph.state import SuuClawGraphState
from suuclaw.graph.workflow import build_complex_workflow, initial_state
from suuclaw.tools.web_search_tool import WebSearchTool


class OfflineModel:
    """Deterministic model responses; all real graph routing/tools remain in use."""

    def __init__(self, verdicts=(True,)):
        self.verdicts = list(verdicts)
        self.verifications = 0

    def bind_tools(self, tools):
        names = {tool.name for tool in tools}
        if "call_code_agent" in names:
            role = "planner"
        elif names == {"web_search"}:
            role = "search"
        elif "file_write" in names:
            role = "code"
        else:
            role = "verifier"
        return OfflineBoundModel(self, role)

    def invoke(self, messages):
        return AIMessage(content=json.dumps({"summary": "Offline compression summary"}))

    def get_num_tokens_from_messages(self, messages):
        return 1000


class OfflineBoundModel:
    def __init__(self, model, role):
        self.model, self.role, self.calls = model, role, 0

    def invoke(self, messages):
        self.calls += 1
        if self.role == "verifier":
            position = self.model.verifications
            self.model.verifications += 1
            passed = self.model.verdicts[min(position, len(self.model.verdicts) - 1)]
            return AIMessage(
                content=json.dumps({"passed": passed, "checks": [], "reason": "Offline verdict"})
            )
        if self.calls > 1:
            return AIMessage(content=f"{self.role} finished")
        if self.role == "planner":
            calls = [
                {
                    "name": "todo_write",
                    "id": "plan",
                    "args": {
                        "plan_summary": "Research and implement",
                        "todos": [{"id": "1", "content": "Write report"}],
                        "acceptance_criteria": ["Report exists"],
                        "verification_commands": [],
                    },
                },
                {
                    "name": "call_search_agent",
                    "id": "delegate-search",
                    "args": {"instruction": "Research"},
                },
                {
                    "name": "call_code_agent",
                    "id": "delegate-code",
                    "args": {"instruction": "Write report"},
                },
            ]
        elif self.role == "search":
            calls = [
                {"name": "web_search", "id": "web-result", "args": {"query": "offline source"}}
            ]
        else:
            calls = [
                {
                    "name": "file_write",
                    "id": "write-result",
                    "args": {"file_path": "report.txt", "content": "Offline artifact"},
                }
            ]
        return AIMessage(content="", tool_calls=calls)


def actual_workflow_events(
    tmp_path, monkeypatch, *, verdicts=(True,), compress=False, search_ok=True, code_loops=8
):
    monkeypatch.setattr(
        WebSearchTool,
        "search",
        lambda self, query, **kwargs: {
            "ok": search_ok,
            "query": query,
            "answer": "Offline finding",
            "error": "" if search_ok else "Offline search unavailable",
            "results": [
                {"title": "Offline source", "url": "https://example.invalid/source", "score": 1.0}
            ],
        },
    )
    runtime = RuntimeState.create(tmp_path, checkpoint_mode="off", trace_mode="off")
    graph = build_complex_workflow(
        OfflineModel(verdicts), summarize=False, retries=0, code_agent_loops=code_loops
    )
    inputs = initial_state(
        "Create report",
        runtime,
        max_attempts=len(verdicts),
        context_token_limit=1 if compress else 10000,
    )
    return list(iter_workflow_events(graph, inputs, recursion_limit=30))


def test_actual_specialist_return_contract(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch)
    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == "Offline artifact"
    # Delegations execute inside one planner superstep: three appearances of
    # planner are resumes around two specialists, not three planning attempts.
    starts = [event["node"] for event in events if event["type"] == "node_start"]
    assert starts[:5] == ["planner", "search", "planner", "code", "planner"]
    planner_outputs = [
        event for event in events if event["type"] == "node_output" and event["node"] == "planner"
    ]
    assert len(planner_outputs) == 1
    assert [event["to"] for event in events if event["type"] == "handoff"] == [
        "searchAgent",
        "codeAgent",
    ]
    search_result = next(event for event in events if event["type"] == "search_results")
    search_call = next(
        event for event in events if event["type"] == "tool_call" and event["node"] == "search"
    )
    assert search_result["id"] == search_call["id"]
    assert search_result["ok"] and search_result["sources"]
    code_result = next(
        event for event in events if event["type"] == "tool_result" and event["node"] == "code"
    )
    assert code_result["output"] and "result" not in code_result
    assert events[-1]["type"] == "final_answer" and events[-1]["passed"]


def test_actual_compression_and_retry_routes(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch, verdicts=(False, True), compress=True)
    outputs = [event["node"] for event in events if event["type"] == "node_output"]
    assert outputs == [
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "context_monitor",
        "context_compressor",
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "context_monitor",
        "final",
    ]
    verifier = [
        event["data"]["passed"]
        for event in events
        if event["type"] == "node_output" and event["node"] == "verifier"
    ]
    assert verifier == [False, True]
    assert events[-1]["passed"] and events[-1]["attempts"] == 2
    assert sum(event["type"] == "final_answer" for event in events) == 1


def test_actual_exhausted_verification_keeps_failed_outcome(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch, verdicts=(False,))
    assert events[-1]["type"] == "final_answer"
    assert events[-1]["passed"] is False and events[-1]["exhausted"] is True
    assert events[-1]["error"] is None  # A failed verdict need not be a runtime exception.
    assert events[-1]["interrupted"] is False


@pytest.mark.parametrize("node", ["context_compressor", "final"])
def test_slow_node_start_reaches_consumer_before_model_finishes(tmp_path: Path, node: str):
    release, invoked, finished = threading.Event(), threading.Event(), threading.Event()

    class BlockingModel:
        def invoke(self, messages):
            invoked.set()
            release.wait(2)
            finished.set()
            return AIMessage(content='{"summary":"Done"}')

    factory = make_context_compressor_node if node == "context_compressor" else make_final_node
    graph = StateGraph(SuuClawGraphState)
    graph.add_node(node, factory(BlockingModel()))
    graph.add_edge(START, node)
    graph.add_edge(node, END)
    runtime = RuntimeState.create(tmp_path, checkpoint_mode="off", trace_mode="off")
    stream = iter_workflow_events(graph.compile(), initial_state("Timing check", runtime))
    try:
        first = next(stream)
        assert first == {"type": "node_start", "node": node}
        assert invoked.wait(1)
        assert not finished.is_set(), "Start arrived only after the model had already finished"
    finally:
        release.set()
    remaining = list(stream)
    assert not any(event["type"] == "node_start" and event["node"] == node for event in remaining)
    assert any(event["type"] == "node_output" and event["node"] == node for event in remaining)


def test_chat_node_starts_before_blocked_model_call_finishes(tmp_path: Path):
    from suuclaw.core.agent import stream_session_events

    release, invoked, finished = threading.Event(), threading.Event(), threading.Event()

    class ChatModel:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content='{"route":"chat","confidence":1,"reason":"greeting"}')
            invoked.set()
            release.wait(2)
            finished.set()
            return AIMessage(content="Hello from the real chat branch")

    runtime = RuntimeState.create(tmp_path, checkpoint_mode="off", trace_mode="off")
    stream = stream_session_events("Hello", state=runtime, model=ChatModel(), entry_timeout=3)
    observed = []
    try:
        for event in stream:
            observed.append(event)
            if event == {"type": "node_start", "node": "chat_responder"}:
                break
        assert invoked.wait(1)
        assert not finished.is_set(), "Chat start arrived after its response had already completed"
    finally:
        release.set()
    observed.extend(stream)
    assert [event["node"] for event in observed if event["type"] == "node_start"] == [
        "intent_router",
        "chat_responder",
    ]
    assert observed[-1]["content"] == "Hello from the real chat branch"
    assert observed[-1]["route"] == "chat"


def test_entry_timeout_falls_back_and_late_chat_cannot_overwrite_session(tmp_path, monkeypatch):
    from suuclaw.core import agent
    from suuclaw.core.session import load_or_create_session

    release, late_finished = threading.Event(), threading.Event()

    class LateChatModel:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content='{"route":"chat","confidence":1,"reason":"greeting"}')
            release.wait(2)
            late_finished.set()
            return AIMessage(content="LATE CHAT MUST NOT WIN")

    def fallback(task, **kwargs):
        release.set()
        assert late_finished.wait(1)
        yield {"type": "node_start", "node": "planner"}
        yield {
            "type": "final_answer",
            "node": "final",
            "content": "Workflow fallback",
            "passed": True,
            "error": None,
        }

    monkeypatch.setattr(agent, "stream_agent_events", fallback)
    runtime = RuntimeState.create(tmp_path, checkpoint_mode="off", trace_mode="off")
    events = list(
        agent.stream_session_events(
            "Hello", state=runtime, model=LateChatModel(), entry_timeout=0.1
        )
    )
    assert any(
        event["type"] == "error" and "entry route exceeded" in event["message"] for event in events
    )
    assert events[-1]["route"] == "workflow" and events[-1]["content"] == "Workflow fallback"
    assert not any("LATE CHAT MUST NOT WIN" in json.dumps(event) for event in events)
    session = load_or_create_session(tmp_path)
    assert "LATE CHAT MUST NOT WIN" not in json.dumps(session)
    assert "Workflow fallback" in json.dumps(session)


def test_entry_deadline_covers_whole_graph_not_each_node():
    from suuclaw.core.agent import _iter_entry_events

    class SlowEntryGraph:
        def stream(self, *args, **kwargs):
            yield "custom", {"type": "node_start", "node": "intent_router"}
            threading.Event().wait(0.07)
            yield "updates", {"intent_router": {"intent_route": "chat"}}
            yield "custom", {"type": "node_start", "node": "chat_responder"}
            threading.Event().wait(0.07)
            yield "updates", {"chat_responder": {"chat_response": "too late"}}

    with pytest.raises(TimeoutError, match="entry route exceeded"):
        list(_iter_entry_events(SlowEntryGraph, {}, 0.1))


def test_slow_entry_consumer_can_drain_on_time_results():
    from suuclaw.core.agent import _iter_entry_events

    done = threading.Event()

    class FastEntryGraph:
        def stream(self, *args, **kwargs):
            yield "custom", {"type": "node_start", "node": "intent_router"}
            yield "updates", {"intent_router": {"intent_route": "workflow"}}
            done.set()

    stream = _iter_entry_events(FastEntryGraph, {}, 0.1)
    assert next(stream)["type"] == "node_start"
    assert done.wait(1)
    threading.Event().wait(0.12)
    remaining = list(stream)
    assert remaining[0]["data"]["intent_route"] == "workflow"


_NODE_REPLAY = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*?;\r?$/gm, '');
const marker = source.indexOf('// UI wiring');
if (marker < 0) throw new Error('Dashboard UI wiring marker changed; update replay harness');
const elements = new Map();
const element = (id) => {
  if (!elements.has(id)) elements.set(id, {innerHTML:'', hidden:false,
    classList:{add(){},remove(){},toggle(){},contains(){return false}},
    setAttribute(){}, querySelector(){return null}, querySelectorAll(){return []}});
  return elements.get(id);
};
let result;
const sandbox = {payload, console, performance, Date, URL, setTimeout(){},clearTimeout(){},
  requestAnimationFrame(){},matchMedia(){return {matches:true}},
  document:{getElementById:element,querySelectorAll(){return []},
    documentElement:{dataset:{},style:{setProperty(){}}}},
  report(value){result=value}};
vm.createContext(sandbox);
vm.runInContext(source.slice(0, marker) + `
state.bootstrap = payload.bootstrap;
state.session = {id:'session-test', messages:[], active_run_id:'run-test'};
resetRun({id:'run-test',status:'running'});
const snapshots=[];
for (const event of payload.events) {
  applyEvent(event, true);
  snapshots.push(JSON.parse(JSON.stringify({type:event.type,eventNode:event.node,seq:event.seq,active:state.activeNode,
    nodes:Object.fromEntries(state.nodes), tools:[...state.tools.values()],
    transitions:[...observedTransitions()]})));
}
report(snapshots);
`, sandbox);
process.stdout.write(JSON.stringify(result));
"""


def replay_in_browser_reducer(events):
    import shutil
    import subprocess

    from suuclaw.dashboard.service import Redactor
    from suuclaw.graph.topology import describe_workflow

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the browser reducer regression check")
    normalized = [{"type": "run_start"}, *events]
    final = events[-1]
    status = (
        "interrupted"
        if final.get("interrupted")
        else "completed"
        if final.get("passed")
        else "failed"
    )
    normalized.append({"type": "run_end", "status": status, "error": final.get("error")})
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
        for index, event in enumerate(normalized, start=1)
    ]
    source = Path(__file__).parents[1] / "src" / "suuclaw" / "dashboard" / "static" / "app.js"
    result = subprocess.run(
        [node, "-e", _NODE_REPLAY, str(source)],
        input=json.dumps({"bootstrap": {"topology": describe_workflow()}, "events": normalized}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_browser_reducer_tracks_real_delegations_and_tool_results(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch)
    snapshots = replay_in_browser_reducer(events)
    search_start = next(
        row for row in snapshots if row["type"] == "node_start" and row["eventNode"] == "search"
    )
    assert search_start["nodes"]["planner"]["status"] in {"waiting", "delegated"}
    assert search_start["nodes"]["search"]["status"] == "running"
    planner_output = next(
        row for row in snapshots if row["type"] == "node_output" and row["eventNode"] == "planner"
    )
    assert planner_output["nodes"]["planner"]["visits"] == 1
    assert not planner_output["active"]
    search_result = next(row for row in snapshots if row["type"] == "search_results")
    search_tool = next(tool for tool in search_result["tools"] if tool["name"] == "web_search")
    assert search_tool["status"] == "completed"
    code_result = next(
        row for row in snapshots if row["type"] == "tool_result" and row["eventNode"] == "code"
    )
    code_tool = next(tool for tool in code_result["tools"] if tool["name"] == "file_write")
    assert code_tool["result"]
    transitions = set(snapshots[-1]["transitions"])
    assert {"planner:search", "search:planner", "planner:code", "code:planner"} <= transitions
    assert not any(
        node["status"] in {"running", "waiting", "delegated"}
        for node in snapshots[-1]["nodes"].values()
    )


def test_browser_reducer_keeps_real_retry_and_compression_route(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch, verdicts=(False, True), compress=True)
    snapshots = replay_in_browser_reducer(events)
    end = snapshots[-1]
    assert end["nodes"]["planner"]["visits"] == 2
    assert end["nodes"]["verifier"]["visits"] == 2
    assert {
        "context_monitor:context_compressor",
        "context_compressor:planner",
        "context_compressor:verifier",
    } <= set(end["transitions"])
    failed_verifier_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "node_output"
        and event["node"] == "verifier"
        and event["data"]["passed"] is False
    )
    assert snapshots[failed_verifier_index + 1]["nodes"]["verifier"]["status"] in {
        "failed",
        "rejected",
    }
    assert end["nodes"]["__end__"]["status"] == "completed"


def test_browser_reducer_does_not_show_failed_run_as_success(tmp_path, monkeypatch):
    events = actual_workflow_events(tmp_path, monkeypatch, verdicts=(False,))
    snapshots = replay_in_browser_reducer(events)
    assert snapshots[-1]["nodes"]["__end__"]["status"] in {"failed", "interrupted"}
    assert not snapshots[-1]["active"]


@pytest.mark.parametrize("case", ["search_failure", "code_exhaustion"])
def test_browser_reducer_preserves_specialist_failure_even_if_run_later_passes(
    tmp_path, monkeypatch, case
):
    events = actual_workflow_events(
        tmp_path,
        monkeypatch,
        search_ok=case != "search_failure",
        code_loops=1 if case == "code_exhaustion" else 8,
    )
    specialist = "search" if case == "search_failure" else "code"
    summary = next(
        event for event in events if event["type"] == "node_summary" and event["node"] == specialist
    )
    assert not summary["error"]  # A blank runtime error is not proof of success.
    snapshots = replay_in_browser_reducer(events)
    closed = next(
        row for row in snapshots if row["type"] == "node_summary" and row["eventNode"] == specialist
    )
    assert closed["nodes"][specialist]["status"] == "failed"
    assert snapshots[-1]["nodes"]["__end__"]["status"] == "completed"


def test_browser_reducer_preserves_runtime_exception_without_inventing_success(tmp_path):
    class BrokenModel(OfflineModel):
        def bind_tools(self, tools):
            class Bound:
                def invoke(self, messages):
                    raise RuntimeError("Offline permanent model failure")

            return Bound()

    runtime = RuntimeState.create(tmp_path, checkpoint_mode="off", trace_mode="off")
    graph = build_complex_workflow(BrokenModel(), summarize=False, retries=0)
    events = list(
        iter_workflow_events(graph, initial_state("Fail safely", runtime, max_attempts=1))
    )
    assert events[-1]["interrupted"] is True
    snapshots = replay_in_browser_reducer(events)
    assert snapshots[-1]["nodes"]["__end__"]["status"] == "failed"
    assert snapshots[-1]["nodes"]["planner"]["status"] == "failed"
    assert not snapshots[-1]["active"]
