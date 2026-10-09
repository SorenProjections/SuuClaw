"""The browser must describe the runnable workflow, including retry paths."""

from unittest.mock import patch

from suuclaw.graph.topology import describe_workflow
from suuclaw.graph.workflow import build_complex_workflow


def test_topology_exposes_every_real_workflow_edge_and_resolves_endpoints():
    topology = describe_workflow()
    ids = {node["id"] for node in topology["nodes"]}
    shown = {
        (edge["source"], edge["target"], edge["conditional"])
        for edge in topology["edges"]
        if edge["graph"] == "workflow"
    }
    actual = {
        (edge["source"], edge["target"], bool(edge.get("conditional")))
        for edge in build_complex_workflow().get_graph().to_json()["edges"]
        if edge["source"] != "__start__"
    }
    assert shown == actual
    assert all(e["source"] in ids and e["target"] in ids for e in topology["edges"])
    assert ("context_monitor", "planner", True) in shown
    assert ("context_compressor", "verifier", True) in shown


def test_entry_handoff_reaches_workflow_and_keeps_chat_path():
    topology = describe_workflow()
    edges = {(e["source"], e["target"]) for e in topology["edges"]}
    assert ("intent_router", "planner") in edges
    assert ("intent_router", "chat_responder") in edges
    assert ("chat_responder", "__end__") in edges
    assert ("intent_router", "__end__") not in edges


def test_specialists_are_explicit_delegations_not_parallel_graph_branches():
    topology = describe_workflow()
    specialists = [n for n in topology["nodes"] if n["source"] == "delegation"]
    assert {n["id"] for n in specialists} == {"search", "code"}
    assert all(n["parent"] == "planner" for n in specialists)
    links = [e for e in topology["edges"] if e["kind"] == "delegation"]
    assert {e["tool"] for e in links} == {"call_search_agent", "call_code_agent"}


def test_topology_never_constructs_or_calls_a_model():
    describe_workflow.cache_clear()
    with patch("suuclaw.providers.openai_provider.create_model") as model:
        describe_workflow()
    model.assert_not_called()
