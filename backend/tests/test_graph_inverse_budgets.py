"""Small graphs reproduce cumulative inverse amplification without stress."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from backend.graph.schema import ContentAttrs
from backend.tests.test_ingest_safety import api_state
from backend.utils import graph_ops
from backend.utils.budgets import JsonWorkBudget


def parallel_edges(api):
    builder = api.graph
    center = next(iter(builder.graph))
    target = builder.add_node(ContentAttrs(name="Other concept"))
    for index in range(20):
        # Structural multiedges remain independent; semantic associations now
        # collapse by unordered pair and cannot exercise record amplification.
        builder.graph.add_edge(center, target, key=index, relation="HAS_CONTENT", source_chunk_ids=[f"chunk_{index}"])
    builder.save()
    return center, target


@pytest.mark.parametrize("kind", ["update_node", "add_edge", "delete_edge", "restore_edges", "restore_nodes", "set_positions"])
def test_batch_inverse_record_budget_is_atomic(api_state, monkeypatch, kind):
    api = api_state
    center, target = parallel_edges(api)
    graph = api.graph.graph
    before = (api.processed / "graph.json").read_bytes()
    edges = [{"key": key, "attrs": deepcopy(attrs)} for key, attrs in graph.get_edge_data(center, target).items()]
    if kind == "update_node":
        operations = [{"op": kind, "id": center, "updates": {"summary": str(index)}} for index in range(3)]
    elif kind == "add_edge":
        operations = [{"op": kind, "edge": {"source": center, "target": target, "relation": "HAS_CONTENT"}} for index in range(3)]
    elif kind == "delete_edge":
        operations = [{"op": kind, "edge": {"source": center, "target": target, "relation": "HAS_CONTENT"}} for index in range(3)]
    elif kind == "restore_edges":
        operations = [{"op": kind, "source": center, "target": target, "edges": edges}] * 3
    elif kind == "restore_nodes":
        operations = [{"op": kind, "ids": [center], "nodes": [{"id": center, "attrs": dict(graph.nodes[center])}],
                       "edges": [{"source": center, "target": target, **edge} for edge in edges]}] * 3
    else:
        operations = [{"op": kind, "positions": [{"id": center, "x": index, "y": 0} for index in range(20)]}] * 3
    monkeypatch.setattr(graph_ops, "MAX_INVERSE_ITEMS", 160)
    response = api.client.post("/api/graph/ops", json={"ops": operations})
    assert response.status_code == 422
    assert "undo processing limit" in response.json()["detail"]
    assert api.main.state.graph_builder.graph.nodes[center].get("summary") != "2"
    assert api.main.state.graph_builder.graph.number_of_edges() == 20
    assert "x" not in api.main.state.graph_builder.graph.nodes[center]
    assert (api.processed / "graph.json").read_bytes() == before
    assert list(api.staging.iterdir()) == []


@pytest.mark.parametrize("snapshot", ["node", "edge"])
def test_snapshot_reserves_all_records_before_deepcopy(api_state, monkeypatch, snapshot):
    center, target = parallel_edges(api_state)
    copy = Mock(wraps=deepcopy)
    monkeypatch.setattr(graph_ops, "deepcopy", copy)
    budget = JsonWorkBudget(max_items=15, max_bytes=100_000, copies=3)
    with pytest.raises(ValueError, match="undo processing limit"):
        if snapshot == "node":
            graph_ops._node_snapshot(api_state.graph.graph, [center], budget)
        else:
            graph_ops._edge_snapshot(api_state.graph.graph, center, target, budget)
    copy.assert_not_called()


def test_inverse_bytes_include_compatibility_response_copies(api_state, monkeypatch):
    center = next(iter(api_state.graph.graph))
    api_state.graph.graph.nodes[center]["summary"] = "x" * 2000
    api_state.graph.save()
    before = (api_state.processed / "graph.json").read_bytes()
    monkeypatch.setattr(graph_ops, "MAX_INVERSE_BYTES", 5000)
    response = api_state.client.post("/api/graph/ops", json={"ops": [{"op": "update_node", "id": center, "updates": {"summary": "short"}}]})
    assert response.status_code == 422
    assert (api_state.processed / "graph.json").read_bytes() == before
    assert api_state.main.state.graph_builder.graph.nodes[center]["summary"] == "x" * 2000


def test_budgeted_inverse_still_undoes_multiedges(api_state):
    center, target = parallel_edges(api_state)
    response = api_state.client.post("/api/graph/ops", json={"ops": [{"op": "update_node", "id": center, "updates": {"summary": "Edited"}}]})
    assert response.status_code == 200
    inverse = response.json()["inverses"]
    assert api_state.client.post("/api/graph/ops", json={"ops": inverse}).status_code == 200
    assert api_state.main.state.graph_builder.graph.number_of_edges(center, target) == 20
    assert api_state.main.state.graph_builder.graph.nodes[center]["summary"] != "Edited"
