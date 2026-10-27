"""Plain connections retain source provenance across API and bundle boundaries."""
from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy

import networkx as nx
import pytest

from backend.schemas import Chunk
from backend.tests.test_ingest_safety import api_state  # noqa: F401
from backend.tests.test_workspace_bundles import workspaces  # noqa: F401
from backend.utils.associations import add_association, association_id
from backend.utils.state import reconcile_graph_sources
from backend.workspace import export_bundle, import_bundle, read_bundle, validate_workspace_data


def _archive(chunks, nodes, edges):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"format": "lumin", "version": 1}))
        archive.writestr("chunks.jsonl", "\n".join(chunk.model_dump_json() for chunk in chunks))
        archive.writestr("graph.json", json.dumps({"nodes": nodes, "edges": edges}))
    return result.getvalue()


def _fail_provider(*_args, **_kwargs):
    pytest.fail("Loading or exporting a graph must not ask an AI to rebuild it")


def test_legacy_round_trip_preserves_hierarchy_citations_and_positions(workspaces, monkeypatch):
    from backend.graph import semantic_linker

    monkeypatch.setattr(semantic_linker, "link_graph_changes", _fail_provider)
    monkeypatch.setattr(semantic_linker, "safe_call_json", _fail_provider)
    chunk = Chunk(id="page:3", source_id="greenhouse", modality="pdf", text="Open vents at 29 C.",
                  metadata={"original_filename": "greenhouse.pdf", "page": 3, "page_start": 3, "page_end": 3})
    nodes = [
        {"id": "topic", "node_type": "topic", "name": "Greenhouses", "source_ids": [chunk.id], "x": 5.5, "y": -9},
        {"id": "subtopic", "node_type": "subtopic", "name": "Ventilation", "source_ids": [chunk.id], "parent_topic_keys": ["topic"]},
        {"id": "content", "node_type": "content", "name": "Temperature", "body": "Keep this separate concept.", "source_ids": [chunk.id], "parent_subtopic_keys": ["subtopic"]},
    ]
    edges = [
        {"source": "topic", "target": "subtopic", "key": "hierarchy", "relation": "HAS_SUBTOPIC"},
        {"source": "subtopic", "target": "content", "key": "child", "relation": "HAS_CONTENT"},
        {"source": "topic", "target": "content", "key": "old-forward", "relation": "RELATED_TO", "label": "causes", "confidence": 0.7, "source_chunk_ids": [chunk.id]},
        {"source": "content", "target": "topic", "key": "old-reversed", "relation": "RELATED_TO", "label": "explains", "confidence": None, "source_chunk_ids": []},
    ]
    import uuid
    ws = workspaces.create(str(uuid.uuid4()))
    import_bundle(ws, _archive([chunk], nodes, edges))
    _, restored_chunks, graph, _ = read_bundle(export_bundle(ws))
    assert set(graph) == {"topic", "subtopic", "content"}
    assert graph.nodes["topic"]["x"] == 5.5 and graph.nodes["topic"]["y"] == -9
    assert graph.nodes["content"]["body"] == "Keep this separate concept."
    assert graph.nodes["content"]["parent_subtopic_keys"] == ["subtopic"]
    assert graph.has_edge("topic", "subtopic", "hierarchy")
    assert graph.has_edge("subtopic", "content", "child")
    restored_chunk = restored_chunks[0]
    assert (restored_chunk.id, restored_chunk.source_id, restored_chunk.text, restored_chunk.modality) == (chunk.id, chunk.source_id, chunk.text, chunk.modality)
    assert all(restored_chunk.metadata[key] == value for key, value in chunk.metadata.items())
    associations = [(u, v, key, attrs) for u, v, key, attrs in graph.edges(keys=True, data=True) if attrs["relation"] == "RELATED_TO"]
    assert len(associations) == 1
    u, v, key, attrs = associations[0]
    assert (u, v) == ("content", "topic") and key == association_id(u, v)
    assert attrs["label"] is None
    assert len(attrs["supports"]) == 2
    assert all(record["kind"] == "legacy" and record["verified"] is False for record in attrs["supports"])
    assert nx.utils.graphs_equal(graph, read_bundle(export_bundle(ws))[2])


def test_source_replacement_removes_only_its_automatic_backing():
    graph = nx.MultiDiGraph()
    for key in ("a", "b"):
        graph.add_node(key, node_type="content", name=key, source_ids=["old-a", "old-b", "new-a", "new-b"])
    add_association(graph, "a", "b", kind="automatic", source_chunk_ids=["old-a", "old-b"], verified=True)
    add_association(graph, "a", "b", kind="automatic", source_chunk_ids=["new-a", "new-b"], verified=True)
    add_association(graph, "a", "b", kind="manual")
    reconcile_graph_sources(graph, ["old-a"])
    attrs = graph["a"]["b"][association_id("a", "b")]
    assert {(support["kind"], tuple(support["source_chunk_ids"])) for support in attrs["supports"]} == {
        ("automatic", ("new-a", "new-b")), ("manual", ()),
    }
    assert "old-b" not in attrs["source_chunk_ids"]
    assert set(graph) == {"a", "b"}


def test_api_delete_undo_redo_retains_all_connection_backing(api_state):
    api = api_state
    graph = api.graph.graph
    graph.clear()
    graph.add_node("a", node_type="content", name="A", source_ids=["one"])
    graph.add_node("b", node_type="content", name="B", source_ids=["two"])
    api.store.save_chunks([Chunk(id=key, source_id=key, text=key, modality="markdown") for key in ("one", "two")])
    add_association(graph, "b", "a", kind="automatic", source_chunk_ids=["one", "two"], verified=True)
    add_association(graph, "a", "b", kind="manual")
    before = deepcopy(api.client.get("/api/graph").json())
    assert before["association_version"] == 1
    edge = before["edges"][0]
    assert edge["id"] == association_id("a", "b") and edge["label"] is None
    assert len(edge["supports"]) == 2
    deleted = api.client.post("/api/graph/ops", json={"ops": [{"op": "delete_edge", "edge": {"id": edge["id"]}}]})
    assert deleted.status_code == 200, deleted.text
    payload = deleted.json()
    assert payload["graph"]["edges"] == []
    inverse = payload["inverses"]
    restored = api.client.post("/api/graph/ops", json={"ops": inverse})
    assert restored.status_code == 200, restored.text
    assert restored.json()["graph"] == before
    redone = api.client.post("/api/graph/ops", json={"ops": restored.json()["inverses"]})
    assert redone.status_code == 200 and redone.json()["graph"]["edges"] == []


def test_existing_edge_endpoints_accept_plain_and_legacy_reversed_requests(api_state):
    graph = api_state.graph.graph
    graph.clear()
    for key in ("a", "b"):
        graph.add_node(key, node_type="content", name=key)
    for source, target, extra in [("a", "b", {}), ("b", "a", {"label": "explains"})]:
        response = api_state.client.post("/api/graph/edge", json={"source": source, "target": target, "relation": "RELATED_TO", **extra})
        assert response.status_code == 201, response.text
        assert response.json()["id"] == association_id("a", "b")
        assert response.json()["label"] is None
        assert len(response.json()["supports"]) == 1
    assert len(api_state.client.get("/api/graph").json()["edges"]) == 1
    response = api_state.client.delete("/api/graph/edge", params={"source": "b", "target": "a", "relation": "RELATED_TO", "label": "any old label"})
    assert response.status_code == 200
    assert response.json()["deleted"] == association_id("a", "b")
    assert api_state.client.get("/api/graph").json()["edges"] == []


@pytest.mark.parametrize("support", [
    {"kind": "invented"}, {"kind": "automatic", "verified": "true"},
    {"kind": "automatic", "confidence": float("nan")},
    {"kind": "manual", "verified": True}, {"kind": "wikilink", "owner": "absent"},
])
def test_bundle_rejects_invalid_connection_support(support):
    nodes = [{"id": key, "node_type": "content", "name": key} for key in ("a", "b")]
    edges = [{"source": "a", "target": "b", "relation": "RELATED_TO", "supports": [support]}]
    with pytest.raises(ValueError):
        read_bundle(_archive([], nodes, edges))


@pytest.mark.parametrize("support_ids", [["one", "foreign"], ["one"]])
def test_verified_backing_requires_available_passages_for_both_concepts(support_ids):
    graph = nx.MultiDiGraph()
    graph.add_node("a", node_type="content", name="A", source_ids=["one"])
    graph.add_node("b", node_type="content", name="B", source_ids=["two"])
    add_association(graph, "a", "b", kind="automatic", verified=True, source_chunk_ids=support_ids)
    chunks = [Chunk(id=key, source_id=key, text=key, modality="markdown") for key in ("one", "two")]
    with pytest.raises(ValueError, match="ground both concepts"):
        validate_workspace_data(chunks, graph)
