"""Graph export contracts, legacy filtering and application-specific fields."""
from __future__ import annotations

import json

import networkx as nx

from backend.graph.graph_export import export_graph


def _hierarchical_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("topic:aaa111", name="NLP", node_type="topic", source_ids=["c1"],
                   summary="Natural language processing", scope="broad")
    graph.add_node("subtopic:bbb222", name="Transformers", node_type="subtopic", source_ids=["c1"],
                   summary="Attention-based models", parent_topic_keys=["topic:aaa111"])
    graph.add_node("content:ccc333", name="Self-Attention", node_type="content", source_ids=["c1"],
                   content_type="definition", summary="Attention formula", parent_subtopic_keys=["subtopic:bbb222"])
    graph.add_edge("topic:aaa111", "subtopic:bbb222", relation="HAS_SUBTOPIC", source_chunk_ids=["c1"])
    graph.add_edge("subtopic:bbb222", "content:ccc333", relation="HAS_CONTENT", source_chunk_ids=["c1"])
    graph.add_edge("topic:aaa111", "subtopic:bbb222", relation="RELATED_TO", label="uses", confidence=.75, source_chunk_ids=[])
    return graph


def test_hierarchy_and_plain_connections_round_trip_to_nested_output(tmp_path):
    path = tmp_path / "nested" / "deep" / "graph.json"
    result = export_graph(_hierarchical_graph(), path)
    assert json.loads(path.read_text()) == result
    assert result["association_version"] == 1
    assert len(result["nodes"]) == len(result["edges"]) == 3
    nodes = {node["id"]: node for node in result["nodes"]}
    assert {key: node["node_type"] for key, node in nodes.items()} == {
        "topic:aaa111": "topic", "subtopic:bbb222": "subtopic", "content:ccc333": "content"}
    assert nodes["topic:aaa111"]["summary"] == "Natural language processing"
    assert nodes["topic:aaa111"]["scope"] == "broad"
    content = nodes["content:ccc333"]
    assert content["summary"] == "Attention formula" and content["content_type"] == "definition"
    assert not {"summary_beginner", "summary_intermediate", "summary_expert"}.intersection(content)
    edges = {edge["relation"]: edge for edge in result["edges"]}
    parent = edges["HAS_SUBTOPIC"]
    assert (parent["source"], parent["target"]) == ("topic:aaa111", "subtopic:bbb222")
    child = edges["HAS_CONTENT"]
    assert "label" not in child and "confidence" not in child
    association = edges["RELATED_TO"]
    assert association["label"] is None and association["id"].startswith("assoc:")
    assert association["confidence"] == .75
    assert association["supports"][0]["kind"] == "legacy"
    assert association["supports"][0]["verified"] is False


def test_legacy_chunk_refs_and_their_evidence_edges_are_filtered(tmp_path):
    graph = _hierarchical_graph()
    graph.add_node("c1", name="page 1", node_type="chunk_ref", modality="pdf", source_id="doc.pdf", locator={"page": 1})
    graph.add_edge("c1", "content:ccc333", relation="EVIDENCE_OF", source_chunk_ids=[])
    result = export_graph(graph, tmp_path / "legacy.json")
    assert "c1" not in {node["id"] for node in result["nodes"]}
    assert not any(edge["relation"] == "EVIDENCE_OF" for edge in result["edges"])


def test_content_enrichment_fields_are_preserved(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("content:xxx", name="Self-Attention", node_type="content", source_ids=["c1"],
                   content_type="definition", summary="...", raw_excerpt="softmax(QK^T / sqrt(d_k)) V",
                   key_terms=["softmax", "QK^T", "d_k"], illustration_path="/static/illustrations/content_xxx.png")
    content = export_graph(graph, tmp_path / "enriched.json")["nodes"][0]
    assert content["raw_excerpt"] == "softmax(QK^T / sqrt(d_k)) V"
    assert content["key_terms"] == ["softmax", "QK^T", "d_k"]
    assert content["illustration_path"] == "/static/illustrations/content_xxx.png"


def test_empty_and_isolated_graphs_export_without_inventing_edges(tmp_path):
    graph = nx.MultiDiGraph()
    assert export_graph(graph, tmp_path / "empty.json") == {"association_version": 1, "nodes": [], "edges": []}
    graph.add_node("topic:xxx", name="Orphan", node_type="topic", source_ids=[])
    result = export_graph(graph, tmp_path / "orphan.json")
    assert len(result["nodes"]) == 1 and result["edges"] == []
