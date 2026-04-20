"""
Tests for backend/graph/graph_export.py

Asserts the exporter emits node_type on every node, preserves rich fields for
the three node types (topic/subtopic/content), surfaces RELATED_TO label/confidence
on semantic edges, and skips legacy chunk_ref nodes.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from backend.graph.graph_export import export_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hierarchical_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("topic:aaa111", name="NLP", node_type="topic", source_ids=["c1"],
               summary="Natural language processing", scope="broad")
    g.add_node("subtopic:bbb222", name="Transformers", node_type="subtopic",
               source_ids=["c1"], summary="Attention-based models",
               parent_topic_keys=["topic:aaa111"])
    g.add_node("content:ccc333", name="Self-Attention", node_type="content",
               source_ids=["c1"], content_type="definition",
               summary="Attention formula",
               parent_subtopic_keys=["subtopic:bbb222"])
    g.add_edge("topic:aaa111", "subtopic:bbb222",
               relation="HAS_SUBTOPIC", source_chunk_ids=["c1"])
    g.add_edge("subtopic:bbb222", "content:ccc333",
               relation="HAS_CONTENT", source_chunk_ids=["c1"])
    g.add_edge("topic:aaa111", "subtopic:bbb222",
               relation="RELATED_TO", label="uses", confidence=0.75,
               source_chunk_ids=[])
    return g


def _graph_with_chunk_ref() -> nx.MultiDiGraph:
    """Legacy graph that still has a chunk_ref node — should be filtered out."""
    g = _hierarchical_graph()
    g.add_node("c1", name="page 1", node_type="chunk_ref",
               modality="pdf", source_id="doc.pdf",
               locator={"page": 1})
    g.add_edge("c1", "content:ccc333", relation="EVIDENCE_OF", source_chunk_ids=[])
    return g


# ---------------------------------------------------------------------------
# Structural output
# ---------------------------------------------------------------------------

class TestExportStructure:
    def test_nodes_and_edges_keys(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        assert "nodes" in result
        assert "edges" in result

    def test_counts_match(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 3

    def test_chunk_ref_nodes_are_skipped(self, tmp_path):
        result = export_graph(_graph_with_chunk_ref(), tmp_path / "g.json")
        ids = [n["id"] for n in result["nodes"]]
        assert "c1" not in ids

    def test_evidence_of_edges_excluded_when_chunk_ref_filtered(self, tmp_path):
        result = export_graph(_graph_with_chunk_ref(), tmp_path / "g.json")
        assert not any(e["relation"] == "EVIDENCE_OF" for e in result["edges"])


# ---------------------------------------------------------------------------
# node_type round-trip
# ---------------------------------------------------------------------------

class TestNodeTypeRoundTrip:
    def test_every_node_has_node_type(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        assert all("node_type" in n for n in result["nodes"])

    def test_node_types_correct(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        by_id = {n["id"]: n["node_type"] for n in result["nodes"]}
        assert by_id["topic:aaa111"] == "topic"
        assert by_id["subtopic:bbb222"] == "subtopic"
        assert by_id["content:ccc333"] == "content"


# ---------------------------------------------------------------------------
# Rich node fields
# ---------------------------------------------------------------------------

class TestRichNodeFields:
    def test_topic_summary_and_scope(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        t = next(n for n in result["nodes"] if n["id"] == "topic:aaa111")
        assert t["summary"] == "Natural language processing"
        assert t["scope"] == "broad"

    def test_content_single_summary(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        c = next(n for n in result["nodes"] if n["id"] == "content:ccc333")
        assert c["summary"] == "Attention formula"
        assert c["content_type"] == "definition"
        assert "summary_beginner" not in c
        assert "summary_intermediate" not in c
        assert "summary_expert" not in c

    def test_content_exports_enrichment_fields(self, tmp_path):
        g = nx.MultiDiGraph()
        g.add_node(
            "content:xxx",
            name="Self-Attention",
            node_type="content",
            source_ids=["c1"],
            content_type="definition",
            summary="...",
            raw_excerpt="softmax(QK^T / sqrt(d_k)) V",
            key_terms=["softmax", "QK^T", "d_k"],
            illustration_path="/static/illustrations/content_xxx.png",
        )
        result = export_graph(g, tmp_path / "g.json")
        c = result["nodes"][0]
        assert c["raw_excerpt"] == "softmax(QK^T / sqrt(d_k)) V"
        assert c["key_terms"] == ["softmax", "QK^T", "d_k"]
        assert c["illustration_path"] == "/static/illustrations/content_xxx.png"


# ---------------------------------------------------------------------------
# Edge fields — structural vs. RELATED_TO
# ---------------------------------------------------------------------------

class TestEdgeFields:
    def test_structural_edge_has_relation(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        has_sub = next(e for e in result["edges"] if e["relation"] == "HAS_SUBTOPIC")
        assert has_sub["source"] == "topic:aaa111"
        assert has_sub["target"] == "subtopic:bbb222"

    def test_structural_edge_has_no_label_fields(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        has_content = next(e for e in result["edges"] if e["relation"] == "HAS_CONTENT")
        assert "label" not in has_content
        assert "confidence" not in has_content

    def test_related_to_carries_label_and_confidence(self, tmp_path):
        result = export_graph(_hierarchical_graph(), tmp_path / "g.json")
        rel = next(e for e in result["edges"] if e["relation"] == "RELATED_TO")
        assert rel["label"] == "uses"
        assert rel["confidence"] == 0.75


# ---------------------------------------------------------------------------
# File output + edge cases
# ---------------------------------------------------------------------------

class TestFileOutput:
    def test_file_is_created(self, tmp_path):
        out = tmp_path / "out.json"
        export_graph(_hierarchical_graph(), out)
        assert out.exists()

    def test_file_is_valid_json(self, tmp_path):
        out = tmp_path / "out.json"
        export_graph(_hierarchical_graph(), out)
        data = json.loads(out.read_text())
        assert "nodes" in data and "edges" in data

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "graph.json"
        export_graph(_hierarchical_graph(), out)
        assert out.exists()

    def test_return_value_matches_file_content(self, tmp_path):
        out = tmp_path / "g.json"
        result = export_graph(_hierarchical_graph(), out)
        file_data = json.loads(out.read_text())
        assert result["nodes"] == file_data["nodes"]
        assert result["edges"] == file_data["edges"]


class TestEdgeCases:
    def test_empty_graph_exports_cleanly(self, tmp_path):
        g = nx.MultiDiGraph()
        result = export_graph(g, tmp_path / "g.json")
        assert result == {"nodes": [], "edges": []}

    def test_isolated_node_exported(self, tmp_path):
        g = nx.MultiDiGraph()
        g.add_node("topic:xxx", name="Orphan", node_type="topic", source_ids=[])
        result = export_graph(g, tmp_path / "g.json")
        assert len(result["nodes"]) == 1
        assert result["edges"] == []
