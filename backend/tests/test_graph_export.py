"""
Tests for backend/graph/graph_export.py

No mocking needed — pure NetworkX + JSON, no external services.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from backend.graph.graph_export import export_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("enzyme", name="Enzyme", source_ids=["chunk_0", "chunk_1"])
    g.add_node("reaction_rate", name="ReactionRate", source_ids=["chunk_0"])
    g.add_edge(
        "enzyme", "reaction_rate",
        relation="CAUSES",
        source_chunk_ids=["chunk_0"],
    )
    return g


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestExportStructure:
    def test_output_has_nodes_key(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "graph.json")
        assert "nodes" in result

    def test_output_has_edges_key(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "graph.json")
        assert "edges" in result

    def test_node_count_matches(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "graph.json")
        assert len(result["nodes"]) == 2

    def test_edge_count_matches(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "graph.json")
        assert len(result["edges"]) == 1


# ---------------------------------------------------------------------------
# Node fields
# ---------------------------------------------------------------------------

class TestNodeFields:
    def test_each_node_has_id(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("id" in n for n in result["nodes"])

    def test_each_node_has_name(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("name" in n for n in result["nodes"])

    def test_each_node_has_source_ids(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("source_ids" in n for n in result["nodes"])

    def test_node_name_matches_display_name(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        names = {n["id"]: n["name"] for n in result["nodes"]}
        assert names["enzyme"] == "Enzyme"
        assert names["reaction_rate"] == "ReactionRate"

    def test_node_source_ids_correct(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        enzyme_node = next(n for n in result["nodes"] if n["id"] == "enzyme")
        assert "chunk_0" in enzyme_node["source_ids"]
        assert "chunk_1" in enzyme_node["source_ids"]


# ---------------------------------------------------------------------------
# Edge fields
# ---------------------------------------------------------------------------

class TestEdgeFields:
    def test_each_edge_has_source(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("source" in e for e in result["edges"])

    def test_each_edge_has_target(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("target" in e for e in result["edges"])

    def test_each_edge_has_relation(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("relation" in e for e in result["edges"])

    def test_each_edge_has_source_chunk_ids(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert all("source_chunk_ids" in e for e in result["edges"])

    def test_edge_relation_value(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        assert result["edges"][0]["relation"] == "CAUSES"

    def test_edge_source_target_keys(self, tmp_path):
        result = export_graph(_simple_graph(), tmp_path / "g.json")
        edge = result["edges"][0]
        assert edge["source"] == "enzyme"
        assert edge["target"] == "reaction_rate"

    def test_multiple_edges_between_same_nodes(self, tmp_path):
        g = nx.MultiDiGraph()
        g.add_node("a", name="A", source_ids=[])
        g.add_node("b", name="B", source_ids=[])
        g.add_edge("a", "b", relation="CAUSES", source_chunk_ids=["c0"])
        g.add_edge("a", "b", relation="EXPLAINS", source_chunk_ids=["c1"])

        result = export_graph(g, tmp_path / "g.json")
        assert len(result["edges"]) == 2


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

class TestFileOutput:
    def test_file_is_created(self, tmp_path):
        out = tmp_path / "out.json"
        export_graph(_simple_graph(), out)
        assert out.exists()

    def test_file_is_valid_json(self, tmp_path):
        out = tmp_path / "out.json"
        export_graph(_simple_graph(), out)
        data = json.loads(out.read_text())
        assert "nodes" in data and "edges" in data

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "graph.json"
        export_graph(_simple_graph(), out)
        assert out.exists()

    def test_return_value_matches_file_content(self, tmp_path):
        out = tmp_path / "g.json"
        result = export_graph(_simple_graph(), out)
        file_data = json.loads(out.read_text())
        assert result["nodes"] == file_data["nodes"]
        assert result["edges"] == file_data["edges"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_graph_exports_cleanly(self, tmp_path):
        g = nx.MultiDiGraph()
        result = export_graph(g, tmp_path / "g.json")
        assert result == {"nodes": [], "edges": []}

    def test_isolated_node_exported(self, tmp_path):
        g = nx.MultiDiGraph()
        g.add_node("orphan", name="Orphan", source_ids=["c0"])
        result = export_graph(g, tmp_path / "g.json")
        assert len(result["nodes"]) == 1
        assert result["edges"] == []
