"""
Tests for backend/graph/graph_builder.py

No mocking needed — NetworkX is pure Python with no external I/O beyond
the JSON file, which is written to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.graph.graph_builder import GraphBuilder
from backend.schemas import GraphTriple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triple(
    head="Enzyme",
    relation="CAUSES",
    tail="ReactionRate",
    chunk_ids=None,
) -> GraphTriple:
    return GraphTriple(
        head=head,
        relation=relation,
        tail=tail,
        source_chunk_ids=chunk_ids or ["chunk_0"],
    )


@pytest.fixture
def builder(tmp_path: Path) -> GraphBuilder:
    return GraphBuilder(tmp_path / "graph.json")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_starts_empty(self, builder):
        assert builder.node_count() == 0
        assert builder.edge_count() == 0

    def test_loads_existing_graph_on_init(self, tmp_path):
        path = tmp_path / "graph.json"
        b1 = GraphBuilder(path)
        b1.add_triples([_triple()])
        b1.save()

        b2 = GraphBuilder(path)
        assert b2.node_count() == 2


# ---------------------------------------------------------------------------
# add_triples — nodes
# ---------------------------------------------------------------------------

class TestAddTriplesNodes:
    def test_adds_head_and_tail_nodes(self, builder):
        builder.add_triples([_triple(head="Enzyme", tail="ReactionRate")])
        assert builder.node_count() == 2

    def test_node_name_preserved(self, builder):
        builder.add_triples([_triple(head="Michaelis-Menten")])
        node = builder.get_node("Michaelis-Menten")
        assert node["name"] == "Michaelis-Menten"

    def test_node_key_is_normalised(self, builder):
        builder.add_triples([_triple(head="  Enzyme  ")])
        assert builder.get_node("enzyme") is not None

    def test_duplicate_node_not_created(self, builder):
        builder.add_triples([
            _triple(head="Enzyme", tail="Rate"),
            _triple(head="enzyme", tail="Rate"),  # same node, different case
        ])
        assert builder.node_count() == 2  # Enzyme + Rate

    def test_source_ids_accumulated_on_node(self, builder):
        builder.add_triples([_triple(head="Enzyme", chunk_ids=["c0"])])
        builder.add_triples([_triple(head="Enzyme", chunk_ids=["c1"])])
        node = builder.get_node("Enzyme")
        assert "c0" in node["source_ids"]
        assert "c1" in node["source_ids"]

    def test_source_ids_not_duplicated(self, builder):
        builder.add_triples([_triple(chunk_ids=["c0"])])
        builder.add_triples([_triple(chunk_ids=["c0"])])
        node = builder.get_node("Enzyme")
        assert node["source_ids"].count("c0") == 1


# ---------------------------------------------------------------------------
# add_triples — edges
# ---------------------------------------------------------------------------

class TestAddTriplesEdges:
    def test_adds_edge(self, builder):
        builder.add_triples([_triple()])
        assert builder.edge_count() == 1

    def test_same_relation_triple_merged(self, builder):
        builder.add_triples([_triple(chunk_ids=["c0"])])
        builder.add_triples([_triple(chunk_ids=["c1"])])
        assert builder.edge_count() == 1  # merged, not duplicated

    def test_merged_edge_accumulates_chunk_ids(self, builder):
        builder.add_triples([_triple(chunk_ids=["c0"])])
        builder.add_triples([_triple(chunk_ids=["c1"])])
        neighbours = builder.get_neighbours("Enzyme")
        assert "c0" in neighbours[0]["source_chunk_ids"]
        assert "c1" in neighbours[0]["source_chunk_ids"]

    def test_different_relations_kept_as_separate_edges(self, builder):
        builder.add_triples([_triple(relation="CAUSES")])
        builder.add_triples([_triple(relation="EXPLAINS")])
        assert builder.edge_count() == 2

    def test_edge_relation_stored_correctly(self, builder):
        builder.add_triples([_triple(relation="PREREQUISITE")])
        neighbours = builder.get_neighbours("Enzyme")
        assert neighbours[0]["relation"] == "PREREQUISITE"

    def test_multiple_distinct_triples(self, builder):
        triples = [
            _triple("A", "CAUSES", "B"),
            _triple("B", "PART_OF", "C"),
            _triple("C", "EXPLAINS", "A"),
        ]
        builder.add_triples(triples)
        assert builder.node_count() == 3
        assert builder.edge_count() == 3


# ---------------------------------------------------------------------------
# get_neighbours
# ---------------------------------------------------------------------------

class TestGetNeighbours:
    def test_returns_outgoing_edges(self, builder):
        builder.add_triples([_triple("Enzyme", "CAUSES", "Rate")])
        neighbours = builder.get_neighbours("Enzyme")
        assert len(neighbours) == 1
        assert neighbours[0]["target"] == "Rate"

    def test_unknown_node_returns_empty(self, builder):
        assert builder.get_neighbours("DoesNotExist") == []

    def test_only_outgoing_edges_returned(self, builder):
        builder.add_triples([_triple("A", "CAUSES", "B")])
        # B has no outgoing edges
        assert builder.get_neighbours("B") == []


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        b = GraphBuilder(tmp_path / "g.json")
        b.add_triples([_triple()])
        b.save()
        assert (tmp_path / "g.json").exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        b = GraphBuilder(tmp_path / "g.json")
        b.add_triples([_triple()])
        b.save()
        data = json.loads((tmp_path / "g.json").read_text())
        assert "nodes" in data
        # NetworkX 3.x uses "edges"; older versions used "links"
        assert "edges" in data or "links" in data

    def test_round_trip_preserves_node_count(self, tmp_path):
        path = tmp_path / "g.json"
        b1 = GraphBuilder(path)
        b1.add_triples([_triple("A", "CAUSES", "B"), _triple("B", "PART_OF", "C")])
        b1.save()

        b2 = GraphBuilder(path)
        assert b2.node_count() == b1.node_count()

    def test_round_trip_preserves_edge_count(self, tmp_path):
        path = tmp_path / "g.json"
        b1 = GraphBuilder(path)
        b1.add_triples([_triple("A", "CAUSES", "B"), _triple("B", "PART_OF", "C")])
        b1.save()

        b2 = GraphBuilder(path)
        assert b2.edge_count() == b1.edge_count()

    def test_round_trip_preserves_source_ids(self, tmp_path):
        path = tmp_path / "g.json"
        b1 = GraphBuilder(path)
        b1.add_triples([_triple(chunk_ids=["chunk_99"])])
        b1.save()

        b2 = GraphBuilder(path)
        node = b2.get_node("Enzyme")
        assert "chunk_99" in node["source_ids"]

    def test_accumulates_across_save_load_cycles(self, tmp_path):
        path = tmp_path / "g.json"

        b1 = GraphBuilder(path)
        b1.add_triples([_triple("A", "CAUSES", "B")])
        b1.save()

        b2 = GraphBuilder(path)  # loads existing
        b2.add_triples([_triple("C", "PART_OF", "D")])
        b2.save()

        b3 = GraphBuilder(path)
        assert b3.node_count() == 4
        assert b3.edge_count() == 2
