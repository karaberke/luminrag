"""Editing inverses preserve graph evidence and multigraph relationships."""
from copy import deepcopy

import networkx as nx

from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import ContentAttrs, TopicAttrs
from backend.utils.associations import association_pair
from backend.utils.graph_ops import apply_op, note_chunks


def _snapshot(graph):
    return {"nodes": {key: deepcopy(attrs) for key, attrs in graph.nodes(data=True)}, "edges": {(u, v, k): deepcopy(attrs) for u, v, k, attrs in graph.edges(keys=True, data=True)}}


def test_merge_deduplicates_edges_and_unions_evidence(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    parent = builder.add_node(TopicAttrs(name="Parent"))
    keep = builder.add_node(ContentAttrs(name="Keep", source_ids=["one"], key_terms=["A"]))
    drop = builder.add_node(ContentAttrs(name="Drop", source_ids=["two"], key_terms=["B"]))
    builder.add_related_to(parent, keep, "explains", source_chunk_ids=["one"])
    builder.add_related_to(parent, drop, "explains", source_chunk_ids=["two"])
    before = _snapshot(builder.graph)
    inverse = apply_op(builder, {"op": "merge_nodes", "keep": keep, "drop": drop})
    assert drop not in builder.graph
    assert builder.graph.nodes[keep]["source_ids"] == ["one", "two"]
    assert builder.graph.nodes[keep]["key_terms"] == ["A", "B"]
    pair = association_pair(parent, keep)
    assert builder.graph.number_of_edges(*pair) == 1
    association = next(iter(builder.graph.get_edge_data(*pair).values()))
    assert association["source_chunk_ids"] == ["one", "two"]
    assert len(association["supports"]) == 2
    apply_op(builder, inverse)
    assert _snapshot(builder.graph) == before


def test_note_wikilinks_and_edit_undo(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    topic = builder.add_node(TopicAttrs(name="Graph Theory"))
    inverse = apply_op(builder, {"op": "add_node", "node": {"id": "my-note", "node_type": "note", "name": "Study", "body": "Read [[graph theory|the chapter]]."}})
    pair = association_pair("my-note", topic)
    assert builder.graph.has_edge(*pair)
    association = next(iter(builder.graph.get_edge_data(*pair).values()))
    assert association["label"] is None
    assert association["supports"][0]["kind"] == "wikilink"
    assert association["supports"][0]["owner"] == "my-note"
    chunks = note_chunks(builder.graph)
    assert chunks[0].modality == "note" and "Read" in chunks[0].text
    before = _snapshot(builder.graph)
    edit_inverse = apply_op(builder, {"op": "update_node", "id": "my-note", "updates": {"body": "No links now"}})
    assert not builder.graph.has_edge(*pair)
    apply_op(builder, edit_inverse)
    assert _snapshot(builder.graph) == before
    apply_op(builder, inverse)
    assert "my-note" not in builder.graph


def test_position_inverse_removes_previously_unset_positions(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    key = builder.add_node(TopicAttrs(name="Graph"))
    before = _snapshot(builder.graph)
    inverse = apply_op(builder, {"op": "set_positions", "positions": [{"id": key, "x": 5, "y": 12}]})
    apply_op(builder, inverse)
    assert _snapshot(builder.graph) == before
