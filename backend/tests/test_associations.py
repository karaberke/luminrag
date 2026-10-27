"""Plain associations preserve independent grounding through edits and migration."""
from copy import deepcopy
import json
from unittest.mock import Mock

import networkx as nx
import pytest

from backend.graph.graph_builder import GraphBuilder
from backend.graph.graph_export import export_graph
from backend.graph.schema import ContentAttrs, RelatedToProposal, ProposalBundle, TopicAttrs
from backend.utils.associations import (
    AssociationSupport, add_association, association_id, normalize_associations,
    prune_association_supports, remove_association_supports,
)
from backend.utils.graph_ops import apply_op, note_chunk_id
from backend.utils import graph_ops
from backend.utils.budgets import JsonWorkBudget


def graph():
    value = nx.MultiDiGraph()
    value.add_node("a", name="A", node_type="content", x=2.0, y=3.0)
    value.add_node("b", name="B", node_type="content")
    return value


def backing(value):
    return value.edges["a", "b", association_id("a", "b")]


def test_symmetric_id_and_one_association_with_independent_backing():
    value = graph()
    assert association_id("a", "b") == association_id("b", "a") == "assoc:0473ef2dc0d324ab659d3580"
    assert add_association(value, "a", "b", kind="automatic", source_chunk_ids=["one", "two"], confidence=.8, verified=True)
    assert not add_association(value, "b", "a", kind="manual")
    assert value.number_of_edges() == 1
    assert backing(value)["label"] is None
    assert [item["kind"] for item in backing(value)["supports"]] == ["automatic", "manual"]
    prune_association_supports(value, {"one"})
    assert [item["kind"] for item in backing(value)["supports"]] == ["manual"]


def test_legacy_reverse_labels_migrate_idempotently_without_losing_hierarchy_or_positions(tmp_path):
    value = graph()
    value.add_edge("a", "b", relation="HAS_CONTENT", source_chunk_ids=["hierarchy"])
    value.add_edge("a", "b", relation="RELATED_TO", label="supports", source_chunk_ids=["one"], confidence=.6)
    value.add_edge("b", "a", relation="RELATED_TO", label="explains", source_chunk_ids=["two"], confidence=.8)
    assert normalize_associations(value) == 1
    before = deepcopy(value)
    assert normalize_associations(value) == 0
    assert nx.utils.graphs_equal(value, before)
    assert value.nodes["a"]["x"] == 2.0
    assert value.number_of_edges() == 2
    assert all(not item["verified"] and item["kind"] == "legacy" for item in backing(value)["supports"])
    result = export_graph(value, tmp_path / "export.json")
    edge = next(item for item in result["edges"] if item["relation"] == "RELATED_TO")
    assert result["association_version"] == 1 and edge["id"] == association_id("a", "b")
    assert edge["label"] is None and edge["supports"] == backing(value)["supports"]


def test_duplicate_backing_merges_confidence_without_accumulating_records():
    value = graph()
    add_association(value, "a", "b", kind="automatic", source_chunk_ids=["two", "one"], confidence=.5, verified=True)
    add_association(value, "b", "a", kind="automatic", source_chunk_ids=["one", "two"], confidence=.9, verified=True)
    assert len(backing(value)["supports"]) == 1
    assert backing(value)["confidence"] == .9


def test_automatic_grounding_group_is_removed_if_either_required_passage_is_removed():
    value = graph()
    add_association(value, "a", "b", kind="automatic", source_chunk_ids=["a1", "b1"], confidence=.8, verified=True)
    add_association(value, "a", "b", kind="automatic", source_chunk_ids=["a2", "b2"], confidence=.7, verified=True)
    prune_association_supports(value, {"a1"})
    assert backing(value)["source_chunk_ids"] == ["a2", "b2"]
    prune_association_supports(value, {"b2"})
    assert value.number_of_edges() == 0


def test_manual_and_unattributed_legacy_connections_survive_unrelated_source_cleanup():
    value = graph()
    add_association(value, "a", "b", kind="legacy")
    add_association(value, "b", "a", kind="manual", source_chunk_ids=["one"])
    prune_association_supports(value, {"one"})
    assert {item["kind"] for item in backing(value)["supports"]} == {"legacy", "manual"}
    assert backing(value)["source_chunk_ids"] == []


@pytest.mark.parametrize("patch", [{"kind": "unknown"}, {"verified": "false"}, {"confidence": float("nan")},
                                    {"source_chunk_ids": "one"}, {"source_chunk_ids": [3]}, {"owner": 3},
                                    {"verified": True}, {"kind": "automatic", "verified": True},
                                    {"source_chunk_ids": ["x" * 1025]}])
def test_new_backing_records_fail_closed(patch):
    record = {"kind": "manual", "owner": None, "source_chunk_ids": [], "confidence": .8, "verified": False, **patch}
    with pytest.raises(ValueError):
        AssociationSupport.model_validate(record)


def test_chunk_ids_may_be_1024_characters():
    assert AssociationSupport.model_validate({"kind": "automatic", "source_chunk_ids": ["x" * 1024]}).source_chunk_ids


def test_note_edit_invalidates_verified_backing_keeps_manual_and_undo_restores_everything(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    target = builder.add_node(TopicAttrs(name="Graph Theory"))
    apply_op(builder, {"op": "add_node", "node": {"id": "my-note", "node_type": "note", "name": "Study", "body": "Read [[Graph Theory]]."}})
    builder.add_related_to(target, "my-note", kind="automatic", source_chunk_ids=["source", note_chunk_id("my-note")], verified=True)
    builder.add_related_to(target, "my-note", kind="manual")
    before = deepcopy(builder.graph)
    inverse = apply_op(builder, {"op": "update_node", "id": "my-note", "updates": {"body": "No links now"}})
    attrs = next(attrs for _, _, attrs in builder.graph.edges(data=True))
    assert {record["kind"] for record in attrs["supports"]} == {"manual"}
    apply_op(builder, inverse)
    assert nx.utils.graphs_equal(builder.graph, before)


def test_note_edit_keeps_current_wikilink_and_manual_backing(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    target = builder.add_node(TopicAttrs(name="Graph Theory"))
    apply_op(builder, {"op": "add_node", "node": {"id": "note", "name": "Study", "body": "[[Graph Theory]]"}})
    builder.add_related_to(target, "note", kind="automatic", source_chunk_ids=["source", "note:note"], verified=True)
    builder.add_related_to(target, "note", kind="manual")
    inverse = apply_op(builder, {"op": "update_node", "id": "note", "updates": {"body": "Read [[Graph Theory]] today."}})
    attrs = next(iter(builder.graph.edges(data=True)))[2]
    assert {record["kind"] for record in attrs["supports"]} == {"manual", "wikilink"}
    redo = apply_op(builder, inverse)
    assert {record["kind"] for record in next(iter(builder.graph.edges(data=True)))[2]["supports"]} == {"automatic", "manual", "wikilink"}
    apply_op(builder, redo)
    assert {record["kind"] for record in next(iter(builder.graph.edges(data=True)))[2]["supports"]} == {"manual", "wikilink"}


def test_nested_backing_bytes_are_reserved_before_snapshot_copy(monkeypatch):
    value = graph()
    add_association(value, "a", "b", source_chunk_ids=[str(index) + "x" * 900 for index in range(4)])
    copy = Mock(wraps=deepcopy)
    monkeypatch.setattr(graph_ops, "deepcopy", copy)
    with pytest.raises(ValueError, match="undo processing limit"):
        graph_ops._edge_snapshot(value, "a", "b", JsonWorkBudget(max_items=100, max_bytes=4000, copies=3), symmetric=True)
    copy.assert_not_called()


def test_legacy_null_confidence_is_unverified_fallback():
    value = graph()
    value.add_edge("a", "b", relation="RELATED_TO", label="supports", confidence=None)
    normalize_associations(value)
    assert backing(value)["confidence"] == .5
    assert backing(value)["supports"][0]["verified"] is False


def test_merge_notes_invalidates_changed_passage_checks_and_preserves_independent_backing(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    target = builder.add_node(TopicAttrs(name="Graph Theory"))
    for key, body in (("keep", "[[Graph Theory]]"), ("drop", "Longer explanation about [[Graph Theory]].")):
        apply_op(builder, {"op": "add_node", "node": {"id": key, "name": key, "body": body}})
        builder.add_related_to(target, key, kind="automatic", source_chunk_ids=["source", "note:" + key], verified=True)
        builder.add_related_to(target, key, kind="manual")
    # Independent passages remain checked after the explicit concept merge.
    builder.add_related_to(target, "drop", kind="automatic", source_chunk_ids=["a", "b"], verified=True)
    before = deepcopy(builder.graph)
    inverse = apply_op(builder, {"op": "merge_nodes", "keep": "keep", "drop": "drop"})
    assert "drop" not in builder.graph and "Longer" in builder.graph.nodes["keep"]["body"]
    attrs = next(iter(builder.graph.edges(data=True)))[2]
    assert {record["kind"] for record in attrs["supports"]} == {"automatic", "manual", "wikilink"}
    assert [record["source_chunk_ids"] for record in attrs["supports"] if record["verified"]] == [["a", "b"]]
    redo = apply_op(builder, inverse)
    assert nx.utils.graphs_equal(builder.graph, before)
    apply_op(builder, redo)
    assert [record["source_chunk_ids"] for record in next(iter(builder.graph.edges(data=True)))[2]["supports"] if record["verified"]] == [["a", "b"]]


def test_delete_reverse_connection_and_undo_restore_all_backings_and_hierarchy(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    a = builder.add_node(TopicAttrs(name="A"))
    b = builder.add_node(ContentAttrs(name="B"))
    builder.add_structural_edge(a, "HAS_CONTENT", b)
    builder.add_related_to(a, b, kind="automatic", source_chunk_ids=["one", "two"], verified=True)
    builder.add_related_to(b, a, kind="manual")
    before = deepcopy(builder.graph)
    inverse = apply_op(builder, {"op": "delete_edge", "source": b, "target": a, "relation": "RELATED_TO", "label": "old verb"})
    assert builder.edge_count() == 1
    apply_op(builder, inverse)
    assert nx.utils.graphs_equal(builder.graph, before)


def test_verified_proposals_persist_and_round_trip_without_any_node_merge(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    a = builder.add_node(ContentAttrs(name="Activation in neurons"))
    b = builder.add_node(ContentAttrs(name="Neural inhibition"))
    builder.apply_proposals(ProposalBundle(related=[RelatedToProposal(source_key=a, target_key=b, verified=True,
        confidence=.85, source_chunk_ids=["one", "two"])]))
    builder.save()
    reloaded = GraphBuilder(builder._path)
    assert reloaded.node_count() == 2
    support = next(iter(reloaded.graph.edges(data=True)))[2]["supports"][0]
    assert support["kind"] == "automatic" and support["verified"]
    assert "label" in json.loads(builder._path.read_text())["edges"][0]


def test_legacy_note_edge_is_separate_backing_without_display_predicate():
    value = graph()
    value.nodes["a"]["node_type"] = "note"
    value.add_edge("a", "b", relation="RELATED_TO", label="links_to", wikilink=True)
    value.add_edge("b", "a", relation="RELATED_TO", label="explains")
    normalize_associations(value)
    remove_association_supports(value, kind="wikilink", owner="a")
    assert len(backing(value)["supports"]) == 1 and backing(value)["supports"][0]["kind"] == "legacy"
