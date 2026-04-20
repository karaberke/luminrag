"""
Tests for backend/graph/graph_builder.py

Covers the hierarchical schema pipeline: typed node insertion, structural +
semantic edges, polyhierarchy, idempotent re-apply, prune_disconnected, and
JSON round-trip. NetworkX is pure-Python — no external I/O beyond tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import (
    ContentAttrs,
    ContentProposal,
    ProposalBundle,
    RelatedToProposal,
    SubtopicAttrs,
    SubtopicProposal,
    TopicAttrs,
    TopicProposal,
    make_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def builder(tmp_path: Path) -> GraphBuilder:
    return GraphBuilder(tmp_path / "graph.json")


def _bundle(
    *,
    topics: list[TopicProposal] | None = None,
    subtopics: list[SubtopicProposal] | None = None,
    contents: list[ContentProposal] | None = None,
    related: list[RelatedToProposal] | None = None,
) -> ProposalBundle:
    return ProposalBundle(
        topics=topics or [],
        subtopics=subtopics or [],
        contents=contents or [],
        related=related or [],
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_starts_empty(self, builder):
        assert builder.node_count() == 0
        assert builder.edge_count() == 0

    def test_loads_existing_graph_on_init(self, tmp_path):
        path = tmp_path / "g.json"
        b1 = GraphBuilder(path)
        b1.add_node(TopicAttrs(name="NLP"))
        b1.save()

        b2 = GraphBuilder(path)
        assert b2.node_count() == 1


# ---------------------------------------------------------------------------
# Low-level add_node
# ---------------------------------------------------------------------------

class TestAddNode:
    def test_creates_topic(self, builder):
        key = builder.add_node(TopicAttrs(name="Graph Theory"))
        assert key == make_key("topic", "Graph Theory")
        attrs = builder.get_node(key)
        assert attrs["name"] == "Graph Theory"
        assert attrs["node_type"] == "topic"

    def test_keys_are_stable_across_casing(self, builder):
        k1 = builder.add_node(TopicAttrs(name="Graph Theory"))
        k2 = builder.add_node(TopicAttrs(name="graph theory"))
        assert k1 == k2
        assert builder.node_count() == 1

    def test_merge_keeps_longer_summary(self, builder):
        builder.add_node(TopicAttrs(name="NLP", summary="short"))
        builder.add_node(TopicAttrs(name="NLP", summary="a longer, richer summary"))
        attrs = builder.get_node(make_key("topic", "NLP"))
        assert attrs["summary"] == "a longer, richer summary"

    def test_merge_accumulates_source_ids(self, builder):
        builder.add_node(TopicAttrs(name="NLP", source_ids=["c1"]))
        builder.add_node(TopicAttrs(name="NLP", source_ids=["c2", "c1"]))
        attrs = builder.get_node(make_key("topic", "NLP"))
        assert attrs["source_ids"] == ["c1", "c2"]

    def test_content_takes_longer_summary(self, builder):
        builder.add_node(ContentAttrs(name="Self-Attention", summary="short"))
        builder.add_node(ContentAttrs(name="Self-Attention", summary="a much longer and richer explanation"))
        attrs = builder.get_node(make_key("content", "Self-Attention"))
        assert attrs["summary"] == "a much longer and richer explanation"

    def test_content_merge_keeps_longer_raw_excerpt(self, builder):
        builder.add_node(ContentAttrs(name="Attn", raw_excerpt="short quote"))
        builder.add_node(ContentAttrs(
            name="Attn", raw_excerpt="a much longer verbatim source quote here"
        ))
        attrs = builder.get_node(make_key("content", "Attn"))
        assert attrs["raw_excerpt"] == "a much longer verbatim source quote here"

    def test_content_merge_unions_key_terms_case_insensitive(self, builder):
        builder.add_node(ContentAttrs(name="Attn", key_terms=["Vmax", "Km"]))
        builder.add_node(ContentAttrs(name="Attn", key_terms=["vmax", "softmax"]))
        attrs = builder.get_node(make_key("content", "Attn"))
        # Case-insensitive dedup preserves first-seen casing + insertion order
        assert attrs["key_terms"] == ["Vmax", "Km", "softmax"]

    def test_illustration_path_not_overwritten_when_already_set(self, builder):
        builder.add_node(ContentAttrs(
            name="Attn", illustration_path="/static/illustrations/attn.png"
        ))
        # A re-ingest without a path must not clear the existing one
        builder.add_node(ContentAttrs(name="Attn", illustration_path=None))
        attrs = builder.get_node(make_key("content", "Attn"))
        assert attrs["illustration_path"] == "/static/illustrations/attn.png"

    def test_set_illustration_path_updates_existing_content_node(self, builder):
        ckey = builder.add_node(ContentAttrs(name="Attn"))
        assert builder.set_illustration_path(ckey, "/static/illustrations/attn.png") is True
        assert builder.get_node(ckey)["illustration_path"] == "/static/illustrations/attn.png"

    def test_set_illustration_path_returns_false_when_missing(self, builder):
        assert builder.set_illustration_path("content:nope", "/x.png") is False


# ---------------------------------------------------------------------------
# Structural edges
# ---------------------------------------------------------------------------

class TestStructuralEdges:
    def test_has_subtopic(self, builder):
        tkey = builder.add_node(TopicAttrs(name="NLP"))
        skey = builder.add_node(SubtopicAttrs(name="Transformers"))
        assert builder.add_structural_edge(tkey, "HAS_SUBTOPIC", skey) is True
        assert builder.edge_count() == 1

    def test_merges_duplicate_structural_edge(self, builder):
        tkey = builder.add_node(TopicAttrs(name="NLP"))
        skey = builder.add_node(SubtopicAttrs(name="Transformers"))
        builder.add_structural_edge(tkey, "HAS_SUBTOPIC", skey, source_chunk_ids=["c1"])
        added = builder.add_structural_edge(tkey, "HAS_SUBTOPIC", skey, source_chunk_ids=["c2"])
        assert added is False
        edges = builder.graph.get_edge_data(tkey, skey)
        assert list(edges.values())[0]["source_chunk_ids"] == ["c1", "c2"]

    def test_rejects_missing_endpoints(self, builder):
        with pytest.raises(ValueError, match="missing node"):
            builder.add_structural_edge("topic:aaa", "HAS_SUBTOPIC", "topic:bbb")

    def test_rejects_unknown_structural_relation(self, builder):
        tkey = builder.add_node(TopicAttrs(name="NLP"))
        skey = builder.add_node(SubtopicAttrs(name="Transformers"))
        with pytest.raises(ValueError, match="Not a structural relation"):
            builder.add_structural_edge(tkey, "PART_OF", skey)

    def test_rejects_evidence_of(self, builder):
        """EVIDENCE_OF is no longer a valid relation."""
        tkey = builder.add_node(TopicAttrs(name="NLP"))
        ckey = builder.add_node(ContentAttrs(name="Attention"))
        with pytest.raises(ValueError, match="Not a structural relation"):
            builder.add_structural_edge(tkey, "EVIDENCE_OF", ckey)


# ---------------------------------------------------------------------------
# RELATED_TO semantic edges
# ---------------------------------------------------------------------------

class TestRelatedTo:
    def test_adds_related_to(self, builder):
        a = builder.add_node(TopicAttrs(name="NLP"))
        b = builder.add_node(TopicAttrs(name="Graph Theory"))
        assert builder.add_related_to(a, b, "uses", confidence=0.8) is True
        edges = builder.graph.get_edge_data(a, b)
        data = list(edges.values())[0]
        assert data["relation"] == "RELATED_TO"
        assert data["label"] == "uses"
        assert data["confidence"] == 0.8

    def test_normalises_label(self, builder):
        a = builder.add_node(TopicAttrs(name="NLP"))
        b = builder.add_node(TopicAttrs(name="Graph Theory"))
        builder.add_related_to(a, b, "Used In!")
        data = list(builder.graph.get_edge_data(a, b).values())[0]
        assert data["label"] == "used_in"

    def test_merges_same_label_max_confidence(self, builder):
        a = builder.add_node(TopicAttrs(name="NLP"))
        b = builder.add_node(TopicAttrs(name="Graph Theory"))
        builder.add_related_to(a, b, "uses", confidence=0.4)
        merged = builder.add_related_to(a, b, "uses", confidence=0.7)
        assert merged is False
        data = list(builder.graph.get_edge_data(a, b).values())[0]
        assert data["confidence"] == 0.7

    def test_different_labels_kept_as_parallel_edges(self, builder):
        a = builder.add_node(TopicAttrs(name="NLP"))
        b = builder.add_node(TopicAttrs(name="Graph Theory"))
        builder.add_related_to(a, b, "uses")
        builder.add_related_to(a, b, "generalises")
        assert len(builder.graph.get_edge_data(a, b)) == 2


# ---------------------------------------------------------------------------
# apply_proposals — full pipeline
# ---------------------------------------------------------------------------

class TestApplyProposals:
    def test_end_to_end_three_node_types(self, builder):
        bundle = _bundle(
            topics=[TopicProposal(name="NLP", summary="field", source_chunk_ids=["c1"])],
            subtopics=[SubtopicProposal(
                name="Transformers", parent_topic_names=["NLP"], source_chunk_ids=["c1"]
            )],
            contents=[ContentProposal(
                title="Self-Attention",
                content_type="definition",
                summary="attention formula explanation",
                parent_subtopic_names=["Transformers"],
                parent_topic_names=["NLP"],
                evidence_chunk_ids=["c1"],
            )],
        )
        result = builder.apply_proposals(bundle)
        assert result.topics_added == 1
        assert result.subtopics_added == 1
        assert result.contents_added == 1
        # 2 structural edges: HAS_SUBTOPIC + HAS_CONTENT
        assert result.structural_edges_added == 2
        assert builder.node_count() == 3

    def test_polyhierarchy_content_under_two_subtopics(self, builder):
        bundle = _bundle(
            topics=[TopicProposal(name="NLP")],
            subtopics=[
                SubtopicProposal(name="Transformers", parent_topic_names=["NLP"]),
                SubtopicProposal(name="Attention Mechanisms", parent_topic_names=["NLP"]),
            ],
            contents=[ContentProposal(
                title="Self-Attention",
                parent_subtopic_names=["Transformers", "Attention Mechanisms"],
                parent_topic_names=["NLP"],
            )],
        )
        builder.apply_proposals(bundle)
        content_key = make_key("content", "Self-Attention")
        in_edges = [
            (u, data["relation"])
            for u, _, data in builder.graph.in_edges(content_key, data=True)
        ]
        has_content_sources = [u for u, rel in in_edges if rel == "HAS_CONTENT"]
        assert len(has_content_sources) == 2

    def test_idempotent_reapply_does_not_duplicate(self, builder):
        bundle = _bundle(
            topics=[TopicProposal(name="NLP", source_chunk_ids=["c1"])],
            subtopics=[SubtopicProposal(name="Transformers", parent_topic_names=["NLP"])],
            contents=[ContentProposal(
                title="Self-Attention",
                parent_subtopic_names=["Transformers"],
                parent_topic_names=["NLP"],
                evidence_chunk_ids=["c1"],
            )],
        )
        builder.apply_proposals(bundle)
        n0, e0 = builder.node_count(), builder.edge_count()
        builder.apply_proposals(bundle)
        assert builder.node_count() == n0
        assert builder.edge_count() == e0

    def test_related_edges_applied(self, builder):
        a = builder.add_node(TopicAttrs(name="NLP"))
        b = builder.add_node(TopicAttrs(name="Graph Theory"))
        bundle = _bundle(related=[
            RelatedToProposal(source_key=a, target_key=b, label="uses", confidence=0.9),
        ])
        result = builder.apply_proposals(bundle)
        assert result.related_edges_added == 1
        assert builder.edge_count() == 1

    def test_related_to_missing_node_is_warned_not_raised(self, builder):
        bundle = _bundle(related=[
            RelatedToProposal(source_key="topic:missing1", target_key="topic:missing2",
                              label="uses"),
        ])
        result = builder.apply_proposals(bundle)
        assert result.related_edges_added == 0
        assert any("missing node" in w for w in result.warnings)

    def test_content_falls_back_to_topic_when_no_subtopic(self, builder):
        bundle = _bundle(
            topics=[TopicProposal(name="NLP")],
            contents=[ContentProposal(
                title="Tokenisation",
                parent_topic_names=["NLP"],
            )],
        )
        result = builder.apply_proposals(bundle)
        assert result.structural_edges_added == 1  # Topic → Content fallback


# ---------------------------------------------------------------------------
# prune_disconnected
# ---------------------------------------------------------------------------

class TestPruneDisconnected:
    def test_removes_isolated_nodes(self, builder):
        builder.add_node(TopicAttrs(name="Orphan"))
        tkey = builder.add_node(TopicAttrs(name="NLP"))
        skey = builder.add_node(SubtopicAttrs(name="Transformers"))
        builder.add_structural_edge(tkey, "HAS_SUBTOPIC", skey)
        assert builder.node_count() == 3
        removed = builder.prune_disconnected()
        assert removed == 1
        assert builder.node_count() == 2

    def test_no_nodes_removed_when_all_connected(self, builder):
        tkey = builder.add_node(TopicAttrs(name="NLP"))
        skey = builder.add_node(SubtopicAttrs(name="Transformers"))
        builder.add_structural_edge(tkey, "HAS_SUBTOPIC", skey)
        assert builder.prune_disconnected() == 0


# ---------------------------------------------------------------------------
# prune_to_ratio — enforce content > subtopics > topics
# ---------------------------------------------------------------------------

def _connected_pyramid(builder: GraphBuilder, n_topics: int, n_subs: int, n_contents: int):
    """Build a fully-connected Topic→Subtopic→Content pyramid."""
    topic_keys = []
    for i in range(n_topics):
        tkey = builder.add_node(TopicAttrs(name=f"Topic{i}"))
        topic_keys.append(tkey)

    sub_keys = []
    for i in range(n_subs):
        skey = builder.add_node(SubtopicAttrs(name=f"Sub{i}"))
        sub_keys.append(skey)
        # Attach each subtopic to the first topic (keeps all connected)
        builder.add_structural_edge(topic_keys[0], "HAS_SUBTOPIC", skey)

    for i in range(n_contents):
        ckey = builder.add_node(ContentAttrs(name=f"Content{i}"))
        # Attach each content to the first subtopic
        builder.add_structural_edge(sub_keys[0], "HAS_CONTENT", ckey)


class TestPruneToRatio:
    def test_already_valid_unchanged(self, builder):
        _connected_pyramid(builder, n_topics=1, n_subs=2, n_contents=3)
        removed = builder.prune_to_ratio()
        assert removed == {"topic": 0, "subtopic": 0}

    def test_too_many_topics_trimmed(self, builder):
        # 3 topics, 3 subtopics, 4 contents → violates subtopic > topic
        _connected_pyramid(builder, n_topics=3, n_subs=3, n_contents=4)
        builder.prune_to_ratio()
        n_t = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "topic")
        n_s = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "subtopic")
        assert n_t < n_s

    def test_too_many_subtopics_trimmed(self, builder):
        # 1 topic, 4 subtopics, 4 contents → violates content > subtopic
        _connected_pyramid(builder, n_topics=1, n_subs=4, n_contents=4)
        builder.prune_to_ratio()
        n_s = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "subtopic")
        n_c = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "content")
        assert n_s < n_c

    def test_both_violations_resolved(self, builder):
        # 5 topics, 5 subtopics, 5 contents → both rules violated
        _connected_pyramid(builder, n_topics=5, n_subs=5, n_contents=5)
        builder.prune_to_ratio()
        builder.prune_disconnected()
        n_t = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "topic")
        n_s = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "subtopic")
        n_c = sum(1 for _, a in builder.graph.nodes(data=True) if a.get("node_type") == "content")
        assert n_c > n_s
        assert n_s > n_t


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_node_cascades_edges(self, builder):
        t = builder.add_node(TopicAttrs(name="NLP"))
        s = builder.add_node(SubtopicAttrs(name="Transformers"))
        builder.add_structural_edge(t, "HAS_SUBTOPIC", s)
        assert builder.remove_node(t) is True
        assert builder.edge_count() == 0

    def test_remove_node_returns_false_when_missing(self, builder):
        assert builder.remove_node("topic:nope") is False

    def test_remove_edge_by_relation(self, builder):
        t = builder.add_node(TopicAttrs(name="NLP"))
        s = builder.add_node(SubtopicAttrs(name="Transformers"))
        builder.add_structural_edge(t, "HAS_SUBTOPIC", s)
        assert builder.remove_edge(t, "HAS_SUBTOPIC", s) is True
        assert builder.edge_count() == 0

    def test_remove_related_to_by_label(self, builder):
        a = builder.add_node(TopicAttrs(name="NLP"))
        b = builder.add_node(TopicAttrs(name="Graph Theory"))
        builder.add_related_to(a, b, "uses")
        builder.add_related_to(a, b, "generalises")
        assert builder.remove_edge(a, "RELATED_TO", b, label="uses") is True
        remaining = list(builder.graph.get_edge_data(a, b).values())
        assert len(remaining) == 1
        assert remaining[0]["label"] == "generalises"


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_valid_json(self, tmp_path):
        b = GraphBuilder(tmp_path / "g.json")
        b.apply_proposals(_bundle(
            topics=[TopicProposal(name="NLP")],
            subtopics=[SubtopicProposal(name="Transformers", parent_topic_names=["NLP"])],
        ))
        b.save()
        data = json.loads((tmp_path / "g.json").read_text())
        assert "nodes" in data
        assert "edges" in data or "links" in data

    def test_round_trip_preserves_everything(self, tmp_path):
        path = tmp_path / "g.json"
        b1 = GraphBuilder(path)
        b1.apply_proposals(_bundle(
            topics=[TopicProposal(name="NLP", source_chunk_ids=["c1"])],
            subtopics=[SubtopicProposal(name="Transformers", parent_topic_names=["NLP"])],
            contents=[ContentProposal(
                title="Self-Attention",
                parent_subtopic_names=["Transformers"],
                evidence_chunk_ids=["c1"],
            )],
        ))
        a = b1.add_node(TopicAttrs(name="Graph Theory"))
        b1.add_related_to(make_key("topic", "NLP"), a, "uses", confidence=0.8)
        b1.save()
        n, e = b1.node_count(), b1.edge_count()

        b2 = GraphBuilder(path)
        assert b2.node_count() == n
        assert b2.edge_count() == e
        edges = b2.graph.get_edge_data(make_key("topic", "NLP"), a)
        assert any(d["confidence"] == 0.8 for d in edges.values())
