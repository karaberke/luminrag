"""New concepts must retain linking budget without re-labeling existing edges."""

from unittest.mock import Mock

import networkx as nx
import numpy as np

from backend.graph.semantic_linker import _candidate_pairs, link_graph_changes, link_nodes, snapshot_linkable_texts
from backend.retrieval.graph_retriever import warm_up
from backend.schemas import Chunk


def nodes():
    return [{"key": name, "name": name, "text": name, "node_type": "content",
             "source_ids": [name], "source_excerpts": [{"id": name, "text": "Shared material about " + name}]}
            for name in ("a", "b", "c")]


def chunks():
    return [Chunk(id=name, text="Shared material about " + name, modality="pdf", source_id="local")
            for name in ("a", "b", "c", "replacement")]


def vectors():
    angles = np.deg2rad([0, 5, 50])
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def test_new_nodes_are_not_starved_by_stronger_old_pairs():
    embedder = Mock()
    pairs = _candidate_pairs(nodes(), embedder, 0.2, 1.0, 1, focus_keys={"c"}, vectors=vectors())
    assert len(pairs) == 1
    assert {pairs[0][1]["key"], pairs[0][2]["key"]} == {"b", "c"}
    embedder.embed.assert_not_called()


def test_existing_pairs_are_excluded_before_applying_the_pair_budget():
    pairs = _candidate_pairs(
        nodes(), Mock(), 0.2, 1.0, 1, focus_keys={"c"},
        excluded_pairs={frozenset(("b", "c"))}, vectors=vectors(),
    )
    assert {pairs[0][1]["key"], pairs[0][2]["key"]} == {"a", "c"}


def test_linking_and_retrieval_share_unchanged_node_embeddings(monkeypatch):
    graph = nx.MultiDiGraph()
    for name in ("a", "b"):
        graph.add_node(name, name=name, node_type="content", summary="", source_ids=[name])
    lookup = dict(zip(("a", "b", "c"), vectors()))
    embedder = Mock()
    embedder.embed.side_effect = lambda texts: np.stack([lookup[text.split(".")[0]] for text in texts])
    warm_up(graph, embedder)
    previous = snapshot_linkable_texts(graph)
    embedder.embed.reset_mock()
    graph.add_node("c", name="c", node_type="content", summary="", source_ids=["c"])
    graph.add_edge("b", "c", relation="HAS_CONTENT")
    llm = Mock(return_value=[{"index": 0, "related": True, "confidence": 0.8, "source_chunk_ids": ["a", "c"]}])
    monkeypatch.setattr("backend.graph.semantic_linker.safe_call_json", llm)
    config = {"semantic_linker": {"sim_lo": 0.2, "sim_hi": 1.0, "max_candidate_pairs": 1}}
    proposals = link_graph_changes(graph, previous, embedder, config, chunks=chunks())
    assert len(proposals) == 1
    assert {proposals[0].source_key, proposals[0].target_key} == {"a", "c"}
    assert embedder.embed.call_args_list[0].args[0] == ["c."]
    warm_up(graph, embedder)
    assert embedder.embed.call_count == 1
    llm.reset_mock()
    assert link_graph_changes(graph, snapshot_linkable_texts(graph), embedder, config, chunks=chunks()) == []
    assert embedder.embed.call_count == 1
    llm.assert_not_called()


def test_changed_source_ids_reverify_without_reembedding_identical_concept_text(monkeypatch):
    graph = nx.MultiDiGraph()
    for name in ("a", "b"):
        graph.add_node(name, name=name, node_type="content", source_ids=[name])
    embedder = Mock()
    embedder.embed.return_value = vectors()[:2]
    warm_up(graph, embedder)
    previous = snapshot_linkable_texts(graph)
    graph.nodes["a"]["source_ids"] = ["replacement"]
    llm = Mock(return_value=[{"index": 0, "related": True, "confidence": .8,
                             "source_chunk_ids": ["replacement", "b"]}])
    monkeypatch.setattr("backend.graph.semantic_linker.safe_call_json", llm)
    proposals = link_graph_changes(graph, previous, embedder,
                                   {"semantic_linker": {"sim_lo": .2, "sim_hi": 1.}}, chunks=chunks())
    assert len(proposals) == 1 and proposals[0].source_chunk_ids == ["replacement", "b"]
    assert embedder.embed.call_count == 1
    llm.assert_called_once()


def test_invalid_model_confidence_cannot_poison_graph_json(monkeypatch):
    monkeypatch.setattr("backend.graph.semantic_linker.safe_call_json", lambda *args, **kwargs: [
        {"index": 0, "related": True, "confidence": value, "source_chunk_ids": ["a", "b"]}
        for value in (float("nan"), float("inf"), 2, -1)
    ])
    assert link_nodes(nodes(), Mock(), {"semantic_linker": {"sim_lo": 0.2, "sim_hi": 1.0}}, vectors=vectors()) == []
