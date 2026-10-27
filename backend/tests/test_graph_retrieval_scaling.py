"""Bound graph evidence reads and encode only graph text that actually changed."""

from weakref import WeakKeyDictionary

import networkx as nx
import numpy as np
import pytest

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval import graph_retriever
from backend.schemas import Chunk


class _CountingEmbedder:
    dimension = 2

    def __init__(self, reverse=False):
        self.calls = []
        self.reverse = reverse

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([
            [0.0, 1.0] if (("changed" in text) ^ self.reverse) else [1.0, 0.0]
            for text in texts
        ], dtype=np.float32)

    def embed_one(self, text):
        return self.embed([text])[0]


@pytest.fixture(autouse=True)
def isolated_graph_cache(monkeypatch):
    monkeypatch.setattr(graph_retriever, "_graph_caches", WeakKeyDictionary())


def _large_graph(count=3000):
    graph = nx.MultiDiGraph()
    for index in range(count):
        graph.add_node(f"n{index}", name=f"Concept {index}", summary="original", node_type="content")
    return graph


def test_warming_large_graph_encodes_only_added_or_edited_nodes():
    graph = _large_graph()
    embedder = _CountingEmbedder()
    graph_retriever.warm_up(graph, embedder)
    assert sum(map(len, embedder.calls)) == 3000
    embedder.calls.clear()
    graph_retriever.warm_up(graph, embedder)
    assert embedder.calls == []

    graph.nodes["n0"]["summary"] = "changed summary"
    graph.add_node("new", name="New concept", summary="changed content", node_type="content")
    graph.remove_node("n1")
    graph_retriever.warm_up(graph, embedder)
    assert sum(map(len, embedder.calls)) == 2
    assert "n1" not in graph_retriever._graph_caches[graph][1]


def test_one_missing_node_does_not_reencode_cached_neighbours():
    graph = _large_graph()
    embedder = _CountingEmbedder()
    graph_retriever.warm_up(graph, embedder)
    embedder.calls.clear()
    graph.add_node("new", name="New concept", summary="changed", node_type="content")
    vectors = graph_retriever.get_node_embeddings(list(graph), graph, embedder)
    assert vectors.shape == (3001, 2)
    assert sum(map(len, embedder.calls)) == 1
    np.testing.assert_array_equal(vectors[-1], [0.0, 1.0])


def test_node_text_edits_invalidate_cache_even_before_explicit_warmup():
    graph = _large_graph(1)
    embedder = _CountingEmbedder()
    original = graph_retriever.get_node_embeddings(["n0"], graph, embedder)
    graph.nodes["n0"]["summary"] = "changed"
    current = graph_retriever.get_node_embeddings(["n0"], graph, embedder)
    assert not np.array_equal(original, current)


def test_different_graph_with_same_node_keys_cannot_reuse_stale_text():
    graph = _large_graph(1)
    embedder = _CountingEmbedder()
    graph_retriever.warm_up(graph, embedder)
    replacement = _large_graph(1)
    replacement.nodes["n0"]["summary"] = "changed"
    vectors = graph_retriever.get_node_embeddings(["n0"], replacement, embedder)
    np.testing.assert_array_equal(vectors[0], [0.0, 1.0])


def test_different_embedding_model_cannot_reuse_prior_models_vectors():
    graph = _large_graph(1)
    graph_retriever.warm_up(graph, _CountingEmbedder())
    replacement = _CountingEmbedder(reverse=True)
    vectors = graph_retriever.get_node_embeddings(["n0"], graph, replacement)
    assert sum(map(len, replacement.calls)) == 1
    np.testing.assert_array_equal(vectors[0], [0.0, 1.0])


@pytest.mark.parametrize("field,value", [("body", "changed note body"), ("key_terms", ["changed term"]), ("raw_excerpt", "changed excerpt")])
def test_editable_node_fields_invalidate_semantic_cache(field, value):
    graph = _large_graph(1)
    embedder = _CountingEmbedder()
    original = graph_retriever.get_node_embeddings(["n0"], graph, embedder)
    graph.nodes["n0"][field] = value
    edited = graph_retriever.get_node_embeddings(["n0"], graph, embedder)
    assert not np.array_equal(original, edited)
    assert len(embedder.calls) == 2


@pytest.mark.parametrize("limit", [None, 3])
def test_broad_topic_evidence_is_bounded_and_keeps_specific_content(tmp_path, limit):
    builder = GraphBuilder(tmp_path / "graph.json")
    identifiers = [f"c{index}" for index in range(300)]
    builder.graph.add_node("topic", name="Photosynthesis", node_type="topic", source_ids=identifiers)
    builder.graph.add_node("content", name="Oxygen release", node_type="content", source_ids=["c299"])
    builder.graph.add_edge("topic", "content", relation="HAS_CONTENT")
    config = {"top_k_anchors": 2, "max_hops": 1}
    if limit is not None:
        config["max_chunks"] = limit
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([
            Chunk(id=identifier, source_id="lecture", modality="pdf", text="Photosynthesis releases oxygen.")
            for identifier in identifiers
        ])
        selects = []
        store._conn.set_trace_callback(lambda sql: selects.append(sql) if sql.lstrip().upper().startswith("SELECT") else None)
        result = graph_retriever.retrieve_graph(
            "Photosynthesis Oxygen release", builder, store, _CountingEmbedder(), {"graph_retriever": config},
        )
        store._conn.set_trace_callback(None)
    assert len(result.chunks) == (80 if limit is None else limit)
    assert result.chunks[0].id == "c299"
    assert len(selects) <= 1, "A broad topic must not create hundreds of SQLite point lookups"
