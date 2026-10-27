"""Functional graph and document retrieval regressions without providers."""

from __future__ import annotations

import numpy as np

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval import graph_retriever
from backend.schemas import Chunk


class _FlatEmbedder:
    def embed(self, texts):
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

    def embed_one(self, _text):
        return np.array([1.0, 0.0], dtype=np.float32)


def _chunk(identifier, text):
    return Chunk(id=identifier, source_id="lecture", modality="pdf", text=text)


def test_graph_relations_survive_when_both_endpoints_are_anchors(tmp_path, monkeypatch):
    graph_retriever.clear_node_embeddings()
    builder = GraphBuilder(tmp_path / "graph.json")
    for key, level, name in (
        ("topic", "topic", "Photosynthesis"),
        ("subtopic", "subtopic", "Gas exchange"),
        ("content", "content", "Oxygen release"),
    ):
        builder.graph.add_node(key, name=name, node_type=level, source_ids=["passage"])
    builder.graph.add_node("leaf", name="source page", node_type="chunk_ref")
    for source, target, relation in (
        ("topic", "subtopic", "HAS_SUBTOPIC"),
        ("subtopic", "content", "HAS_CONTENT"),
        ("subtopic", "content", "RELATED_TO"),
        ("leaf", "content", "EVIDENCE_OF"),
    ):
        builder.graph.add_edge(source, target, relation=relation, source_chunk_ids=["passage"])
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([_chunk("passage", "Photosynthesis releases oxygen.")])
        result = graph_retriever.retrieve_graph(
            "Photosynthesis Gas exchange Oxygen release", builder, store, _FlatEmbedder(),
            {"graph_retriever": {"top_k_anchors": 3, "max_hops": 2}},
        )
    assert set(result.subgraph_node_keys) == {"topic", "subtopic", "content"}
    assert {triple.relation for triple in result.subgraph} == {
        "HAS_SUBTOPIC", "HAS_CONTENT", "RELATED_TO",
    }
    assert {chunk.id for chunk in result.chunks} == {"passage"}


def test_legacy_evidence_provenance_is_retrieved_without_exposing_leaf_relations(tmp_path, monkeypatch):
    graph_retriever.clear_node_embeddings()
    builder = GraphBuilder(tmp_path / "graph.json")
    builder.graph.add_node("content", name="Oxygen release", node_type="content", source_ids=[])
    builder.graph.add_node("legacy-passage", name="page 1", node_type="chunk_ref")
    builder.graph.add_edge("legacy-passage", "content", relation="EVIDENCE_OF")
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([_chunk("legacy-passage", "Photosynthesis releases oxygen.")])
        result = graph_retriever.retrieve_graph(
            "Oxygen release", builder, store, _FlatEmbedder(),
            {"graph_retriever": {"top_k_anchors": 1}},
        )
    assert [chunk.id for chunk in result.chunks] == ["legacy-passage"]
    assert result.subgraph_node_keys == ["content"]
    assert result.subgraph == []


def test_relationship_specific_evidence_is_included_in_retrieved_chunks(tmp_path, monkeypatch):
    graph_retriever.clear_node_embeddings()
    builder = GraphBuilder(tmp_path / "graph.json")
    builder.graph.add_node("first", name="Oxygen", node_type="content", source_ids=[])
    builder.graph.add_node("second", name="Photosynthesis", node_type="content", source_ids=[])
    builder.graph.add_edge("first", "second", relation="RELATED_TO", source_chunk_ids=["relation-passage"])
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([_chunk("relation-passage", "Photosynthesis releases oxygen.")])
        result = graph_retriever.retrieve_graph(
            "Oxygen Photosynthesis", builder, store, _FlatEmbedder(),
            {"graph_retriever": {"top_k_anchors": 2}},
        )
    assert [chunk.id for chunk in result.chunks] == ["relation-passage"]
    assert [triple.relation for triple in result.subgraph] == ["RELATED_TO"]


def test_bulk_chunk_lookup_preserves_request_order_without_query_per_chunk(tmp_path):
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([_chunk(f"c{index}", f"Source passage {index}.") for index in range(1250)])
        requested = [*(f"c{index}" for index in range(1249, -1, -1)), "c7", "missing", "c7"]
        selects = []
        store._conn.set_trace_callback(lambda sql: selects.append(sql) if sql.lstrip().upper().startswith("SELECT") else None)
        result = store.get_chunks_by_ids(identifier for identifier in requested)
        store._conn.set_trace_callback(None)
    assert [chunk.id for chunk in result] == [f"c{index}" for index in range(1249, -1, -1)]
    assert len(selects) <= 4, "Large retrievals must batch SQLite reads rather than issue a query per chunk"


def test_empty_bulk_chunk_lookup_does_not_query_database(tmp_path):
    with DocumentStore(tmp_path / "chunks.db") as store:
        selects = []
        store._conn.set_trace_callback(selects.append)
        assert store.get_chunks_by_ids([]) == []
    assert selects == []
