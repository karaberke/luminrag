"""Verify selected retrieval modes through the real API orchestration boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.db.document_store import DocumentStore
from backend.generation import generator
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval import reranker
from backend.retrieval.embedder import Embedder
from backend.retrieval.graph_retriever import retrieve_graph
from backend.schemas import Chunk, ReflectionVerdict
from backend.self_rag import multi_hop_reasoner
from backend.tests.test_ingest_safety import api_state  # noqa: F401


class _FlatEmbedder:
    dimension = 2

    def embed(self, texts):
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

    def embed_one(self, _text):
        return self.embed([_text])[0]

    embed_query = embed_one


@pytest.mark.parametrize("mode", ["dense", "graph", "hybrid"])
@pytest.mark.parametrize("query_mode", ["fast", "thorough"])
def test_mode_dispatch_cross_encoding_generation_and_trace(api_state, monkeypatch, mode, query_mode):
    api = api_state
    events = []
    prompts = []
    passages = {
        "dense": "Vector-only passage about proteins.",
        "graph": "Graph-only passage: enzymes lower activation energy.",
        "shared": "Shared passage: catalysts speed up reactions.",
    }
    api.store.clear_all()
    api.store.save_chunks([Chunk(id=key, source_id=key, modality="pdf", text=text) for key, text in passages.items()])
    graph = api.graph.graph
    graph.clear()
    graph.add_node("bio-topic", node_type="topic", name="Biology", source_ids=["shared"])
    graph.add_node("enzyme-node", node_type="content", name="Enzymes", source_ids=["graph"])
    graph.add_edge("bio-topic", "enzyme-node", relation="HAS_CONTENT")

    def search(question, _top_k):
        events.append("vector-index")
        return [("shared", 0.9), ("dense", 0.8)]

    monkeypatch.setattr(api.main.state, "vector_index", SimpleNamespace(search=search))
    monkeypatch.setattr(api.main.state, "embedder", _FlatEmbedder())
    dense = multi_hop_reasoner.retrieve_dense
    graph_retrieval = multi_hop_reasoner.retrieve_graph

    def dense_spy(*args, **kwargs):
        events.append("dense")
        assert mode != "graph", "Explicit Graph mode must not silently use dense retrieval"
        return dense(*args, **kwargs)

    def graph_spy(*args, **kwargs):
        events.append("graph")
        assert mode != "dense", "Explicit Dense mode must not search graph nodes"
        return graph_retrieval(*args, **kwargs)

    monkeypatch.setattr(multi_hop_reasoner, "retrieve_dense", dense_spy)
    monkeypatch.setattr(multi_hop_reasoner, "retrieve_graph", graph_spy)
    monkeypatch.setattr(multi_hop_reasoner, "_decompose", lambda question, _cfg: [question])

    class Encoder:
        def rerank(self, question, documents):
            assert question == "Explain Biology and Enzymes"
            assert "generate" not in events
            events.append("cross-encode")
            # The actual cross-encoder wrapper applies sigmoid, sorting and
            # source limits to these logits before answer generation.
            logits = {passages["dense"]: 3.0, passages["shared"]: 2.0, passages["graph"]: 1.0}
            return [logits[text] for text in documents]

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda _model: Encoder())

    def provider(prompt, _system, _cfg):
        assert events[-1] == "cross-encode"
        events.append("generate")
        prompts.append(prompt)
        return generator._AnswerOutput(answer="Enzymes are biological catalysts [1].")

    monkeypatch.setattr(generator, "_call_llm", provider)
    monkeypatch.setattr(generator, "reflect_answer", lambda *_: ReflectionVerdict(
        needs_retrieval=True, is_relevant=True, is_supported=True, is_useful=True, reasoning="fixture",
    ))
    response = api.main.query(api.main.QueryRequest(query="Explain Biology and Enzymes", routing_mode=mode, query_mode=query_mode))

    assert events.count("dense") == (mode in {"dense", "hybrid"})
    assert events.count("graph") == (mode in {"graph", "hybrid"})
    assert events.count("cross-encode") == events.count("generate") == 1
    assert response.routing_mode == mode
    assert [item.id for item in response.evidence] == {
        "dense": ["dense", "shared"], "graph": ["shared", "graph"], "hybrid": ["dense", "shared", "graph"],
    }[mode]
    expected_sources = {
        "dense": {"dense": "dense", "shared": "dense"},
        "graph": {"shared": "graph", "graph": "graph"},
        "hybrid": {"dense": "dense", "shared": "both", "graph": "graph"},
    }[mode]
    assert {item.id: item.retrieval_source for item in response.evidence} == expected_sources
    assert all(item.relevancy_score is not None for item in response.evidence)
    assert set(response.hops) == (set() if mode == "dense" else {"bio-topic", "enzyme-node"})
    for key, text in passages.items():
        assert (text in prompts[0]) == (key in expected_sources)
    assert ("Biology → Enzymes (hierarchy)" in prompts[0]) == (mode != "dense")


def test_real_bge_graph_search_excludes_disconnected_history_topic(tmp_path):
    """Small relevant branches must not be padded with unrelated anchor nodes."""
    embedder = Embedder()
    builder = GraphBuilder(tmp_path / "graph.json")
    for key, kind, name, summary, source in [
        ("bio-topic", "topic", "Biology", "Enzymes, biological catalysts, activation energy and cellular reactions.", "bio"),
        ("bio-content", "content", "Enzyme catalysis", "Enzymes speed up chemical reactions by lowering activation energy without being consumed.", "bio"),
        ("history-topic", "topic", "History", "The French Revolution, monarchy, political reform and events in 1789.", "history"),
        ("history-content", "content", "French Revolution", "The French Revolution began in 1789 with political and social changes in France.", "history"),
    ]:
        builder.graph.add_node(key, node_type=kind, name=name, summary=summary, source_ids=[source])
    builder.graph.add_edge("bio-topic", "bio-content", relation="HAS_CONTENT")
    builder.graph.add_edge("history-topic", "history-content", relation="HAS_CONTENT")
    question = "How do enzymes act as biological catalysts?"
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([
            Chunk(id="bio", source_id="biology.pdf", modality="pdf", text="Enzymes lower activation energy."),
            Chunk(id="history", source_id="history.pdf", modality="pdf", text="The French Revolution began in 1789."),
        ])
        result = retrieve_graph(question, builder, store, embedder, {})
    assert set(result.subgraph_node_keys) == {"bio-topic", "bio-content"}
    assert [chunk.id for chunk in result.chunks] == ["bio"]
    assert all(chunk.metadata["retrieval_source"] == "graph" for chunk in result.chunks)
    assert len(result.subgraph) == 1
    assert result.subgraph[0].source_chunk_ids == []
    assert result.subgraph[0].head == "Biology"
    assert result.subgraph[0].tail == "Enzyme catalysis"
