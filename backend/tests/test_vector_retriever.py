"""
Tests for backend/retrieval/vector_retriever.py

Uses real embeddings + FAISS (no mocking). The embedder fixture is
module-scoped so the model loads once across all tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.db.document_store import DocumentStore
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_retriever import VectorIndex, retrieve_dense
from backend.schemas import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder("all-MiniLM-L6-v2")


def _chunk(id: str, text: str, source_id: str = "src", modality: str = "pdf") -> Chunk:
    return Chunk(id=id, text=text, source_id=source_id, modality=modality)


CORPUS = [
    _chunk("c0", "Enzymes are biological catalysts that speed up chemical reactions."),
    _chunk("c1", "The Michaelis-Menten equation models enzyme kinetics."),
    _chunk("c2", "RAG combines retrieval with large language model generation."),
    _chunk("c3", "GraphRAG uses a knowledge graph for multi-hop reasoning."),
    _chunk("c4", "Hallucination in LLMs occurs when models generate false information."),
]


@pytest.fixture(scope="module")
def built_index(embedder) -> VectorIndex:
    idx = VectorIndex(embedder)
    idx.build(CORPUS)
    return idx


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    with DocumentStore(tmp_path / "test.db") as s:
        s.save_chunks(CORPUS)
        yield s


# ---------------------------------------------------------------------------
# VectorIndex.build
# ---------------------------------------------------------------------------

class TestBuild:
    def test_len_after_build(self, embedder):
        idx = VectorIndex(embedder)
        idx.build(CORPUS)
        assert len(idx) == len(CORPUS)

    def test_build_empty_raises(self, embedder):
        idx = VectorIndex(embedder)
        with pytest.raises(ValueError):
            idx.build([])


# ---------------------------------------------------------------------------
# VectorIndex.search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_correct_count(self, built_index):
        results = built_index.search("enzyme kinetics", top_k=3)
        assert len(results) == 3

    def test_returns_tuples_of_str_and_float(self, built_index):
        results = built_index.search("enzyme", top_k=2)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_top_result_is_semantically_closest(self, built_index):
        results = built_index.search("enzyme reaction catalysis", top_k=5)
        top_id = results[0][0]
        assert top_id in ("c0", "c1")  # enzyme-related chunks

    def test_rag_query_returns_rag_chunk(self, built_index):
        results = built_index.search("retrieval augmented generation LLM", top_k=3)
        top_ids = [r[0] for r in results]
        assert "c2" in top_ids or "c3" in top_ids

    def test_scores_sorted_descending(self, built_index):
        results = built_index.search("enzyme", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_capped_at_corpus_size(self, built_index):
        results = built_index.search("enzyme", top_k=100)
        assert len(results) == len(CORPUS)

    def test_search_before_build_raises(self, embedder):
        idx = VectorIndex(embedder)
        with pytest.raises(RuntimeError):
            idx.search("query")


# ---------------------------------------------------------------------------
# VectorIndex save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_files(self, embedder, tmp_path):
        idx = VectorIndex(embedder)
        idx.build(CORPUS)
        path = tmp_path / "test.index"
        idx.save(path)
        assert path.exists()
        assert path.with_suffix(".json").exists()

    def test_load_restores_len(self, embedder, tmp_path):
        idx1 = VectorIndex(embedder)
        idx1.build(CORPUS)
        path = tmp_path / "test.index"
        idx1.save(path)

        idx2 = VectorIndex(embedder)
        idx2.load(path)
        assert len(idx2) == len(idx1)

    def test_round_trip_returns_same_top_result(self, embedder, tmp_path):
        idx1 = VectorIndex(embedder)
        idx1.build(CORPUS)
        path = tmp_path / "test.index"
        idx1.save(path)

        idx2 = VectorIndex(embedder)
        idx2.load(path)

        q = "enzyme catalysis reaction"
        assert idx1.search(q, 1)[0][0] == idx2.search(q, 1)[0][0]

    def test_load_missing_file_raises(self, embedder, tmp_path):
        idx = VectorIndex(embedder)
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "nonexistent.index")


# ---------------------------------------------------------------------------
# retrieve_dense
# ---------------------------------------------------------------------------

class TestRetrieveDense:
    def _config(self, top_k: int = 3) -> dict:
        return {"vector_retriever": {"top_k": top_k}}

    def test_returns_retrieval_result(self, built_index, store):
        result = retrieve_dense("enzyme", built_index, store, self._config())
        assert isinstance(result, RetrievalResult)

    def test_routing_mode_is_dense(self, built_index, store):
        result = retrieve_dense("enzyme", built_index, store, self._config())
        assert result.routing_mode == "dense"

    def test_subgraph_is_empty(self, built_index, store):
        result = retrieve_dense("enzyme", built_index, store, self._config())
        assert result.subgraph == []

    def test_returns_chunk_objects(self, built_index, store):
        result = retrieve_dense("enzyme", built_index, store, self._config())
        assert all(isinstance(c, Chunk) for c in result.chunks)

    def test_chunk_count_respects_top_k(self, built_index, store):
        result = retrieve_dense("enzyme", built_index, store, self._config(top_k=2))
        assert len(result.chunks) == 2

    def test_missing_chunk_in_store_skipped(self, built_index, tmp_path):
        # Store is empty — chunks exist in index but not in store
        with DocumentStore(tmp_path / "empty.db") as empty_store:
            result = retrieve_dense("enzyme", built_index, empty_store, self._config())
        assert result.chunks == []
