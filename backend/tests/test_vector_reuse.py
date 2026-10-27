"""Vector reuse must retain exact retrieval results without stale embeddings."""

import hashlib
import json

import numpy as np
import pytest

from backend.retrieval.vector_retriever import IndexModelMismatchError, VectorIndex
from backend.schemas import Chunk


class CountingEmbedder:
    dimension = 8
    model_name = "counting-model-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = np.array([
            list(hashlib.sha256(text.encode()).digest()[:self.dimension]) for text in texts
        ], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def embed_one(self, text):
        return self.embed([text])[0]


def chunk(identifier, text):
    return Chunk(id=identifier, source_id="lesson", modality="pdf", text=text)


def test_reuse_matches_fresh_index_for_reordered_changed_removed_and_new_chunks(tmp_path):
    embedder = CountingEmbedder()
    previous_chunks = [chunk("a", "old alpha"), chunk("b", "bravo"), chunk("c", "charlie")]
    previous = VectorIndex(embedder)
    previous.build(previous_chunks)
    before = previous.embeddings.copy()
    path = tmp_path / "index.faiss"
    previous.save(path)
    previous.load(path)  # Reuse also works after a process restart.

    current_chunks = [chunk("c", "charlie"), chunk("a", "new alpha"), chunk("d", "delta")]
    embedder.calls.clear()
    candidate = VectorIndex(embedder)
    candidate.build(current_chunks, reuse_from=previous, previous_chunks=previous_chunks)
    assert embedder.calls == [["new alpha", "delta"]]
    np.testing.assert_array_equal(previous.embeddings, before)

    fresh = VectorIndex(embedder)
    fresh.build(current_chunks)
    np.testing.assert_array_equal(candidate.embeddings, fresh.embeddings)
    assert candidate.search("new alpha", 10) == fresh.search("new alpha", 10)


def test_reuse_handles_more_than_one_reconstruction_batch_without_embedding():
    embedder = CountingEmbedder()
    previous_chunks = [chunk(str(i), f"course paragraph {i}") for i in range(4105)]
    previous = VectorIndex(embedder)
    previous.build(previous_chunks)
    embedder.calls.clear()
    current = VectorIndex(embedder)
    current.build(list(reversed(previous_chunks)), reuse_from=previous, previous_chunks=previous_chunks)
    assert embedder.calls == []
    np.testing.assert_array_equal(
        current.embeddings, previous.embeddings[::-1]
    )


def test_different_embedder_or_missing_previous_texts_are_never_reused():
    old_embedder, new_embedder = CountingEmbedder(), CountingEmbedder()
    documents = [chunk("a", "alpha")]
    previous = VectorIndex(old_embedder)
    previous.build(documents)
    current = VectorIndex(new_embedder)
    current.build(documents, reuse_from=previous, previous_chunks=documents)
    assert new_embedder.calls == [["alpha"]]
    old_embedder.calls.clear()
    current = VectorIndex(old_embedder)
    current.build(documents, reuse_from=previous)
    assert old_embedder.calls == [["alpha"]]


@pytest.mark.parametrize("bad_ids", [["a"], ["a", "a"], ["a", 4], {"a": 0}, ["a", ""]])
def test_invalid_sidecar_does_not_replace_a_working_index(tmp_path, bad_ids):
    embedder = CountingEmbedder()
    index = VectorIndex(embedder)
    index.build([chunk("a", "alpha"), chunk("b", "bravo")])
    before = index.search("alpha")
    path = tmp_path / "index.faiss"
    index.save(path)
    path.with_suffix(".json").write_text(json.dumps(bad_ids), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        index.load(path)
    assert index.search("alpha") == before


@pytest.mark.parametrize("dimension,metric", [(4, "ip"), (8, "l2")])
def test_incompatible_index_is_rejected_on_load(tmp_path, dimension, metric):
    path = tmp_path / "index.faiss"
    with path.open("wb") as stream:
        if metric == "ip":
            np.save(stream, np.ones((1, dimension), dtype=np.float32), allow_pickle=False)
        else:
            stream.write(b"legacy-faiss-index")
    path.with_suffix(".json").write_text('["a"]', encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        VectorIndex(CountingEmbedder()).load(path)


def test_empty_and_zero_limit_searches_do_not_call_model_or_faiss_search(tmp_path):
    embedder = CountingEmbedder()
    index = VectorIndex(embedder)
    index.build([chunk("a", "alpha")])
    embedder.calls.clear()
    assert index.search("alpha", 0) == []
    assert index.search("alpha", -1) == []
    path = tmp_path / "empty.faiss"
    with path.open("wb") as stream:
        np.save(stream, np.empty((0, embedder.dimension), dtype=np.float32), allow_pickle=False)
    path.with_suffix(".json").write_text("[]", encoding="utf-8")
    index.load(path)
    assert index.search("alpha", 5) == []
    assert embedder.calls == []


def test_duplicate_ids_fail_without_changing_previous_index():
    embedder = CountingEmbedder()
    index = VectorIndex(embedder)
    index.build([chunk("a", "alpha")])
    with pytest.raises(ValueError, match="unique"):
        index.build([chunk("a", "alpha"), chunk("a", "bravo")])
    assert len(index) == 1
    assert index.search("alpha", 1)[0][0] == "a"


def test_model_identity_roundtrips_and_rejects_same_dimension_switch(tmp_path):
    documents = [chunk("a", "alpha")]
    previous = VectorIndex(CountingEmbedder())
    previous.build(documents)
    path = tmp_path / "index.faiss"
    previous.save(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata == {"version": 1, "model": "counting-model-v1", "chunk_ids": ["a"]}
    other = CountingEmbedder()
    other.model_name = "another-model-same-dimensions"
    candidate = VectorIndex(other)
    with pytest.raises(IndexModelMismatchError, match="different embedding model"):
        candidate.load(path)
    assert len(candidate) == 0
    candidate.build(documents, reuse_from=previous, previous_chunks=documents)
    assert other.calls == [["alpha"]]


def test_legacy_index_requires_rebuild_without_mutating_live_index(tmp_path):
    index = VectorIndex(CountingEmbedder())
    index.build([chunk("a", "alpha")])
    expected = index.search("alpha")
    path = tmp_path / "index.faiss"
    index.save(path)
    path.with_suffix(".json").write_text('["a"]', encoding="utf-8")
    with pytest.raises(IndexModelMismatchError, match="sidecar"):
        index.load(path)
    assert index.search("alpha") == expected


def test_changed_model_identity_prevents_reuse_even_with_same_object():
    embedder = CountingEmbedder()
    documents = [chunk("a", "alpha")]
    previous = VectorIndex(embedder)
    previous.build(documents)
    embedder.calls.clear()
    embedder.model_name = "changed-model"
    candidate = VectorIndex(embedder)
    candidate.build(documents, reuse_from=previous, previous_chunks=documents)
    assert embedder.calls == [["alpha"]]


def test_hostile_numpy_header_is_rejected_before_numpy_allocates(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    path = tmp_path / "embeddings.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream, {
            "descr": "<f4", "fortran_order": False, "shape": (10**12, 8),
        })
    def allocate(*_args, **_kwargs):
        raise AssertionError("Untrusted header must be validated before allocation")
    monkeypatch.setattr(np, "load", allocate)
    with pytest.raises(ValueError, match="incompatible"):
        VectorIndex(CountingEmbedder()).load(path)


def test_truncated_vector_payload_is_rejected_without_replacing_index(tmp_path):
    index = VectorIndex(CountingEmbedder())
    index.build([chunk("a", "alpha")])
    expected = index.embeddings
    path = tmp_path / "embeddings.npy"
    index.save(path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incompatible"):
        index.load(path)
    np.testing.assert_array_equal(index.embeddings, expected)
