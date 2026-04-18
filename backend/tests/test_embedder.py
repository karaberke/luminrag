"""
Tests for backend/retrieval/embedder.py

Uses the real model (all-MiniLM-L6-v2, ~22 MB, cached after first download).
No mocking — the embedder has no external I/O beyond the one-time model fetch.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.retrieval.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Load model once for the whole module — downloading takes ~5s first time."""
    return Embedder("all-MiniLM-L6-v2")


class TestEmbedShape:
    def test_single_text_shape(self, embedder):
        out = embedder.embed(["Hello world"])
        assert out.shape == (1, embedder.dimension)

    def test_batch_shape(self, embedder):
        out = embedder.embed(["Text one", "Text two", "Text three"])
        assert out.shape == (3, embedder.dimension)

    def test_embed_one_shape(self, embedder):
        out = embedder.embed_one("Hello world")
        assert out.shape == (embedder.dimension,)

    def test_embed_one_equals_first_row_of_embed(self, embedder):
        text = "Enzyme kinetics"
        one = embedder.embed_one(text)
        batch = embedder.embed([text])
        np.testing.assert_allclose(one, batch[0], atol=1e-5)


class TestEmbedProperties:
    def test_output_is_float32(self, embedder):
        out = embedder.embed(["test"])
        assert out.dtype == np.float32

    def test_vectors_are_l2_normalised(self, embedder):
        out = embedder.embed(["Enzyme", "RAG", "Temperature"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_identical_texts_produce_identical_vectors(self, embedder):
        text = "Michaelis-Menten kinetics"
        a = embedder.embed_one(text)
        b = embedder.embed_one(text)
        np.testing.assert_array_equal(a, b)

    def test_dimension_property_matches_output(self, embedder):
        out = embedder.embed_one("test")
        assert out.shape[0] == embedder.dimension


class TestSemanticQuality:
    def test_similar_texts_score_higher_than_dissimilar(self, embedder):
        query = embedder.embed_one("enzyme catalysis")
        similar = embedder.embed_one("enzymes accelerate biochemical reactions")
        dissimilar = embedder.embed_one("the French Revolution began in 1789")
        assert float(query @ similar) > float(query @ dissimilar)

    def test_cosine_between_identical_texts_is_one(self, embedder):
        text = "retrieval augmented generation"
        a = embedder.embed_one(text)
        score = float(a @ a)
        assert abs(score - 1.0) < 1e-5
