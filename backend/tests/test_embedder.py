"""Real cached ONNX model contracts and semantic discrimination."""
import numpy as np
import pytest

from backend.retrieval.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder("BAAI/bge-small-en-v1.5")


def test_scalar_and_batch_embeddings_agree_and_are_normalized_float32(embedder):
    texts = ["Enzyme kinetics", "RAG", "Temperature"]
    batch = embedder.embed(texts)
    one = embedder.embed_one(texts[0])
    single = embedder.embed([texts[0]])
    assert batch.shape == (3, embedder.dimension)
    assert one.shape == (embedder.dimension,) and single.shape == (1, embedder.dimension)
    assert batch.dtype == one.dtype == single.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), np.ones(3), atol=1e-5)
    np.testing.assert_allclose(one, batch[0], atol=1e-5)
    np.testing.assert_allclose(single[0], batch[0], atol=1e-5)
    np.testing.assert_array_equal(one, embedder.embed_one(texts[0]))


def test_similar_text_is_closer_than_disconnected_history(embedder):
    query, related, unrelated = embedder.embed([
        "enzyme catalysis", "enzymes accelerate biochemical reactions", "the French Revolution began in 1789"])
    assert float(query @ related) > float(query @ unrelated)
