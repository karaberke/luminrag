"""Background ingestion must not enter a shared tokenizer concurrently."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from backend.retrieval.embedder import Embedder


def test_shared_embedding_model_serializes_inference_without_losing_results(monkeypatch):
    class NonReentrantModel:
        def __init__(self, model_name, **_kwargs):
            self.busy = threading.Lock()

        @staticmethod
        def get_embedding_size(model_name):
            return 2

        def embed(self, texts, **kwargs):
            assert self.busy.acquire(blocking=False), "Concurrent model/tokenizer access"
            try:
                time.sleep(0.002)
                return np.array([[len(text), 1] for text in texts], dtype=np.float32)
            finally:
                self.busy.release()

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=NonReentrantModel))
    monkeypatch.setattr("backend.retrieval.embedder.pinned_model_path", lambda _: "verified-test-model")
    embedder = Embedder("test")
    texts = ["a" * length for length in range(1, 33)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(embedder.embed_one, texts))
    expected = np.array([[length, 1] for length in range(1, 33)], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(results, expected)
