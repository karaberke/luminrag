"""
Shared embedding layer used by both retrieval modules.

Wraps FastEmbed ONNX with L2-normalised output so that inner product
equals cosine similarity for NumPy exact search and for the
relevance-guided BFS in graph_retriever.

Public API:
    embedder = Embedder(model_name)
    embedder.embed(["text1", "text2"])   # -> np.ndarray shape (N, D)
    embedder.embed_one("text")           # -> np.ndarray shape (D,)
    embedder.dimension                   # -> int
"""

from __future__ import annotations

import threading
import os

import numpy as np

from backend.utils.resources import reserve_model_resource
from backend.utils.models import pinned_model_path


class Embedder:
    """
    Thin wrapper around a FastEmbed ONNX model.
    All embeddings are L2-normalised on output so cosine similarity
    reduces to inner product.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding
        if model_name == "all-MiniLM-L6-v2":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._model_name = model_name
        threads = min(8, max(1, int(os.getenv("LUMIN_MODEL_THREADS", "2"))))
        self._model = TextEmbedding(model_name=model_name, threads=threads,
                                    specific_model_path=str(pinned_model_path(model_name)))
        self._dimension = TextEmbedding.get_embedding_size(model_name)
        # ONNX/tokenizer inference is serialized independently of state locks.
        self._inference_lock = threading.RLock()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: Non-empty list of strings.

        Returns:
            Float32 array of shape (len(texts), D), L2-normalised row-wise.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with reserve_model_resource(self._inference_lock):
            vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return self._normalize(vecs)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        if vecs.ndim != 2 or vecs.shape[1] != self.dimension or not np.isfinite(vecs).all():
            raise ValueError("Embedding model returned invalid vectors.")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return np.ascontiguousarray(vecs / np.maximum(norms, np.finfo(np.float32).tiny), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a query using BGE's retrieval prefix, not the passage API."""
        with reserve_model_resource(self._inference_lock):
            vecs = np.asarray(list(self._model.query_embed(text)), dtype=np.float32)
        return self._normalize(vecs)[0]

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (D,)."""
        return self.embed([text])[0]

    @property
    def model_name(self) -> str:
        """Configured model identity persisted alongside vector indexes."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the underlying model."""
        return self._dimension
