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
from contextlib import contextmanager

import numpy as np

from backend.utils.resources import (
    NATIVE_CPU_MEM_ARENA,
    ModelResourcesBusy,
    model_resource_timeout,
    reserve_native_inference,
)
from backend.utils.models import pinned_model_path

# Small native batches keep peak RSS of 512-token passages within 2 GiB.
EMBEDDING_BATCH_SIZE = 16


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
        with reserve_native_inference():
            self._model = TextEmbedding(model_name=model_name, threads=threads,
                                        specific_model_path=str(pinned_model_path(model_name)),
                                        enable_cpu_mem_arena=NATIVE_CPU_MEM_ARENA)
        self._dimension = TextEmbedding.get_embedding_size(model_name)
        # ONNX/tokenizer inference is serialized independently of state locks.
        self._inference_condition = threading.Condition()
        self._inference_active = False
        self._interactive_waiters = 0

    @contextmanager
    def _reserve_inference(self):
        """Give queued interactive calls the next batch boundary, within a deadline."""
        timeout = model_resource_timeout()
        with self._inference_condition:
            if timeout is None:
                self._inference_condition.wait_for(lambda: not self._inference_active and not self._interactive_waiters)
            else:
                self._interactive_waiters += 1
                self._inference_condition.notify_all()
                try:
                    available = self._inference_condition.wait_for(lambda: not self._inference_active, timeout=timeout)
                finally:
                    self._interactive_waiters -= 1
                    self._inference_condition.notify_all()
                if not available:
                    raise ModelResourcesBusy()
            self._inference_active = True
        try:
            yield
        finally:
            with self._inference_condition:
                self._inference_active = False
                self._inference_condition.notify_all()

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
        # A large import must not reserve the shared tokenizer/model for its
        # entire corpus. Release it between bounded native inference batches.
        vectors = np.empty((len(texts), self.dimension), dtype=np.float32)
        # Similar-length passages waste less tokenizer padding/native compute.
        # Only inference order changes; callers retain exact input row order.
        lengths = np.fromiter((len(text) for text in texts), dtype=np.intp, count=len(texts))
        positions = np.argsort(lengths, kind="stable")
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            selected = positions[start:start + EMBEDDING_BATCH_SIZE]
            batch = [texts[index] for index in selected]
            # Per-model reservation first, then the process-wide native gate.
            with self._reserve_inference(), reserve_native_inference():
                result = np.asarray(list(self._model.embed(batch, batch_size=EMBEDDING_BATCH_SIZE)), dtype=np.float32)
            if result.shape != (len(batch), self.dimension):
                raise ValueError("Embedding model returned an unexpected number of vectors.")
            vectors[selected] = self._normalize(result)
        return vectors

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        if vecs.ndim != 2 or vecs.shape[1] != self.dimension or not np.isfinite(vecs).all():
            raise ValueError("Embedding model returned invalid vectors.")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return np.ascontiguousarray(vecs / np.maximum(norms, np.finfo(np.float32).tiny), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a query using BGE's retrieval prefix, not the passage API."""
        with self._reserve_inference(), reserve_native_inference():
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
