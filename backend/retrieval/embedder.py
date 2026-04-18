"""
Shared embedding layer used by both retrieval modules.

Wraps sentence-transformers with L2-normalised output so that inner product
equals cosine similarity — required for FAISS IndexFlatIP and for the
relevance-guided BFS in graph_retriever.

Public API:
    embedder = Embedder(model_name)
    embedder.embed(["text1", "text2"])   # -> np.ndarray shape (N, D)
    embedder.embed_one("text")           # -> np.ndarray shape (D,)
    embedder.dimension                   # -> int
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model.
    All embeddings are L2-normalised on output so cosine similarity
    reduces to inner product (faster, and required by FAISS IndexFlatIP).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: Non-empty list of strings.

        Returns:
            Float32 array of shape (len(texts), D), L2-normalised row-wise.
        """
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (D,)."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the underlying model."""
        return self._model.get_embedding_dimension()
