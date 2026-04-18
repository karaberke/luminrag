"""
Stage 4.1 - Dense Retrieval: vector_retriever.py

Builds and queries a FAISS vector index over ingested Chunks.
Uses IndexFlatIP (inner product = cosine on L2-normalised vectors).

VectorIndex lifecycle:
    index = VectorIndex(embedder)
    index.build(chunks)           # embed all chunks and populate FAISS
    index.save(path)              # persist index + chunk-ID sidecar
    index = VectorIndex(embedder)
    index.load(path)              # restore from disk
    hits = index.search(query)   # [(chunk_id, score), ...]

Public retrieval function:
    result = retrieve_dense(query, index, store, config)
    # -> RetrievalResult(chunks=[...], subgraph=[], routing_mode="dense")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from backend.db.document_store import DocumentStore
from backend.retrieval.embedder import Embedder
from backend.schemas import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    FAISS-backed dense vector index over Chunk objects.

    Chunks are embedded once at build time. Queries are embedded at search time.
    A JSON sidecar file alongside the FAISS index maps integer positions → chunk IDs.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._chunk_ids: list[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed all chunks and populate the FAISS index.
        Replaces any previously built index.
        """
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")

        logger.info(f"Embedding {len(chunks)} chunks…")
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunk_ids = [c.id for c in chunks]
        logger.info(f"FAISS index built: {len(chunks)} vectors, dim={dim}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str | Path) -> None:
        """
        Write the FAISS index to index_path and a chunk-ID map to
        index_path.with_suffix('.json').
        """
        if self._index is None:
            raise RuntimeError("Index has not been built yet.")
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        index_path.with_suffix(".json").write_text(
            json.dumps(self._chunk_ids), encoding="utf-8"
        )
        logger.info(f"FAISS index saved to {index_path}")

    def load(self, index_path: str | Path) -> None:
        """Restore a previously saved FAISS index and its chunk-ID map."""
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self._index = faiss.read_index(str(index_path))
        self._chunk_ids = json.loads(
            index_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        logger.info(
            f"FAISS index loaded from {index_path} ({len(self._chunk_ids)} vectors)"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Embed query and return top-K (chunk_id, cosine_score) pairs,
        sorted by score descending.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built or loaded.")

        k = min(top_k, len(self._chunk_ids))
        q_emb = self._embedder.embed_one(query).reshape(1, -1)
        scores, indices = self._index.search(q_emb, k)

        return [
            (self._chunk_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunk_ids)


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------

def retrieve_dense(
    query: str,
    index: VectorIndex,
    store: DocumentStore,
    config: dict,
) -> RetrievalResult:
    """
    Run a dense vector search and return a RetrievalResult.

    Args:
        query:  Raw user question.
        index:  A built/loaded VectorIndex.
        store:  DocumentStore for fetching full Chunk objects.
        config: Parsed config/llm.yaml.

    Returns:
        RetrievalResult with routing_mode="dense" and empty subgraph.
    """
    top_k = config.get("vector_retriever", {}).get("top_k", 5)
    hits = index.search(query, top_k)

    chunks: list[Chunk] = []
    for chunk_id, score in hits:
        chunk = store.get_chunk(chunk_id)
        if chunk:
            chunks.append(chunk)
        else:
            logger.warning(f"Chunk '{chunk_id}' in index but not in store — skipping.")

    logger.debug(f"Dense retrieval: {len(chunks)} chunks for query '{query[:60]}'")
    return RetrievalResult(chunks=chunks, subgraph=[], routing_mode="dense")
