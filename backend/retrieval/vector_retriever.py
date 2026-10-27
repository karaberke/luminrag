"""
Stage 4.1 - Dense Retrieval: vector_retriever.py

Builds and queries a NumPy vector index over ingested Chunks.
Uses IndexFlatIP (inner product = cosine on L2-normalised vectors).

VectorIndex lifecycle:
    index = VectorIndex(embedder)
    index.build(chunks)           # embed all chunks and populate NumPy
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

import numpy as np

from backend.db.document_store import DocumentStore
from backend.credentials import is_hosted
from backend.retrieval.embedder import Embedder
from backend.schemas import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class IndexModelMismatchError(ValueError):
    """Stored vectors need rebuilding with the currently configured model."""


class VectorIndex:
    """Exact cosine retrieval over a compact float32 matrix, with no DB server."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._index: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._model_name: str | None = None

    @property
    def embeddings(self) -> np.ndarray:
        if self._index is None:
            return np.empty((0, self._embedder.dimension), dtype=np.float32)
        return self._index.copy()

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    @property
    def model_name(self) -> str | None:
        return self._model_name or getattr(self._embedder, "model_name", None)

    def build(self, chunks: list[Chunk], *, reuse_from: VectorIndex | None = None,
              previous_chunks: list[Chunk] | None = None) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")
        chunk_ids = [chunk.id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs must be unique within an index.")
        dim = self._embedder.dimension
        model_name = getattr(self._embedder, "model_name", None)
        previous_positions: dict[str, int] = {}
        previous_texts: dict[str, str] = {}
        if (reuse_from is not None and reuse_from._embedder is self._embedder
                and reuse_from._index is not None and reuse_from._index.shape[1] == dim
                and reuse_from._model_name == model_name and previous_chunks is not None):
            previous_positions = {cid: i for i, cid in enumerate(reuse_from._chunk_ids)}
            previous_texts = {chunk.id: chunk.text for chunk in previous_chunks}
        embeddings = np.empty((len(chunks), dim), dtype=np.float32)
        changed = []
        for i, chunk in enumerate(chunks):
            if chunk.id in previous_positions and previous_texts.get(chunk.id) == chunk.text:
                embeddings[i] = reuse_from._index[previous_positions[chunk.id]]
            else:
                changed.append(i)
        if changed:
            vectors = self._embedder.embed([chunks[i].text for i in changed])
            if vectors.shape != (len(changed), dim) or not np.isfinite(vectors).all():
                raise ValueError("Embedding dimensions do not match the configured model.")
            embeddings[changed] = vectors
        self._index = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._chunk_ids = chunk_ids
        self._model_name = model_name
        logger.info("NumPy index built: %d vectors, dim=%d", len(chunks), dim)

    def add(self, chunks: list[Chunk]) -> None:
        """Append new chunks; duplicate IDs require a complete replacement build."""
        if not chunks:
            return
        ids = [chunk.id for chunk in chunks]
        if len(set(ids)) != len(ids) or set(ids).intersection(self._chunk_ids):
            raise ValueError("Chunk IDs must be unique within an index.")
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        if vectors.shape != (len(chunks), self._embedder.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding matrix.")
        self._index = np.ascontiguousarray(vectors if self._index is None else np.concatenate([self._index, vectors]), dtype=np.float32)
        self._chunk_ids = self._chunk_ids + ids
        self._model_name = getattr(self._embedder, "model_name", None)

    def save(self, index_path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("Index has not been built yet.")
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        # A file handle prevents NumPy from appending another suffix to old paths.
        with index_path.open("wb") as stream:
            np.save(stream, self._index, allow_pickle=False)
        index_path.with_suffix(".json").write_text(json.dumps({
            "version": 1, "model": self._model_name, "chunk_ids": self._chunk_ids,
        }), encoding="utf-8")

    def load(self, index_path: str | Path) -> None:
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Vector index not found: {index_path}")
        try:
            with index_path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                if version not in {(1, 0), (2, 0)}:
                    raise ValueError("Unsupported NumPy version.")
                read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fortran_order, dtype = read_header(stream, max_header_size=4096)
                row_limit = 20000 if is_hosted() else 100000
                if (len(shape) != 2 or shape[0] > row_limit or dtype != np.dtype("float32")
                        or shape[1] != self._embedder.dimension):
                    raise IndexModelMismatchError("Vector index dimensions or format are incompatible with the configured embedding model.")
                # Reject declared giant arrays or truncated payloads before
                # np.load can allocate memory from an untrusted bundle header.
                if index_path.stat().st_size != stream.tell() + shape[0] * shape[1] * dtype.itemsize:
                    raise ValueError("NumPy vector payload length does not match its header.")
                stream.seek(0)
                candidate = np.load(stream, allow_pickle=False)
        except (ValueError, OSError, EOFError):
            raise IndexModelMismatchError("Vector index is incompatible; rebuild from stored documents.") from None
        if not isinstance(candidate, np.ndarray) or candidate.ndim != 2 or candidate.dtype != np.float32 or not np.isfinite(candidate).all():
            raise ValueError("Vector index is incompatible with the configured embedding model.")
        if candidate.shape[1] != self._embedder.dimension:
            raise IndexModelMismatchError("Vector index dimensions are incompatible with the configured embedding model.")
        metadata = json.loads(index_path.with_suffix(".json").read_text(encoding="utf-8"))
        legacy = isinstance(metadata, list)
        if not legacy and (not isinstance(metadata, dict) or metadata.get("version") != 1):
            raise ValueError("Unsupported vector index sidecar format.")
        chunk_ids = metadata if legacy else metadata.get("chunk_ids")
        if (not isinstance(chunk_ids, list) or not all(isinstance(cid, str) and cid for cid in chunk_ids)
                or len(set(chunk_ids)) != len(chunk_ids) or len(chunk_ids) != len(candidate)):
            raise ValueError("Vector index and chunk-ID sidecar do not match.")
        model_name = getattr(self._embedder, "model_name", None)
        if chunk_ids and (legacy or not model_name or metadata.get("model") != model_name):
            raise IndexModelMismatchError("Vector index sidecar has an unknown or different embedding model; rebuild from stored documents.")
        self._index = np.ascontiguousarray(candidate)
        self._chunk_ids = chunk_ids
        self._model_name = model_name

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Index has not been built or loaded.")
        k = min(top_k, len(self._chunk_ids))
        if k <= 0:
            return []
        query_embed = getattr(self._embedder, "embed_query", self._embedder.embed_one)
        query_vector = np.asarray(query_embed(query), dtype=np.float32).reshape(-1)
        if query_vector.shape != (self._index.shape[1],) or not np.isfinite(query_vector).all():
            raise ValueError("Invalid query embedding.")
        scores = self._index @ query_vector
        positions = np.argpartition(-scores, k - 1)[:k]
        # Stable tie ordering by original position makes bundle round trips exact.
        positions = positions[np.lexsort((positions, -scores[positions]))]
        return [(self._chunk_ids[i], float(scores[i])) for i in positions]

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

    chunks = store.get_chunks_by_ids(chunk_id for chunk_id, _score in hits)
    if len(chunks) != len(hits):
        logger.warning("Skipping %d indexed chunks missing from the document store", len(hits) - len(chunks))

    chunks = [chunk.model_copy(update={"metadata": {**chunk.metadata, "retrieval_source": "dense"}})
              for chunk in chunks]

    logger.debug(f"Dense retrieval: {len(chunks)} chunks for query '{query[:60]}'")
    return RetrievalResult(chunks=chunks, subgraph=[], routing_mode="dense")
