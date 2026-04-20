"""
Reranker: re-scores retrieved chunks against the original query using a
cross-encoder model, applies an optional relevancy cutoff, and caps the
source count.

Cross-encoders (unlike bi-encoders) take the full (query, passage) pair and
produce a single calibrated relevance logit, which we map to (0, 1) via
sigmoid.  Scores are much more meaningful than raw cosine similarity from the
embedding model.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22 MB, CPU-friendly).
Lazy-loaded on first call and cached for the process lifetime.

Public API:
    chunks = rerank(query, chunks, config, min_relevancy, max_sources)
"""

from __future__ import annotations

import logging
import math

from backend.schemas import Chunk

logger = logging.getLogger(__name__)

_cross_encoder = None
_loaded_model_name: str | None = None

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder(model_name: str):
    global _cross_encoder, _loaded_model_name
    if _cross_encoder is None or _loaded_model_name != model_name:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder '{model_name}' …")
        _cross_encoder = CrossEncoder(model_name)
        _loaded_model_name = model_name
        logger.info("Cross-encoder ready.")
    return _cross_encoder


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    chunks: list[Chunk],
    config: dict,
    min_relevancy: float = 0.0,
    max_sources: int | None = None,
) -> list[Chunk]:
    """
    Re-score chunks against *query* with a cross-encoder, filter, sort, cap.

    Args:
        query:          The original user question (not sub-questions).
        chunks:         Merged chunks from multi-hop retrieval.
        config:         Parsed config/llm.yaml (reads reranker.model).
        min_relevancy:  Drop chunks whose sigmoid(logit) < this threshold.
                        0.0 keeps everything.  Meaningful range ~0.3–0.7.
        max_sources:    Maximum chunks to return.  None = no cap.

    Returns:
        Filtered, sorted list of Chunk objects.  Each chunk has
        ``metadata["relevancy_score"]`` set to sigmoid(logit) in (0, 1).
        Returns an empty list when all chunks fall below min_relevancy
        (the generator falls back to the NO_RETRIEVAL prompt path).
    """
    if not chunks:
        return chunks

    model_name = config.get("reranker", {}).get("model", _DEFAULT_MODEL)

    try:
        ce = _get_cross_encoder(model_name)
        pairs = [(query, c.text) for c in chunks]
        raw_scores: list[float] = ce.predict(pairs).tolist()
    except Exception as exc:
        logger.error(f"Cross-encoder failed ({exc}); returning chunks unranked.")
        return chunks

    scored: list[tuple[Chunk, float]] = []
    for chunk, raw in zip(chunks, raw_scores):
        sig = round(_sigmoid(float(raw)), 4)
        annotated = chunk.model_copy(
            update={"metadata": {**chunk.metadata, "relevancy_score": sig}}
        )
        scored.append((annotated, sig))

    before = len(scored)
    if min_relevancy > 0.0:
        scored = [(c, s) for c, s in scored if s >= min_relevancy]
        dropped = before - len(scored)
        if dropped:
            logger.info(
                f"Reranker: dropped {dropped}/{before} chunks below "
                f"min_relevancy={min_relevancy:.2f}"
            )

    scored.sort(key=lambda x: x[1], reverse=True)

    if max_sources is not None:
        scored = scored[:max_sources]

    logger.debug(
        f"Reranker: {before} → {len(scored)} chunks "
        f"(min_relevancy={min_relevancy}, max_sources={max_sources})"
    )
    return [c for c, _ in scored]
