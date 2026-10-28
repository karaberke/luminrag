"""
Reranker: re-scores retrieved chunks against the original query using a
cross-encoder model, applies an optional relevancy cutoff, and caps the
source count.

Cross-encoders (unlike bi-encoders) take the full (query, passage) pair and
produce a single calibrated relevance logit, which we map to (0, 1) via
sigmoid.  Scores are much more meaningful than raw cosine similarity from the
embedding model.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~91 MB, CPU-friendly).
Lazy-loaded on first call and cached for the process lifetime.

Public API:
    chunks = rerank(query, chunks, config, min_relevancy, max_sources)
"""

from __future__ import annotations

import logging
import math
import os
import threading

from backend.schemas import Chunk
from backend.utils.models import pinned_model_path
from backend.utils.query_intent import multi_document_source_limit
from backend.utils.resources import (
    NATIVE_CPU_MEM_ARENA,
    ModelResourcesBusy,
    reserve_model_resource,
    reserve_native_inference,
)

logger = logging.getLogger(__name__)

_cross_encoder = None
_loaded_model_name: str | None = None
_inference_lock = threading.RLock()

_DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# FastEmbed's default of 64 pairs exceeds a 2 GiB container at 512 tokens.
RERANK_BATCH_SIZE = 8


def _get_cross_encoder(model_name: str):
    global _cross_encoder, _loaded_model_name
    if model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2":
        model_name = _DEFAULT_MODEL
    if _cross_encoder is None or _loaded_model_name != model_name:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        logger.info(f"Loading cross-encoder '{model_name}' …")
        threads = min(8, max(1, int(os.getenv("LUMIN_MODEL_THREADS", "2"))))
        # Callers hold _inference_lock; the shared native gate nests inside it.
        with reserve_native_inference():
            _cross_encoder = TextCrossEncoder(model_name=model_name, threads=threads,
                                             specific_model_path=str(pinned_model_path(model_name)),
                                             enable_cpu_mem_arena=NATIVE_CPU_MEM_ARENA)
        _loaded_model_name = model_name
        logger.info("Cross-encoder ready.")
    return _cross_encoder


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp = math.exp(x)
    return exp / (1.0 + exp)


def _cap(items: list, max_sources: int | None) -> list:
    return items if max_sources is None else items[:max(0, max_sources)]


def _source_representatives_first(scored: list[tuple[Chunk, float]], source_limit: int) -> list[tuple[Chunk, float]]:
    """Move the best eligible passage of up to source_limit sources to the front.

    Only reorders already scored and filtered candidates; the rest keep their
    stable score order, so a later cap keeps every requested document first.
    """
    first: list[int] = []
    seen: set[str] = set()
    for index, (chunk, _) in enumerate(scored):
        if len(seen) >= source_limit:
            break
        if chunk.source_id not in seen:
            seen.add(chunk.source_id)
            first.append(index)
    chosen = set(first)
    return [scored[index] for index in first] + [item for index, item in enumerate(scored) if index not in chosen]


def _degraded(
    chunks: list[Chunk],
    exc: Exception,
    min_relevancy: float,
    max_sources: int | None,
) -> list[Chunk]:
    """
    Without scores, a positive cutoff fails closed. When no cutoff was
    requested, preserve retrieval order and still enforce max_sources.
    """
    if min_relevancy > 0.0:
        logger.error(
            "Cross-encoder failed (%s); returning no passages because min_relevancy=%.2f cannot be verified without scores.",
            type(exc).__name__, min_relevancy,
        )
        return []
    capped = _cap(chunks, max_sources)
    logger.error(
        f"Cross-encoder failed ({exc}); degraded mode: returning {len(capped)}/"
        f"{len(chunks)} chunks in retrieval order, max_sources={max_sources}, "
        "no relevancy cutoff requested."
    )
    return capped


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
        Returns an empty list when all chunks fall below min_relevancy,
        or when a positive cutoff cannot be checked after model failure.
        The generator then reports that no supporting passage was found.
    """
    if not chunks:
        return chunks

    model_name = config.get("reranker", {}).get("model", _DEFAULT_MODEL)

    try:
        # Queries now generate independently of the knowledge-state lock.
        # Keep lazy loading and mutable model/tokenizer access serialized
        # without blocking graph reads, edits, or upload publication.
        with reserve_model_resource(_inference_lock):
            ce = _get_cross_encoder(model_name)
        texts = [chunk.text for chunk in chunks]
        raw_scores: list = []
        # Release both locks between bounded batches so no reservation spans
        # every candidate; model lock first, shared native gate second.
        for start in range(0, len(texts), RERANK_BATCH_SIZE):
            batch = texts[start:start + RERANK_BATCH_SIZE]
            with reserve_model_resource(_inference_lock), reserve_native_inference():
                raw_scores.extend(ce.rerank(query, batch, batch_size=RERANK_BATCH_SIZE))
        if len(raw_scores) != len(chunks) or not all(math.isfinite(float(score)) for score in raw_scores):
            raise ValueError("Reranker returned invalid scores.")
    except ModelResourcesBusy:
        # Contention is retryable; never mistake it for "no relevant passage".
        raise
    except Exception as exc:
        return _degraded(chunks, exc, min_relevancy, max_sources)

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

    source_limit = multi_document_source_limit(query)
    if source_limit:
        scored = _source_representatives_first(scored, source_limit)

    scored = _cap(scored, max_sources)

    logger.debug(
        f"Reranker: {before} → {len(scored)} chunks "
        f"(min_relevancy={min_relevancy}, max_sources={max_sources})"
    )
    return [c for c, _ in scored]
