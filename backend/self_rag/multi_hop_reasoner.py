"""
Stage 6 - Multi-Hop Reasoner: multi_hop_reasoner.py

Decomposes complex questions into sub-questions, retrieves and reflects on each,
then merges all evidence into a single RetrievalResult for the generator.

Public API:
    reason(question, vector_index, graph_builder, store, embedder, config)
        -> RetrievalResult
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from backend.llm import structured
from backend.schemas import Chunk, GraphTriple, RetrievalResult
from backend.retrieval.query_router import route_query
from backend.retrieval.vector_retriever import retrieve_dense
from backend.retrieval.graph_retriever import retrieve_graph
from backend.self_rag.self_rag_reflector import reflect_retrieval

logger = logging.getLogger(__name__)


class _Decomposition(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=8)


# ---------------------------------------------------------------------------
# Decomposition prompt
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a question decomposer for an educational QA system.

Complex question:
{question}

Break this into at most {max_sub} simpler, self-contained sub-questions that together
cover all aspects needed to answer the original question. If the question is already
simple, return it as-is in a list of one.

Respond with ONLY valid JSON (no markdown fences):
{{"sub_questions": ["<sub-question 1>", "<sub-question 2>", ...]}}\
"""

# ---------------------------------------------------------------------------
# LLM callers (same pattern as self_rag_reflector)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, cfg: dict) -> str:
    return structured(prompt, {**cfg, "provider": "ollama"}, _Decomposition).model_dump_json()


def _call_anthropic(prompt: str, cfg: dict) -> str:
    return structured(prompt, {**cfg, "provider": "anthropic"}, _Decomposition).model_dump_json()


def _call_llm(prompt: str, cfg: dict) -> str:
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return _call_ollama(prompt, cfg)
    if provider == "anthropic":
        return _call_anthropic(prompt, cfg)
    return structured(prompt, cfg, _Decomposition).model_dump_json()


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _decompose(question: str, cfg: dict) -> list[str]:
    """
    Use LLM to decompose question into sub-questions.
    Falls back to [question] on any failure.
    """
    max_sub = cfg.get("max_sub_questions", 4)
    prompt = _DECOMPOSE_PROMPT.format(question=question, max_sub=max_sub)

    try:
        raw = _call_llm(prompt, cfg)
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
        data = json.loads(text)
        sub_qs = data.get("sub_questions", [])
        if not isinstance(sub_qs, list) or not sub_qs:
            return [question]
        cleaned = [q for q in sub_qs[:max_sub] if isinstance(q, str) and q.strip()]
        return cleaned or [question]
    except Exception as exc:
        logger.warning(f"Decomposition failed: {exc} — treating as single question")
        return [question]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _merge_results(results: list[RetrievalResult]) -> RetrievalResult:
    """
    Merge multiple RetrievalResults into one.
    Chunks are deduplicated by id; triples by (head, relation, tail).
    Routing mode is 'hybrid' when both dense and graph evidence are present,
    'graph' if only graph results, 'dense' otherwise.
    """
    chunk_positions: dict[str, int] = {}
    seen_triple_keys: set[tuple] = set()
    seen_node_keys: set[str] = set()
    merged_chunks: list[Chunk] = []
    merged_triples: list[GraphTriple] = []
    merged_node_keys: list[str] = []

    for r in results:
        for chunk in r.chunks:
            if chunk.id not in chunk_positions:
                chunk_positions[chunk.id] = len(merged_chunks)
                merged_chunks.append(chunk)
            else:
                position = chunk_positions[chunk.id]
                existing = merged_chunks[position]
                sources = {existing.metadata.get("retrieval_source"), chunk.metadata.get("retrieval_source")}
                if "both" in sources or {"dense", "graph"} <= sources:
                    merged_chunks[position] = existing.model_copy(update={
                        "metadata": {**existing.metadata, "retrieval_source": "both"},
                    })
        for triple in r.subgraph:
            key = (triple.head, triple.relation, triple.tail)
            if key not in seen_triple_keys:
                seen_triple_keys.add(key)
                merged_triples.append(triple)
        for nkey in r.subgraph_node_keys:
            if nkey not in seen_node_keys:
                seen_node_keys.add(nkey)
                merged_node_keys.append(nkey)

    modes = {r.routing_mode for r in results}
    if "hybrid" in modes or ("dense" in modes and "graph" in modes):
        routing_mode = "hybrid"
    elif "graph" in modes:
        routing_mode = "graph"
    else:
        routing_mode = "dense"

    return RetrievalResult(
        chunks=merged_chunks,
        subgraph=merged_triples,
        routing_mode=routing_mode,
        subgraph_node_keys=merged_node_keys,
    )


def _retrieve_hybrid(
    sub_q: str,
    vector_index,
    graph_builder,
    store,
    embedder,
    config: dict,
) -> RetrievalResult:
    """
    Run dense and graph retrieval, tag each chunk with its retrieval_source
    in metadata ('dense', 'graph', or 'both' when found by both), then merge.
    """
    dense_result = retrieve_dense(sub_q, vector_index, store, config)
    graph_result = retrieve_graph(sub_q, graph_builder, store, embedder, config)

    graph_ids = {c.id for c in graph_result.chunks}
    tagged_chunks: list[Chunk] = []
    seen: set[str] = set()

    for chunk in dense_result.chunks:
        source = "both" if chunk.id in graph_ids else "dense"
        tagged_chunks.append(
            chunk.model_copy(update={"metadata": {**chunk.metadata, "retrieval_source": source}})
        )
        seen.add(chunk.id)

    for chunk in graph_result.chunks:
        if chunk.id not in seen:
            tagged_chunks.append(
                chunk.model_copy(update={"metadata": {**chunk.metadata, "retrieval_source": "graph"}})
            )

    # Merge subgraph data (triples + node keys) from both results
    seen_triple_keys: set[tuple] = set()
    seen_node_keys: set[str] = set()
    merged_triples: list[GraphTriple] = []
    merged_node_keys: list[str] = []

    for triple in graph_result.subgraph:
        key = (triple.head, triple.relation, triple.tail)
        if key not in seen_triple_keys:
            seen_triple_keys.add(key)
            merged_triples.append(triple)
    for nkey in graph_result.subgraph_node_keys:
        if nkey not in seen_node_keys:
            seen_node_keys.add(nkey)
            merged_node_keys.append(nkey)

    return RetrievalResult(
        chunks=tagged_chunks,
        subgraph=merged_triples,
        routing_mode="hybrid",
        subgraph_node_keys=merged_node_keys,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_for_mode(
    sub_question: str,
    mode: str,
    vector_index,
    graph_builder,
    store,
    embedder,
    config: dict,
) -> RetrievalResult:
    """Dispatch retrieval independently of the decomposition and reflection LLM calls."""
    if mode == "hybrid":
        return _retrieve_hybrid(sub_question, vector_index, graph_builder, store, embedder, config)
    if mode == "graph":
        return retrieve_graph(sub_question, graph_builder, store, embedder, config)
    return retrieve_dense(sub_question, vector_index, store, config)


def reason(
    question: str,
    vector_index,
    graph_builder,
    store,
    embedder,
    config: dict,
    force_routing_mode: str | None = None,
    *,
    retrieve_callback: Callable[[str, str], RetrievalResult] | None = None,
) -> RetrievalResult:
    """
    Decompose a complex question, retrieve+reflect per sub-question, merge evidence.

    Steps:
      1. Decompose question into sub-questions via LLM.
      2. For each sub-question: route → retrieve → reflect_retrieval.
         Skip sub-questions with irrelevant or empty results.
      3. Merge all passing RetrievalResults into one (deduped).

    Args:
        question:            Raw user question.
        vector_index:        VectorIndex (from vector_retriever).
        graph_builder:       GraphBuilder (from graph_builder).
        store:               DocumentStore.
        embedder:            Embedder (shared, used by graph retriever).
        config:              Parsed config/llm.yaml.
        force_routing_mode:  When set to "dense" or "graph", skip the query router
                             and use this mode for every sub-question.
        retrieve_callback:   Optional retrieval boundary, called with the sub-question
                             and routing mode. A server can acquire its state lock,
                             validate a captured revision, and call retrieve_for_mode
                             here while the LLM calls run without holding that lock.
                             Exceptions propagate, so stale work can be rejected.

    Returns:
        Merged RetrievalResult. Returns an empty result if nothing passes reflection.
    """
    cfg = config.get("multi_hop_reasoner", {})
    sub_questions = [question] if config.get("query_mode") == "fast" else _decompose(question, cfg)
    logger.debug(f"Decomposed '{question}' → {len(sub_questions)} sub-questions")

    results: list[RetrievalResult] = []
    routed_modes: set[str] = set()

    for sub_q in sub_questions:
        if force_routing_mode is not None:
            mode = force_routing_mode
            logger.debug(f"Using forced routing mode '{mode}' for sub-question: {sub_q!r}")
        else:
            mode = route_query(sub_q, config)
        routed_modes.add(mode)

        if mode == "none":
            logger.debug(f"Sub-question routed to 'none', skipping: {sub_q!r}")
            continue

        if retrieve_callback is not None:
            result = retrieve_callback(sub_q, mode)
        else:
            result = retrieve_for_mode(sub_q, mode, vector_index, graph_builder, store, embedder, config)

        # When the user forces a retrieval mode, skip the relevance gate — they
        # explicitly chose this path and expect results from it.
        if force_routing_mode is None and config.get("query_mode") != "fast":
            verdict = reflect_retrieval(sub_q, result, config)
            if not verdict.is_relevant:
                logger.debug(f"Reflection deemed irrelevant, skipping: {sub_q!r}")
                continue

        results.append(result)

    if not results:
        logger.warning("No relevant results found across all sub-questions")
        # Preserve an intentional conversational route. A failed/irrelevant
        # retrieval attempt must still reach the source-grounded empty-result
        # guard rather than silently falling back to general knowledge.
        fallback_mode = force_routing_mode or ("none" if routed_modes == {"none"} else "dense")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode=fallback_mode, subgraph_node_keys=[])

    return _merge_results(results)
