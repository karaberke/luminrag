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

from backend.schemas import Chunk, GraphTriple, RetrievalResult
from backend.retrieval.query_router import route_query
from backend.retrieval.vector_retriever import retrieve_dense
from backend.retrieval.graph_retriever import retrieve_graph
from backend.self_rag.self_rag_reflector import reflect_retrieval

logger = logging.getLogger(__name__)

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
    import httpx

    payload = {
        "model": cfg.get("ollama_model", "llama3.2:3b"),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_predict": cfg.get("max_tokens", 256)},
    }
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def _call_anthropic(prompt: str, cfg: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=cfg.get("anthropic_model", "claude-haiku-4-5-20251001"),
        max_tokens=cfg.get("max_tokens", 256),
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_llm(prompt: str, cfg: dict) -> str:
    """Dispatch to the configured LLM provider."""
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return _call_ollama(prompt, cfg)
    if provider == "anthropic":
        return _call_anthropic(prompt, cfg)
    raise ValueError(f"Unknown multi_hop_reasoner provider: '{provider}'")


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
    Routing mode is 'graph' if any result used graph retrieval, else 'dense'.
    """
    seen_chunk_ids: set[str] = set()
    seen_triple_keys: set[tuple] = set()
    merged_chunks: list[Chunk] = []
    merged_triples: list[GraphTriple] = []

    for r in results:
        for chunk in r.chunks:
            if chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.id)
                merged_chunks.append(chunk)
        for triple in r.subgraph:
            key = (triple.head, triple.relation, triple.tail)
            if key not in seen_triple_keys:
                seen_triple_keys.add(key)
                merged_triples.append(triple)

    routing_mode = "graph" if any(r.routing_mode == "graph" for r in results) else "dense"

    return RetrievalResult(
        chunks=merged_chunks,
        subgraph=merged_triples,
        routing_mode=routing_mode,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason(
    question: str,
    vector_index,
    graph_builder,
    store,
    embedder,
    config: dict,
    force_routing_mode: str | None = None,
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

    Returns:
        Merged RetrievalResult. Returns an empty result if nothing passes reflection.
    """
    cfg = config.get("multi_hop_reasoner", {})
    sub_questions = _decompose(question, cfg)
    logger.debug(f"Decomposed '{question}' → {len(sub_questions)} sub-questions")

    results: list[RetrievalResult] = []

    for sub_q in sub_questions:
        if force_routing_mode is not None:
            mode = force_routing_mode
            logger.debug(f"Using forced routing mode '{mode}' for sub-question: {sub_q!r}")
        else:
            mode = route_query(sub_q, config)

        if mode == "none":
            logger.debug(f"Sub-question routed to 'none', skipping: {sub_q!r}")
            continue

        if mode == "graph":
            result = retrieve_graph(sub_q, graph_builder, store, embedder, config)
        else:
            result = retrieve_dense(sub_q, vector_index, store, config)

        # When the user forces a retrieval mode, skip the relevance gate — they
        # explicitly chose this path and expect results from it.
        if force_routing_mode is None:
            verdict = reflect_retrieval(sub_q, result, config)
            if not verdict.is_relevant:
                logger.debug(f"Reflection deemed irrelevant, skipping: {sub_q!r}")
                continue

        results.append(result)

    if not results:
        logger.warning("No relevant results found across all sub-questions")
        fallback_mode = force_routing_mode or "dense"
        return RetrievalResult(chunks=[], subgraph=[], routing_mode=fallback_mode)

    return _merge_results(results)
