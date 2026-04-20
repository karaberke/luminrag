"""
Stage 7 - Generation: generator.py

Selects the appropriate prompt template, formats context from the retrieval
result, calls the configured LLM, runs a post-generation Self-RAG reflection,
and returns a structured GenerationResult.

Public API:
    generate(question, retrieval_result, config) -> GenerationResult
"""

from __future__ import annotations

import logging
import re

from backend.schemas import GenerationResult, ReflectionVerdict, RetrievalResult
from backend.self_rag.self_rag_reflector import reflect_answer
from backend.generation.prompts import (
    DENSE_RAG_PROMPT,
    GRAPH_RAG_PROMPT,
    HYBRID_RAG_PROMPT,
    NO_RETRIEVAL_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------

def _format_chunks(result: RetrievalResult, max_chars: int) -> str:
    """
    Format retrieved chunks as a numbered list, truncated to max_chars.
    Returns a placeholder when no chunks are available.
    """
    if not result.chunks:
        return "(no context retrieved)"

    lines: list[str] = []
    total = 0

    for i, chunk in enumerate(result.chunks, start=1):
        line = f"[{i}] ({chunk.modality}) {chunk.text}"
        if total + len(line) > max_chars:
            lines.append(f"[{i}] ... (truncated — context limit reached)")
            break
        lines.append(line)
        total += len(line)

    return "\n\n".join(lines)


def _format_relationships(result: RetrievalResult) -> str:
    """Format subgraph triples as readable lines."""
    if not result.subgraph:
        return "(no relationships extracted)"
    return "\n".join(
        f"- {t.head} —[{t.relation}]→ {t.tail}" for t in result.subgraph
    )


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str, cfg: dict) -> str:
    import httpx

    payload = {
        "model": cfg.get("ollama_model", "llama3.2:3b"),
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": cfg.get("max_tokens", 1024)},
    }
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"]


def _call_anthropic(prompt: str, system: str, cfg: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=cfg.get("anthropic_model", "claude-haiku-4-5-20251001"),
        max_tokens=cfg.get("max_tokens", 1024),
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_llm(prompt: str, system: str, cfg: dict) -> str:
    """Dispatch to the configured LLM provider."""
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return _call_ollama(prompt, system, cfg)
    if provider == "anthropic":
        return _call_anthropic(prompt, system, cfg)
    raise ValueError(f"Unknown generator provider: '{provider}'")


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _extract_cited_chunk_ids(answer: str, result: RetrievalResult) -> list[str]:
    """
    Parse [N] citations from the answer text and map back to chunk ids.
    Returns an empty list if the result has no chunks or no citations found.
    """
    cited_ids: list[str] = []
    seen: set[str] = set()

    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        idx = n - 1  # citations are 1-based
        if 0 <= idx < len(result.chunks):
            chunk_id = result.chunks[idx].id
            if chunk_id not in seen:
                seen.add(chunk_id)
                cited_ids.append(chunk_id)

    return cited_ids


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    question: str,
    result: RetrievalResult,
    max_chars: int,
) -> str:
    """Select and fill the appropriate prompt template."""
    mode = result.routing_mode

    if mode == "none" or not result.chunks:
        return NO_RETRIEVAL_PROMPT.format(question=question)

    context = _format_chunks(result, max_chars)

    if mode == "graph":
        relationships = _format_relationships(result)
        return GRAPH_RAG_PROMPT.format(
            question=question,
            context=context,
            relationships=relationships,
        )

    if mode == "hybrid":
        relationships = _format_relationships(result)
        return HYBRID_RAG_PROMPT.format(
            question=question,
            context=context,
            relationships=relationships,
        )

    # dense
    return DENSE_RAG_PROMPT.format(question=question, context=context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    question: str,
    retrieval_result: RetrievalResult,
    config: dict,
) -> GenerationResult:
    """
    Generate an answer for the question using the retrieval result.

    Steps:
      1. Build a prompt (dense / graph / no-retrieval) from the retrieval result.
      2. Call the configured LLM.
      3. Run reflect_answer (Self-RAG post-generation gate).
      4. Extract [N] citations from the answer text.
      5. Return GenerationResult.

    Args:
        question:         Raw user question.
        retrieval_result: Output of retrieval (or multi-hop reasoner).
        config:           Parsed config/llm.yaml.

    Returns:
        GenerationResult with answer, routing_mode, reflection, evidence_chunk_ids.
    """
    cfg = config.get("generator", {})
    max_chars = cfg.get("max_context_chars", 4000)

    prompt = _build_prompt(question, retrieval_result, max_chars)

    try:
        answer = _call_llm(prompt, SYSTEM_PROMPT, cfg)
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        answer = (
            "I'm sorry, I was unable to generate a response at this time. "
            "Please try again later."
        )

    reflection = reflect_answer(question, retrieval_result, answer, config)
    evidence_chunk_ids = _extract_cited_chunk_ids(answer, retrieval_result)

    logger.debug(
        f"Generated answer ({len(answer)} chars) — "
        f"supported={reflection.is_supported} useful={reflection.is_useful} "
        f"citations={len(evidence_chunk_ids)}"
    )

    return GenerationResult(
        answer=answer,
        routing_mode=retrieval_result.routing_mode,
        reflection=reflection,
        evidence_chunk_ids=evidence_chunk_ids,
    )
