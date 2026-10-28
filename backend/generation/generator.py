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

from pydantic import BaseModel, Field

from backend.llm import structured
from backend.schemas import CitationSpan, GenerationResult, ReflectionVerdict, RetrievalResult
from backend.self_rag.self_rag_reflector import reflect_answer
from backend.utils.context import (
    context_chunks as _context_chunks,
    format_context as _format_chunks,
    generation_context_chars,
)
from backend.utils.text import find_quote_span
from backend.generation.prompts import (
    DENSE_RAG_PROMPT,
    GRAPH_RAG_PROMPT,
    HYBRID_RAG_PROMPT,
    NO_EVIDENCE_ANSWER,
    NO_RETRIEVAL_PROMPT,
    SYSTEM_PROMPT,
    UNSUPPORTED_ANSWER,
)

logger = logging.getLogger(__name__)


class _AnswerCitation(BaseModel):
    source_number: int = Field(ge=1, description="The passage number [N] used in answer.")
    quote: str = Field(max_length=4000, description="Exact supporting sentence copied verbatim from that passage.")


class _AnswerOutput(BaseModel):
    citations: list[_AnswerCitation] = Field(max_length=32)
    answer: str = Field(min_length=1, description="The complete Markdown answer, preserving numbered citations [N].")


_CITATION_INSTRUCTIONS = """\
Return the answer and its source quotes together in the required JSON object.
Select supporting quotes first, then write only the answer those quotes support.
The answer field contains the complete Markdown answer with [N] source references.
For each source referenced in the answer, include one citations entry with its
source_number N and a short exact supporting sentence copied from passage [N].
Copy spelling, punctuation and words exactly; never paraphrase or combine separate
sentences into a quote. Cite only numbered passages shown in retrieved_context.
If no passage supports a claim, say so. With no retrieved passages, citations is [].
"""

_FAST_ANSWER_INSTRUCTIONS = """\
Keep the answer concise: aim for around 150-200 words, shorter for simple questions.
Still answer the question completely and use math where needed. Include every
source reference needed to support the answer, with only the short exact quotes
needed for those sources. The length is a guideline, never a reason to omit
necessary evidence, sources or claims.
"""

# Room for a complete structured answer with quotes; explicit settings win.
DEFAULT_GENERATION_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------

def _format_relationships(result: RetrievalResult, max_chars: int | None = None) -> str:
    """Keep hierarchy explicit; semantic connections carry no named predicate."""
    if not result.subgraph:
        return "(no connections retrieved)" if max_chars is None else "(no connections retrieved)"[:max(0, max_chars)]
    lines: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for triple in result.subgraph:
        if triple.relation in {"HAS_SUBTOPIC", "HAS_CONTENT"}:
            identity = ("hierarchy", triple.head, triple.tail)
            line = f"- {triple.head} → {triple.tail} (hierarchy)"
        else:
            head, tail = sorted((triple.head, triple.tail))
            identity = ("association", head, tail)
            line = f"- {head} — {tail} (association)"
        if identity not in seen:
            lines.append(line)
            seen.add(identity)
    text = "\n".join(lines)
    if max_chars is None or len(text) <= max_chars:
        return text
    notice = "... (connections truncated)"
    budget = max(0, max_chars - len(notice) - 1)
    included: list[str] = []
    total = 0
    for line in lines:
        size = len(line) + bool(included)
        if total + size > budget:
            break
        included.append(line)
        total += size
    if len(notice) <= max_chars:
        included.append(notice)
    return "\n".join(included)


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str, cfg: dict) -> _AnswerOutput:
    return structured(prompt, {**cfg, "provider": "ollama"}, _AnswerOutput, system=system)


def _call_anthropic(prompt: str, system: str, cfg: dict) -> _AnswerOutput:
    return structured(prompt, {**cfg, "provider": "anthropic"}, _AnswerOutput, system=system)


def _call_llm(prompt: str, system: str, cfg: dict) -> _AnswerOutput:
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return _call_ollama(prompt, system, cfg)
    if provider == "anthropic":
        return _call_anthropic(prompt, system, cfg)
    if provider == "openrouter":
        cfg = {"structured_mode": "md_json", **cfg}
    return structured(prompt, cfg, _AnswerOutput, system=system)


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


def _validate_citations(output: _AnswerOutput, result: RetrievalResult) -> list[CitationSpan]:
    cited = set(_extract_cited_chunk_ids(output.answer, result))
    validated: list[CitationSpan] = []
    seen: set[str] = set()
    for citation in output.citations:
        if citation.source_number > len(result.chunks):
            continue
        chunk = result.chunks[citation.source_number - 1]
        if chunk.id not in cited or chunk.id in seen:
            continue
        span = find_quote_span(chunk.text, citation.quote)
        if span is None:
            continue
        start, end = span
        validated.append(CitationSpan(chunk_id=chunk.id, source_number=citation.source_number,
                                      quote=chunk.text[start:end], quote_start=start, quote_end=end))
        seen.add(chunk.id)
    return validated


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

    if mode == "none":
        return NO_RETRIEVAL_PROMPT.format(question=question)

    context = _format_chunks(result, max_chars, question=question)

    if mode == "graph":
        relationships = _format_relationships(result, max_chars)
        return GRAPH_RAG_PROMPT.format(
            question=question,
            context=context,
            relationships=relationships,
        )

    if mode == "hybrid":
        relationships = _format_relationships(result, max_chars)
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
    cfg = {"max_tokens": DEFAULT_GENERATION_MAX_TOKENS, **config.get("generator", {})}
    max_chars = generation_context_chars(question, config)

    visible_result = retrieval_result.model_copy(update={
        "chunks": (_context_chunks(retrieval_result, max_chars, question=question)
                   if retrieval_result.routing_mode != "none" else []),
    })
    if retrieval_result.routing_mode != "none" and not any(chunk.text.strip() for chunk in visible_result.chunks):
        # Retrieval and its source filters must not silently become a general-
        # knowledge answer. With no usable evidence, neither generation nor
        # answer reflection can supply grounded facts, so spend no provider call.
        return GenerationResult(
            answer=NO_EVIDENCE_ANSWER,
            routing_mode=retrieval_result.routing_mode,
            reflection=ReflectionVerdict(
                needs_retrieval=True, is_relevant=False, is_supported=False,
                is_useful=False, reasoning="No supporting source passage was available for a grounded answer.",
                verification_status="skipped",
            ),
            evidence_chunk_ids=[], citations=[],
        )
    prompt = _build_prompt(question, retrieval_result, max_chars)
    citations: list[CitationSpan] = []

    system = SYSTEM_PROMPT + "\n\n" + _CITATION_INSTRUCTIONS
    if config.get("query_mode") == "fast":
        system += "\n" + _FAST_ANSWER_INSTRUCTIONS

    failed = False
    try:
        output = _call_llm(prompt, system, cfg)
        answer = output.answer
        citations = _validate_citations(output, visible_result)
    except Exception as exc:
        failed = True
        logger.error(f"LLM generation failed: {exc}")
        answer = (
            "I'm sorry, I was unable to generate a response at this time. "
            "Please try again later."
        )

    if failed:
        # Reflecting on a provider-error placeholder wastes another paid call.
        reflection = ReflectionVerdict(needs_retrieval=bool(retrieval_result.chunks), is_relevant=True,
                                       is_supported=False, is_useful=False, reasoning="Answer generation was unavailable.",
                                       verification_status="unavailable")
    elif config.get("query_mode") == "fast":
        reflection = ReflectionVerdict(needs_retrieval=bool(retrieval_result.chunks), is_relevant=False,
                                       is_supported=False, is_useful=False, reasoning="Fast mode: answer verification was not run.",
                                       verification_status="skipped")
    else:
        # Reflection formats the same question with generation_context_chars,
        # so it sees exactly the numbered passages shown to generation.
        reflection = reflect_answer(question, retrieval_result, answer, config)
        if (retrieval_result.routing_mode != "none" and reflection.verification_status == "verified"
                and not reflection.is_supported):
            # A failed source check must not leave the known unsupported claim
            # as the answer. Keep the retrieved passages available to inspect.
            answer = UNSUPPORTED_ANSWER
            citations = []
    evidence_chunk_ids = _extract_cited_chunk_ids(answer, visible_result)

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
        citations=citations,
    )
