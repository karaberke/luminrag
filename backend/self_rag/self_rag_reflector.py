"""
Stage 5 - Self-RAG Reflection: self_rag_reflector.py

Verification gate between retrieval and generation.
The LLM evaluates retrieved context and (after generation) the answer,
returning a structured ReflectionVerdict.

Two evaluation points:

  reflect_retrieval(question, retrieval_result, config)
      Called after retrieval, before generation.
      Evaluates: needs_retrieval, is_relevant.
      is_supported / is_useful are set to True as neutral placeholders.
      → If is_relevant=False, the caller should re-route or re-retrieve.

  reflect_answer(question, retrieval_result, answer, config)
      Called after generation.
      Evaluates all four fields including is_supported and is_useful.
      → If is_supported=False, flag answer as potentially hallucinated.
      → If is_useful=False, flag answer as off-topic.

On any LLM failure (network error, bad JSON, timeout) both functions
return a safe pass-through verdict so the pipeline never hard-crashes.
"""

from __future__ import annotations

import json
import logging
import re

from backend.llm import structured
from backend.schemas import ReflectionVerdict, RetrievalResult
from backend.utils.context import DEFAULT_GENERATION_CONTEXT_CHARS, format_context as _format_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe default — returned on any reflector failure
# ---------------------------------------------------------------------------

_SAFE_DEFAULT = ReflectionVerdict(
    needs_retrieval=True,
    is_relevant=True,
    is_supported=True,
    is_useful=True,
    reasoning="Reflection unavailable — defaulting to pass-through.",
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RETRIEVAL_PROMPT = """\
You are a quality evaluator for an educational QA system.

Student question:
{question}

Retrieved context:
{context}

Evaluate and respond with ONLY valid JSON (no markdown fences):

1. needs_retrieval: Does this question require external knowledge, or can a \
language model answer it accurately from general knowledge alone?
2. is_relevant: Is the retrieved context relevant and useful for answering \
the student's question?

Required JSON format:
{{
  "needs_retrieval": <true or false>,
  "is_relevant": <true or false>,
  "is_supported": true,
  "is_useful": true,
  "reasoning": "<one concise sentence>"
}}\
"""

_ANSWER_PROMPT = """\
You are a quality evaluator for an educational QA system.

Student question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Evaluate and respond with ONLY valid JSON (no markdown fences):

1. needs_retrieval: Was external context necessary to answer this question?
2. is_relevant: Is the retrieved context relevant to the question?
3. is_supported: {support_criterion}
4. is_useful: Does the generated answer directly and substantively address the \
student's question?

Required JSON format:
{{
  "needs_retrieval": <true or false>,
  "is_relevant": <true or false>,
  "is_supported": <true or false>,
  "is_useful": <true or false>,
  "reasoning": "<one concise sentence>"
}}\
"""

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_verdict(response: str) -> ReflectionVerdict:
    """
    Parse a JSON ReflectionVerdict from an LLM response.
    Strips markdown code fences if present.
    Raises ValueError / json.JSONDecodeError on failure.
    """
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())

    data = json.loads(text)

    return ReflectionVerdict(
        needs_retrieval=bool(data.get("needs_retrieval", True)),
        is_relevant=bool(data.get("is_relevant", True)),
        is_supported=bool(data.get("is_supported", True)),
        is_useful=bool(data.get("is_useful", True)),
        reasoning=str(data.get("reasoning", "")),
    )


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, cfg: dict) -> str:
    return structured(prompt, {**cfg, "provider": "ollama"}, ReflectionVerdict).model_dump_json()


def _call_anthropic(prompt: str, cfg: dict) -> str:
    return structured(prompt, {**cfg, "provider": "anthropic"}, ReflectionVerdict).model_dump_json()


def _call_llm(prompt: str, cfg: dict) -> str:
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return _call_ollama(prompt, cfg)
    if provider == "anthropic":
        return _call_anthropic(prompt, cfg)
    return structured(prompt, cfg, ReflectionVerdict).model_dump_json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reflect_retrieval(
    question: str,
    retrieval_result: RetrievalResult,
    config: dict,
) -> ReflectionVerdict:
    """
    Evaluate whether the retrieved context is relevant to the question.

    Call this after retrieval and before generation.
    If verdict.is_relevant is False, the caller should re-route or re-retrieve.

    Args:
        question:         Raw user question.
        retrieval_result: Output from vector_retriever or graph_retriever.
        config:           Parsed config/llm.yaml.

    Returns:
        ReflectionVerdict with needs_retrieval and is_relevant populated.
        is_supported and is_useful are True placeholders (no answer yet).
    """
    cfg = config.get("self_rag", {})
    max_chars = cfg.get("max_context_chars", 2000)

    context = _format_context(retrieval_result, max_chars)
    prompt = _RETRIEVAL_PROMPT.format(question=question, context=context)

    try:
        raw = _call_llm(prompt, cfg)
        verdict = _parse_verdict(raw)
        logger.debug(
            f"Retrieval reflection — relevant={verdict.is_relevant} "
            f"needs_retrieval={verdict.needs_retrieval} | {verdict.reasoning}"
        )
        return verdict
    except Exception as exc:
        logger.warning(f"reflect_retrieval failed: {exc} — using safe default")
        return _SAFE_DEFAULT


def reflect_answer(
    question: str,
    retrieval_result: RetrievalResult,
    answer: str,
    config: dict,
) -> ReflectionVerdict:
    """
    Evaluate whether the generated answer is supported by context and useful.

    Call this after generation to gate the final response.
    If verdict.is_supported is False, flag the answer as potentially hallucinated.
    If verdict.is_useful is False, flag the answer as off-topic.

    Args:
        question:         Raw user question.
        retrieval_result: The context used during generation.
        answer:           The LLM-generated answer text.
        config:           Parsed config/llm.yaml.

    Returns:
        ReflectionVerdict with all four fields populated.
    """
    cfg = config.get("self_rag", {})
    max_chars = config.get("generator", {}).get("max_context_chars", DEFAULT_GENERATION_CONTEXT_CHARS)
    conversation = retrieval_result.routing_mode == "none"
    context = "(no source context required)" if conversation else _format_context(retrieval_result, max_chars)
    support_criterion = (
        "For this ordinary conversation, assess factual accuracy using general knowledge. No course-source citation is required."
        if conversation else
        "Every factual claim must be supported by the retrieved passages shown here. Mark false for unsupported claims, "
        "contradictions, or invented facts; accurate general background alone cannot supply missing source evidence. "
        "Source numbers refer only to these visible passages."
    )
    prompt = _ANSWER_PROMPT.format(
        question=question, context=context, answer=answer, support_criterion=support_criterion
    )

    try:
        raw = _call_llm(prompt, cfg)
        verdict = _parse_verdict(raw)
        logger.debug(
            f"Answer reflection — supported={verdict.is_supported} "
            f"useful={verdict.is_useful} | {verdict.reasoning}"
        )
        return verdict
    except Exception as exc:
        logger.warning(f"reflect_answer failed: {exc} — using safe default")
        return _SAFE_DEFAULT
