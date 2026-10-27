"""
Shared LLM-calling + JSON-parsing helpers for the Stage-1 graph extractors.

The four extractors (topic_extractor, subtopic_extractor, content_synthesizer,
semantic_linker) all share the same Ollama/Anthropic provider switch and the
same markdown-fenced-JSON parsing dance. Rather than duplicating that code,
they import `call_llm` and `parse_json_list` from here.

Each extractor's config section is expected to carry:
    provider:             "ollama" | "anthropic"
    ollama_model:         str
    ollama_base_url:      str
    anthropic_model:      str
    max_tokens:           int   (optional, default 512)
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from backend.llm import complete, structured

if TYPE_CHECKING:
    from backend.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


def call_llm(prompt: str, cfg: dict, *, max_tokens: int | None = None,
             response_model: type[BaseModel] | None = None) -> str:
    """Compatibility re-export for integrations from earlier self-host releases."""
    options = cfg if max_tokens is None else {**cfg, "max_tokens": max_tokens}
    if response_model is not None:
        return structured(prompt, options, response_model).model_dump_json()
    return complete(prompt, options)


_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")


def _strip_fences(text: str) -> str:
    cleaned = _FENCE_OPEN.sub("", text.strip())
    cleaned = _FENCE_CLOSE.sub("", cleaned)
    return cleaned.strip()


def parse_json(response: str) -> Any:
    """Parse the first valid JSON value in an LLM response. Strips markdown fences."""
    return json.loads(_strip_fences(response))


def parse_json_list(response: str) -> list[dict]:
    """
    Parse an LLM response that should be a JSON array of objects.
    If the response is a dict wrapping the array under a known key
    ({"items": [...]}, {"results": [...]}, etc.), unwrap it.
    Returns [] on parse failure.
    """
    try:
        value = parse_json(response)
    except json.JSONDecodeError:
        logger.debug(f"Failed to parse LLM JSON: {response[:200]!r}")
        return []

    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        for key in ("items", "results", "data", "topics", "subtopics", "contents", "relations"):
            inner = value.get(key)
            if isinstance(inner, list):
                return [v for v in inner if isinstance(v, dict)]
        # Single-object response — wrap for caller convenience
        return [value]
    return []


def safe_call_json(prompt: str, cfg: dict, *, max_tokens: int | None = None,
                   response_model: type[BaseModel] | None = None) -> list[dict]:
    """Validated list extraction; legacy callers can still parse raw text."""
    try:
        options = cfg if max_tokens is None else {**cfg, "max_tokens": max_tokens}
        if response_model is not None:
            result = structured(prompt + "\nReturn an object with an items array.", options, response_model)
            return result.model_dump(mode="json")["items"]
        return parse_json_list(call_llm(prompt, options))
    except Exception as exc:
        logger.warning("LLM extraction failed (%s)", type(exc).__name__)
        return []


def resolve_synonym(
    proposed: str,
    existing: list[str],
    embedder: "Embedder | None",
    threshold: float,
) -> str:
    """
    Swap *proposed* for the nearest existing name if cosine similarity
    (L2-normalised dot product) meets *threshold*; otherwise return
    *proposed* unchanged. Used by Stage 1.2–1.4 extractors to converge
    synonyms to a single graph key.
    """
    if not existing or embedder is None:
        return proposed
    try:
        candidate = embedder.embed_one(proposed)
        pool = embedder.embed(existing)
        sims = pool @ candidate
        best_idx = int(sims.argmax())
        if float(sims[best_idx]) >= threshold:
            return existing[best_idx]
    except Exception as exc:
        logger.debug(f"Embedding-based synonym lookup failed: {exc}")
    return proposed
