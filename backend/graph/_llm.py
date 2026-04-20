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
from typing import Any

logger = logging.getLogger(__name__)


def call_llm(prompt: str, cfg: dict, *, max_tokens: int | None = None) -> str:
    """
    Run a one-shot prompt against the configured provider. Returns raw
    response text. Raises on transport errors.
    """
    provider = cfg.get("provider", "ollama")
    tokens = max_tokens if max_tokens is not None else cfg.get("max_tokens", 512)

    if provider == "ollama":
        return _call_ollama(prompt, cfg, tokens)
    if provider == "anthropic":
        return _call_anthropic(prompt, cfg, tokens)
    raise ValueError(f"Unknown provider: '{provider}'")


def _call_ollama(prompt: str, cfg: dict, max_tokens: int) -> str:
    import httpx

    payload = {
        "model": cfg.get("ollama_model", "llama3.2:3b"),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_predict": max_tokens},
    }
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def _call_anthropic(prompt: str, cfg: dict, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=cfg.get("anthropic_model", "claude-haiku-4-5-20251001"),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


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


def safe_call_json(prompt: str, cfg: dict, *, max_tokens: int | None = None) -> list[dict]:
    """
    Convenience: call the LLM and parse the response as a JSON array.
    Returns [] on any failure (logged at WARN).
    """
    try:
        raw = call_llm(prompt, cfg, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning(f"LLM call failed: {exc}")
        return []
    return parse_json_list(raw)
