"""
Shared captioner interface used by all ingestion modules.

Providers:
  - OllamaCaptioner  — local vision model via Ollama (e.g. llama3.2-vision). Fully offline.
  - AnthropicCaptioner — Claude via the Anthropic API. Uses request credentials, or local environment keys when self hosted.

Switch provider with one line in config/llm.yaml:
    captioning:
      provider: ollama   # or: anthropic
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


_CAPTION_PROMPT = (
    "Describe this lecture slide or classroom visual in 1-2 sentences "
    "for a student study guide. Be concise and factual."
)


class BaseCaptioner(ABC):
    @abstractmethod
    def caption(self, image_path: Path) -> str:
        """Return a text caption for the image at image_path."""


class LLMCaptioner(BaseCaptioner):
    """All vision providers share the same transport and request credentials."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = dict(cfg)

    def caption(self, image_path: Path) -> str:
        from backend.llm import complete
        return complete(_CAPTION_PROMPT, self.cfg, images=[image_path]).strip()


class OllamaCaptioner(LLMCaptioner):
    """Compatibility constructor for self-host integrations."""

    def __init__(self, model: str, base_url: str, max_tokens: int) -> None:
        super().__init__({"provider": "ollama", "model": model, "ollama_base_url": base_url, "max_tokens": max_tokens})


class AnthropicCaptioner(LLMCaptioner):
    """Compatibility constructor; hosted applications use build_captioner."""

    def __init__(self, model: str, max_tokens: int) -> None:
        super().__init__({"provider": "anthropic", "model": model, "max_tokens": max_tokens})


def build_captioner(cfg: dict) -> BaseCaptioner:
    section = cfg.get("captioning", {})
    if section.get("provider", "ollama") not in {"ollama", "anthropic", "openrouter", "openai"}:
        raise ValueError("Unknown captioning provider; choose Ollama, Anthropic, OpenRouter or OpenAI.")
    return LLMCaptioner(section)


def safe_caption(captioner: BaseCaptioner, image_path: Path, label: str) -> str:
    """
    Call ``captioner.caption(image_path)`` and swallow any exception into a
    warning log, returning ``""`` on failure. *label* is inserted into the
    log message so callers can identify the source (e.g. a source_id +
    page number, or a keyframe path).
    """
    try:
        return captioner.caption(image_path)
    except Exception as exc:
        logger.warning(f"Captioning failed ({label}): {exc}")
        return ""
