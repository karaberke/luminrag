"""
Shared captioner interface used by all ingestion modules.

Providers:
  - OllamaCaptioner  — local vision model via Ollama (e.g. llama3.2-vision). Fully offline.
  - AnthropicCaptioner — Claude via the Anthropic API. Reads ANTHROPIC_API_KEY from env.

Switch provider with one line in config/llm.yaml:
    captioning:
      provider: ollama   # or: anthropic
"""

from __future__ import annotations

import base64
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


class OllamaCaptioner(BaseCaptioner):
    """Calls a local Ollama vision model (e.g. llama3.2-vision). Fully offline."""

    def __init__(self, model: str, base_url: str, max_tokens: int) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens

    def caption(self, image_path: Path) -> str:
        import httpx

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.model,
            "prompt": _CAPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }
        response = httpx.post(
            f"{self.base_url}/api/generate", json=payload, timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class AnthropicCaptioner(BaseCaptioner):
    """Calls Claude via the Anthropic API. Reads ANTHROPIC_API_KEY from env."""

    def __init__(self, model: str, max_tokens: int) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def caption(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": _CAPTION_PROMPT},
                    ],
                }
            ],
        )
        return message.content[0].text.strip()


def build_captioner(cfg: dict) -> BaseCaptioner:
    """Instantiate the captioner declared in config/llm.yaml."""
    cap_cfg = cfg["captioning"]
    provider = cap_cfg["provider"]
    max_tokens = cap_cfg.get("max_tokens", 150)

    if provider == "ollama":
        return OllamaCaptioner(
            model=cap_cfg["ollama_model"],
            base_url=cap_cfg.get("ollama_base_url", "http://localhost:11434"),
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        return AnthropicCaptioner(
            model=cap_cfg["anthropic_model"],
            max_tokens=max_tokens,
        )
    raise ValueError(
        f"Unknown captioning provider: '{provider}'. Use 'ollama' or 'anthropic'."
    )


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
