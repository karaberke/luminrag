"""Per-request BYOK credentials; no process-global client or key mutation."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from backend.utils.environment import hosted_mode

Provider = Literal["anthropic", "openrouter", "openai", "ollama"]
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")
_FAST_SECTIONS = (
    "captioning", "topic_extractor", "subtopic_extractor", "content_synthesizer",
    "semantic_linker", "query_router", "multi_hop_reasoner", "self_rag",
)
_SMART_SECTIONS = ("generator",)


class LLMCredentials(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: Provider
    api_key: SecretStr = Field(repr=False)
    model_fast: str
    model_smart: str


def is_hosted() -> bool:
    return hosted_mode()


def get_credentials(request: Request) -> LLMCredentials | None:
    """Require explicit headers in hosted mode; preserve self-host defaults."""
    headers = request.headers
    values = [headers.get(name, "").strip() for name in (
        "X-LLM-Provider", "X-LLM-Key", "X-LLM-Model-Fast", "X-LLM-Model-Smart",
    )]
    if not any(values) and not is_hosted():
        return None
    provider, key, fast, smart = values
    if provider not in ("anthropic", "openrouter", "openai", "ollama") or (is_hosted() and provider == "ollama"):
        raise HTTPException(401, "Connect OpenAI, OpenRouter or Anthropic in Settings.")
    if provider != "ollama" and (not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise HTTPException(401, "A valid provider key is required.")
    if not _MODEL.fullmatch(fast) or not _MODEL.fullmatch(smart):
        raise HTTPException(401, "Choose both a fast model and a smart model in Settings.")
    return LLMCredentials(provider=provider, api_key=SecretStr(key), model_fast=fast, model_smart=smart)


def apply_credentials(config: dict, creds: LLMCredentials | None) -> dict:
    """Detach config before passing it across worker threads and executors."""
    result = deepcopy(config)
    if creds is None:
        if is_hosted():
            # A config created without credentials must never inherit local keys.
            for section in _FAST_SECTIONS + _SMART_SECTIONS:
                current = result.setdefault(section, {})
                current.pop("api_key", None)
                current["hosted"] = True
        return result
    for section in _FAST_SECTIONS + _SMART_SECTIONS:
        current = result.setdefault(section, {})
        model = creds.model_smart if section in _SMART_SECTIONS else creds.model_fast
        current.update(provider=creds.provider, api_key=creds.api_key.get_secret_value(), model=model, hosted=is_hosted())
        # Config from old self-host releases uses provider-specific names.
        current[f"{creds.provider}_model"] = model
        if section == "query_router":
            current.update(llm_provider=creds.provider, llm_model=model)
    return result
