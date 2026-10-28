"""The sole provider transport for text, vision and validated structured output.

Clients are short lived and request scoped. In hosted mode their credentials
must be explicit and their remote endpoints cannot be overridden by callers.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from backend.credentials import is_hosted

T = TypeVar("T", bound=BaseModel)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openrouter": "anthropic/claude-haiku-4.5",
    "openai": "gpt-5.4-mini",
    "ollama": "qwen2.5:7b",
}
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}
# These model families reject forced tool choice. Native schema output avoids
# that incompatibility without another provider request or model lookup.
_NATIVE_ANTHROPIC = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1")


class LLMError(RuntimeError):
    """A sanitized transport failure, safe to return or log without a key."""


def _settings(cfg: dict) -> tuple[str, str, str, int, float]:
    provider = cfg.get("provider", cfg.get("llm_provider", "ollama"))
    hosted = is_hosted() or cfg.get("hosted", False)
    if provider not in _DEFAULT_MODELS or (hosted and provider == "ollama"):
        raise LLMError("This LLM provider is unavailable.")
    model = cfg.get("model") or cfg.get("llm_model") or cfg.get(f"{provider}_model") or cfg.get(f"llm_{provider}_model") or _DEFAULT_MODELS[provider]
    key = cfg.get("api_key")
    if not key and not hosted and provider in _KEY_ENV:
        key = os.getenv(_KEY_ENV[provider], "")
    if provider != "ollama" and not key:
        raise LLMError("Connect your LLM provider in Settings.")
    return provider, str(model), str(key or "ollama"), min(32768, max(1, int(cfg.get("max_tokens", cfg.get("llm_max_tokens", 1024))))), min(300.0, max(1.0, float(cfg.get("timeout", 120))))


def _image_parts(images: list | None, provider: str) -> list[dict]:
    parts = []
    for image in images or []:
        path = Path(image)
        if path.stat().st_size > 20 * 1024 * 1024:
            raise LLMError("Image is too large for captioning.")
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        if media_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
            raise LLMError("Unsupported image type for captioning.")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        if provider == "anthropic":
            parts.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        else:
            parts.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
    return parts


def _client(provider: str, key: str, cfg: dict, timeout: float):
    retries = min(2, max(0, int(cfg.get("sdk_retries", 2))))
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=key, base_url=ANTHROPIC_BASE_URL, timeout=timeout, max_retries=retries)
    from openai import OpenAI
    base_url = OPENAI_BASE_URL if provider == "openai" else OPENROUTER_BASE_URL
    if provider == "ollama":
        base_url = str(cfg.get("ollama_base_url", cfg.get("llm_ollama_base_url", "http://localhost:11434"))).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=retries)


def _anthropic_options(model: str) -> dict:
    """Avoid spending small extraction budgets on optional up-front thinking."""
    if model.startswith("claude-sonnet-5-5"):
        return {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}
    if model.startswith("claude-haiku-5-5"):
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    if model.startswith(_NATIVE_ANTHROPIC):
        return {"output_config": {"effort": "low"}}
    return {}


def _completion_options(provider: str, tokens: int) -> dict:
    # OpenAI's completion budget includes reasoning tokens. Avoid temperature
    # overrides so reasoning models can use the same text and vision transport.
    if provider == "openai":
        return {"max_completion_tokens": tokens, "store": False}
    return {"max_tokens": tokens}


def _validate_completion(text: str | None, stop_reason: str | None, refusal: str | None = None) -> str:
    if stop_reason in {"max_tokens", "length"}:
        raise LLMError("The model reached its token limit. Choose a lighter model or increase its token budget.")
    if refusal or stop_reason in {"refusal", "content_filter", "error"}:
        raise LLMError("The provider could not complete this response. Try a different model or request.")
    if not isinstance(text, str) or not text.strip():
        raise LLMError("The provider returned an empty response. Check the model and its token budget.")
    return text


def _messages(prompt: str, system: str | None, images: list | None, provider: str) -> list[dict]:
    parts = _image_parts(images, provider)
    content = [*parts, {"type": "text", "text": prompt}] if parts else prompt
    messages = [{"role": "user", "content": content}]
    if system and provider != "anthropic":
        messages.insert(0, {"role": "system", "content": system})
    return messages


def complete(prompt: str, cfg: dict, system: str | None = None, images: list | None = None) -> str:
    provider, model, key, tokens, timeout = _settings(cfg)
    try:
        with _client(provider, key, cfg, timeout) as client:
            messages = _messages(prompt, system, images, provider)
            if provider == "anthropic":
                kwargs = {**_anthropic_options(model), **({"system": system} if system else {})}
                response = client.messages.create(model=model, max_tokens=tokens, messages=messages, **kwargs)
                text = "".join(part.text for part in response.content if getattr(part, "type", "text") == "text")
                return _validate_completion(text, response.stop_reason)
            response = client.chat.completions.create(model=model, messages=messages, **_completion_options(provider, tokens))
            return _validate_completion(response.choices[0].message.content, response.choices[0].finish_reason, response.choices[0].message.refusal)
    except LLMError:
        raise
    except Exception:
        # SDK exceptions can contain request bodies, including credentials.
        raise LLMError(f"{provider} request failed; check your key, model and spending limit.") from None


def structured(prompt: str, cfg: dict, response_model: type[T], system: str | None = None, images: list | None = None) -> T:
    """Validate with Pydantic and retry schema failures through Instructor."""
    provider, model, key, tokens, timeout = _settings(cfg)
    try:
        import instructor
        # Instructor owns retrying schema failures. SDK retries must not multiply
        # its attempt count and unexpectedly charge a user's key up to nine times.
        with _client(provider, key, {**cfg, "sdk_retries": 0}, timeout) as client:
            messages = _messages(prompt, system, images, provider)
            retries = min(2, max(0, int(cfg.get("structured_retries", 1))))
            mode_name = cfg.get("structured_mode") or ("native" if provider == "openai" else None)
            native = mode_name in {"native", "json_schema"}
            if provider == "anthropic":
                kwargs = {**_anthropic_options(model), **({"system": system} if system else {})}
                if native or model.startswith(_NATIVE_ANTHROPIC):
                    response = client.messages.parse(model=model, max_tokens=tokens, messages=messages, output_format=response_model, **kwargs)
                    _validate_completion("".join(part.text for part in response.content if part.type == "text"), response.stop_reason)
                    result = response.parsed_output
                else:
                    mode = instructor.Mode.JSON if cfg.get("structured_mode") == "json" else instructor.Mode.TOOLS
                    wrapped = instructor.from_anthropic(client, mode=mode)
                    result = wrapped.messages.create(model=model, max_tokens=tokens, messages=messages, response_model=response_model, max_retries=retries, **kwargs)
            else:
                # MD_JSON asks for schema-shaped text without API-specific
                # response_format/tools, so arbitrary routed models can use it.
                prompt_json = mode_name in {"md_json", "prompt"}
                routing = {"extra_body": {"provider": {"require_parameters": True}}} if provider == "openrouter" and not prompt_json else {}
                if native:
                    response = client.chat.completions.parse(model=model, messages=messages, response_format=response_model, **_completion_options(provider, tokens), **routing)
                    _validate_completion(response.choices[0].message.content, response.choices[0].finish_reason, response.choices[0].message.refusal)
                    result = response.choices[0].message.parsed
                else:
                    mode = instructor.Mode.MD_JSON if prompt_json else (instructor.Mode.TOOLS if mode_name == "tools" else instructor.Mode.JSON)
                    wrapped = instructor.from_openai(client, mode=mode)
                    result = wrapped.chat.completions.create(model=model, messages=messages, response_model=response_model, max_retries=retries, **_completion_options(provider, tokens), **routing)
            # Instructor subclasses may carry transport-private attributes;
            # return an ordinary domain model across module boundaries.
            return response_model.model_validate(result.model_dump())
    except LLMError:
        raise
    except Exception:
        raise LLMError(f"{provider} did not return valid structured output; please retry.") from None
