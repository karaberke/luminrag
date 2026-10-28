"""The sole provider transport for text, vision and validated structured output.

Clients are short lived and request scoped. In hosted mode their credentials
must be explicit and their remote endpoints cannot be overridden by callers.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from backend.credentials import is_hosted

T = TypeVar("T", bound=BaseModel)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openrouter": "anthropic/claude-haiku-4.5",
    "openai": "gpt-5.4-mini",
    "ollama_cloud": "gemma4:31b",
    "ollama": "qwen2.5:7b",
}
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}
# These model families reject forced tool choice. Native schema output avoids
# that incompatibility without another provider request or model lookup.
_NATIVE_ANTHROPIC = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1", "claude-mythos-5-1")

# Provider SDKs can log request/response bodies and invalid generated content
# before raising. Callers report sanitized LLMError instead of these bodies.
for _name in ("instructor", "openai", "anthropic"):
    _provider_logger = logging.getLogger(_name)
    _provider_logger.addHandler(logging.NullHandler())
    _provider_logger.setLevel(logging.CRITICAL + 1)
    _provider_logger.propagate = False


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
        from anthropic import Anthropic, DefaultHttpxClient
        logging.getLogger("anthropic").setLevel(logging.CRITICAL + 1)
        return Anthropic(api_key=key, base_url=ANTHROPIC_BASE_URL, timeout=timeout, max_retries=retries,
                         http_client=DefaultHttpxClient(follow_redirects=False))
    from openai import DefaultHttpxClient, OpenAI
    logging.getLogger("openai").setLevel(logging.CRITICAL + 1)
    base_url = {
        "openai": OPENAI_BASE_URL, "ollama_cloud": OLLAMA_CLOUD_BASE_URL,
    }.get(provider, OPENROUTER_BASE_URL)
    if provider == "ollama":
        base_url = str(cfg.get("ollama_base_url", cfg.get("llm_ollama_base_url", "http://localhost:11434"))).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
    # Even a redirect must not replay a visitor's source context to a new host.
    options = {"http_client": DefaultHttpxClient(follow_redirects=False)} if provider != "ollama" else {}
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=retries, **options)


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
    if provider == "ollama_cloud":
        return {"max_tokens": tokens, "reasoning_effort": "none"}
    return {"max_tokens": tokens}


def _provider_error(provider: str, error: Exception, *, structured_output: bool = False) -> LLMError:
    """Classify only trusted exception types; never expose vendor text or bodies."""
    if provider == "ollama_cloud":
        from openai import APIConnectionError, APIStatusError, APITimeoutError
        from instructor.core.exceptions import IncompleteOutputException

        current: BaseException | None = error
        seen: set[int] = set()
        for _ in range(8):
            if current is None or id(current) in seen:
                break
            seen.add(id(current))
            if isinstance(current, APITimeoutError):
                return LLMError("Ollama Cloud timed out. Please retry later.")
            if isinstance(current, APIStatusError):
                if 300 <= current.status_code < 400:
                    return LLMError("Ollama Cloud returned an unexpected redirect. Please retry later.")
                if current.status_code in {401, 403}:
                    return LLMError("Ollama Cloud authentication failed. Check your API key and account access.")
                if current.status_code == 404:
                    return LLMError("The selected Ollama Cloud model is unavailable. Choose an available Cloud API model ID.")
                if current.status_code == 429:
                    return LLMError("Ollama Cloud usage or rate limit reached. Check your account allowance and retry later.")
                if current.status_code >= 500:
                    return LLMError("Ollama Cloud is temporarily unavailable. Please retry later.")
                return LLMError("Ollama Cloud rejected the request. Check model support and token limits.")
            if isinstance(current, APIConnectionError):
                return LLMError("Could not connect to Ollama Cloud. Please retry later.")
            if isinstance(current, IncompleteOutputException):
                return LLMError("The model reached its token limit. Choose a lighter model or increase its token budget.")
            current = current.__cause__ or current.__context__
        if structured_output:
            return LLMError("Ollama Cloud did not return valid structured output. Please retry or choose a different model.")
        return LLMError("Ollama Cloud request failed. Check your key, model and account allowance.")
    if structured_output:
        return LLMError(f"{provider} did not return valid structured output; please retry.")
    return LLMError(f"{provider} request failed; check your key, model and spending limit.")


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


def _anthropic_output_format(response_model: type[BaseModel]) -> dict:
    """Build the SDK's native schema, retaining supported top-level string patterns.

    Schemas are application-defined, never supplied by documents or visitors.
    The SDK still removes unsupported bounds; Pydantic checks them locally.
    """
    from anthropic import transform_schema
    schema = response_model.model_json_schema()
    patterns = {name: field["pattern"] for name, field in schema.get("properties", {}).items()
                if field.get("type") == "string" and "pattern" in field}
    schema = transform_schema(schema)
    for name, pattern in patterns.items():
        schema["properties"][name]["pattern"] = pattern
    return {"type": "json_schema", "schema": schema}


def _correction_messages(messages: list[dict], error: ValidationError, key: str) -> list[dict]:
    """Copy the user turns with bounded feedback that never contains input values.

    Invalid output is not replayed. Pydantic locations, types and messages
    describe the schema; inputs, contexts and URLs are excluded explicitly.
    """
    lines = []
    for item in error.errors(include_input=False, include_context=False, include_url=False)[:4]:
        path = ".".join(str(part) for part in item.get("loc", ())) or "(root)"
        lines.append("- " + "; ".join(str(value)[:200] for value in (path, item.get("type", ""), item.get("msg", ""))))
    feedback = ("Correction: the previous response failed schema validation. Return the complete JSON object "
                "again with these errors fixed. Parent indices are zero-based and must refer to items actually "
                "returned in this response.\n" + "\n".join(lines))
    feedback = (feedback.replace(key, "[redacted]") if key else feedback)[:1500]
    corrected = [dict(message) for message in messages]
    content = corrected[-1]["content"]
    corrected[-1]["content"] = f"{content}\n\n{feedback}" if isinstance(content, str) else [*content, {"type": "text", "text": feedback}]
    return corrected


def _anthropic_native(client, response_model: type[T], messages: list[dict], retries: int, key: str, **request) -> T:
    """Reject incomplete output before validation; retry only invalid complete JSON.

    messages.parse validates before stop_reason is checked, so a truncated
    response could otherwise be retried as if it were a schema failure.
    """
    attempt_messages = messages
    for remaining in range(retries, -1, -1):
        response = client.messages.create(messages=attempt_messages, **request)
        text = _validate_completion("".join(part.text for part in response.content if part.type == "text"), response.stop_reason)
        try:
            return response_model.model_validate_json(text)
        except ValidationError as error:
            if not remaining:
                raise
            attempt_messages = _correction_messages(messages, error, key)


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
    except Exception as exc:
        # SDK exceptions can contain request bodies, including credentials.
        raise _provider_error(provider, exc) from None


def structured(prompt: str, cfg: dict, response_model: type[T], system: str | None = None, images: list | None = None) -> T:
    """Validate with Pydantic and retry only bounded schema failures."""
    provider, model, key, tokens, timeout = _settings(cfg)
    try:
        import instructor
        # Instructor owns retrying schema failures. SDK retries must not multiply
        # its attempt count and unexpectedly charge a user's key up to nine times.
        with _client(provider, key, {**cfg, "sdk_retries": 0}, timeout) as client:
            messages = _messages(prompt, system, images, provider)
            retries = min(2, max(0, int(cfg.get("structured_retries", 1))))
            # Cloud has no native structured-output support. Configuration left
            # over from another provider must never forward schemas or tools.
            mode_name = "md_json" if provider == "ollama_cloud" else (
                cfg.get("structured_mode") or ("native" if provider == "openai" else None)
            )
            native = mode_name in {"native", "json_schema"}
            if provider == "anthropic":
                kwargs = {**_anthropic_options(model), **({"system": system} if system else {})}
                if native or model.startswith(_NATIVE_ANTHROPIC):
                    # Native output has no Instructor; retries apply only to
                    # local ValidationError of complete output in this call.
                    kwargs["output_config"] = {**kwargs.get("output_config", {}), "format": _anthropic_output_format(response_model)}
                    result = _anthropic_native(client, response_model, messages, retries, key,
                                               model=model, max_tokens=tokens, **kwargs)
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
    except Exception as exc:
        raise _provider_error(provider, exc, structured_output=True) from None
