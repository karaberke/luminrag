"""Provider contract tests exercise SDK serialization without external calls."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException
from pydantic import BaseModel, SecretStr
from starlette.requests import Request

from backend import credentials, llm
from backend.credentials import LLMCredentials, apply_credentials, get_credentials


def _request(**headers):
    return Request({"type": "http", "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()]})


def test_hosted_credentials_are_required_even_when_server_has_a_key(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key-never-use")
    with pytest.raises(HTTPException) as error:
        get_credentials(_request())
    assert error.value.status_code == 401
    with pytest.raises(llm.LLMError, match="Connect"):
        llm.complete("hello", {"provider": "anthropic"})


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
def test_credentials_copy_config_and_route_fast_and_smart_models(monkeypatch, provider):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    original = {"generator": {"nested": {"value": 1}}, "query_router": {"llm_provider": "ollama"}}
    creds = get_credentials(_request(**{
        "X-LLM-Provider": provider, "X-LLM-Key": "user-owned-key",
        "X-LLM-Model-Fast": "vendor/fast", "X-LLM-Model-Smart": "vendor/smart",
    }))
    result = apply_credentials(original, creds)
    assert result["generator"]["model"] == "vendor/smart"
    for section in credentials._FAST_SECTIONS:
        assert result[section]["model"] == "vendor/fast"
        assert result[section]["api_key"] == "user-owned-key"
    assert result["query_router"]["llm_provider"] == provider
    result["generator"]["nested"]["value"] = 2
    assert original["generator"] == {"nested": {"value": 1}}
    assert "user-owned-key" not in repr(creds)


def test_self_host_without_headers_preserves_local_config(monkeypatch):
    monkeypatch.delenv("LUMIN_HOSTED", raising=False)
    assert get_credentials(_request()) is None
    original = {"generator": {"provider": "ollama", "model": "local-model"}}
    assert apply_credentials(original, None) == original
    assert apply_credentials(original, None) is not original


@pytest.mark.parametrize("headers", [
    {"X-LLM-Provider": "ollama"},
    {"X-LLM-Provider": "https://attacker.invalid", "X-LLM-Key": "key"},
    {"X-LLM-Provider": "anthropic", "X-LLM-Key": "bad key"},
    {"X-LLM-Provider": "anthropic", "X-LLM-Key": "key", "X-LLM-Model-Fast": "model\nInjected"},
])
def test_hosted_credentials_reject_partial_or_invalid_headers(monkeypatch, headers):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    with pytest.raises(HTTPException) as error:
        get_credentials(_request(**headers))
    assert error.value.status_code == 401


class Answer(BaseModel):
    answer: int


@pytest.fixture
def provider_transport(monkeypatch):
    calls = []
    replies = []
    overrides = {}
    client_options = []

    def respond(request):
        data = json.loads(request.content)
        calls.append((request, data))
        answer = replies.pop(0) if replies else 42
        response_data = answer if isinstance(answer, dict) else {"answer": answer}
        if request.url.host == "api.anthropic.com":
            native = data.get("output_config", {}).get("format")
            content = [{"type": "tool_use", "id": "toolu_test", "name": data["tools"][0]["name"], "input": response_data}] if data.get("tools") else [{"type": "text", "text": json.dumps(response_data) if native or isinstance(answer, dict) else "hello"}]
            payload = {"id": "msg_test", "type": "message", "role": "assistant", "model": "test-model", "content": content, "stop_reason": "tool_use" if data.get("tools") else "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            payload = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "test-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(response_data) if data.get("response_format") or isinstance(answer, dict) else "hello"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        payload.update(overrides)
        return httpx.Response(200, json=payload)

    def make_client(provider, key, cfg, timeout):
        client_options.append(dict(cfg))
        transport = httpx.Client(transport=httpx.MockTransport(respond))
        if provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=key, http_client=transport, max_retries=0)
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=llm.OPENROUTER_BASE_URL if provider == "openrouter" else "http://localhost:11434/v1", http_client=transport, max_retries=0)

    monkeypatch.setattr(llm, "_client", make_client)
    monkeypatch.delenv("LUMIN_HOSTED", raising=False)
    return SimpleNamespace(calls=calls, replies=replies, overrides=overrides, client_options=client_options)


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "ollama"])
def test_complete_uses_actual_sdk_with_request_key(provider_transport, provider, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key-never-use")
    cfg = {"provider": provider, "api_key": "user-owned-key", "model": "test-model"}
    assert llm.complete("hello", cfg, system="Tutor") == "hello"
    request, data = provider_transport.calls[0]
    key = request.headers.get("x-api-key") or request.headers.get("authorization")
    assert "user-owned-key" in key and "server-key" not in key
    assert data["model"] == "test-model"


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "ollama"])
def test_structured_validates_with_instructor_and_retries_bad_types(provider_transport, provider):
    provider_transport.replies[:] = ["not-an-integer", 42]
    result = llm.structured("Return answer 42", {"provider": provider, "api_key": "user-key", "model": "test-model"}, Answer)
    assert result == Answer(answer=42)
    assert len(provider_transport.calls) == 2
    payload = provider_transport.calls[0][1]
    assert "tools" in payload if provider == "anthropic" else payload["response_format"]["type"] == "json_object"
    assert provider_transport.client_options[0]["sdk_retries"] == 0


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "ollama"])
def test_structured_retry_count_is_bounded_without_sdk_multiplier(provider_transport, provider):
    provider_transport.replies[:] = ["bad-type"] * 5
    with pytest.raises(llm.LLMError):
        llm.structured("Return answer", {"provider": provider, "api_key": "key", "model": "test-model"}, Answer)
    assert len(provider_transport.calls) == 2
    assert all(cfg["sdk_retries"] == 0 for cfg in provider_transport.client_options)


@pytest.mark.parametrize("model,thinking", [("claude-opus-5-5", None), ("claude-sonnet-5-5", {"type": "between_tools"})])
def test_new_claude_models_use_native_schema_without_forced_tools(provider_transport, model, thinking):
    result = llm.structured("Return answer 42", {"provider": "anthropic", "api_key": "key", "model": model}, Answer)
    assert result.answer == 42
    assert len(provider_transport.calls) == 1
    payload = provider_transport.calls[0][1]
    assert "tools" not in payload and "tool_choice" not in payload
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload.get("thinking") == thinking


def test_openrouter_native_schema_is_strict_and_requires_parameter_support(provider_transport):
    result = llm.structured("Return answer 42", {"provider": "openrouter", "api_key": "key", "model": "test-model", "structured_mode": "native"}, Answer)
    assert result.answer == 42
    payload = provider_transport.calls[0][1]
    assert payload["provider"]["require_parameters"]
    assert payload["response_format"]["json_schema"]["strict"]
    assert payload["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "ollama"])
def test_final_answer_and_validated_quote_share_one_sdk_request(provider_transport, provider):
    from backend.generation import generator
    from backend.schemas import Chunk, RetrievalResult

    provider_transport.replies[:] = [{"answer": "Enzymes lower activation energy [1].", "citations": [
        {"source_number": 1, "quote": "Enzymes lower activation energy."},
    ]}]
    retrieved = RetrievalResult(chunks=[Chunk(id="bio", source_id="bio.pdf", modality="pdf",
                                              text="Enzymes lower activation energy.")], subgraph=[], routing_mode="dense")
    output = generator.generate("How do enzymes work?", retrieved, {"query_mode": "fast", "generator": {
        "provider": provider, "api_key": "request-key", "model": "test-model",
    }})
    assert len(provider_transport.calls) == 1
    assert output.citations[0].quote == "Enzymes lower activation energy."
    assert output.citations[0].chunk_id == "bio"
    assert output.citations[0].source_number == 1
    assert provider_transport.client_options[0]["sdk_retries"] == 0
    payload = provider_transport.calls[0][1]
    if provider == "openrouter":
        assert "response_format" not in payload and "tools" not in payload
        assert "provider" not in payload
        assert any("schema" in str(message).lower() for message in payload["messages"])


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
@pytest.mark.parametrize("truncated", [False, True])
def test_empty_and_truncated_completions_are_explicit_failures(provider_transport, provider, truncated):
    if provider == "anthropic":
        provider_transport.overrides.update(content=[{"type": "text", "text": "partial" if truncated else ""}], stop_reason="max_tokens" if truncated else "end_turn")
    else:
        provider_transport.overrides["choices"] = [{"index": 0, "message": {"role": "assistant", "content": "partial" if truncated else ""}, "finish_reason": "length" if truncated else "stop"}]
    with pytest.raises(llm.LLMError, match="token limit" if truncated else "empty response"):
        llm.complete("hello", {"provider": provider, "api_key": "key", "model": "test-model"})


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
def test_vision_uses_correct_mime_and_standard_content_parts(provider_transport, provider, tmp_path):
    path = tmp_path / "slide.png"
    path.write_bytes(b"png-image-bytes")
    llm.complete("Caption this", {"provider": provider, "api_key": "key", "model": "vision"}, images=[path])
    content = provider_transport.calls[0][1]["messages"][-1]["content"]
    if provider == "anthropic":
        assert content[0]["source"]["media_type"] == "image/png"
    else:
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_transport_error_never_exposes_key(monkeypatch):
    monkeypatch.delenv("LUMIN_HOSTED", raising=False)
    monkeypatch.setattr(llm, "_client", Mock(side_effect=ValueError("secret-key-in-sdk-error")))
    with pytest.raises(llm.LLMError) as error:
        llm.complete("hello", {"provider": "anthropic", "api_key": "secret-key-in-sdk-error"})
    assert "secret-key" not in str(error.value)
    assert error.value.__suppress_context__


def test_simultaneous_credentials_never_mutate_shared_config(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    base = {"subtopic_extractor": {"provider": "ollama", "max_workers": 2}}
    barrier = threading.Barrier(2)

    def run(identifier):
        cfg = apply_credentials(base, LLMCredentials(provider="anthropic", api_key=SecretStr("key-" + identifier),
                                model_fast="fast-" + identifier, model_smart="smart-" + identifier))
        barrier.wait(timeout=5)
        # Each request may fan out in a separate ThreadPoolExecutor. Credentials
        # travel in the explicit config, independently of ContextVar inheritance.
        with ThreadPoolExecutor(max_workers=2) as workers:
            return list(workers.map(lambda _: llm._settings(cfg["subtopic_extractor"])[1:3], range(2)))

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(run, ["one", "two"]))
    assert results == [[("fast-one", "key-one")] * 2, [("fast-two", "key-two")] * 2]
    assert base == {"subtopic_extractor": {"provider": "ollama", "max_workers": 2}}


def test_fast_query_skips_decomposition_and_both_reflections(monkeypatch):
    from backend.generation import generator
    from backend.self_rag import multi_hop_reasoner
    from backend.schemas import Chunk, RetrievalResult
    monkeypatch.setattr(multi_hop_reasoner, "_decompose", Mock(side_effect=AssertionError("Fast mode must not decompose")))
    monkeypatch.setattr(multi_hop_reasoner, "reflect_retrieval", Mock(side_effect=AssertionError("Fast mode must not reflect")))
    monkeypatch.setattr(generator, "reflect_answer", Mock(side_effect=AssertionError("Fast mode must not reflect")))
    monkeypatch.setattr(generator, "_call_llm", lambda *_: generator._AnswerOutput(answer="Answer"))
    callback = Mock(return_value=RetrievalResult(
        chunks=[Chunk(id="catalyst-source", text="A catalyst speeds up a reaction.", source_id="chemistry", modality="pdf")],
        subgraph=[], routing_mode="dense",
    ))
    result = multi_hop_reasoner.reason("What is a catalyst?", None, None, None, None, {"query_mode": "fast"}, retrieve_callback=callback)
    generated = generator.generate("Question", result, {"query_mode": "fast"})
    callback.assert_called_once_with("What is a catalyst?", "dense")
    assert "skipped" in generated.reflection.reasoning


def test_failed_generation_does_not_spend_another_call_on_reflection(monkeypatch):
    from backend.generation import generator
    from backend.schemas import Chunk, RetrievalResult
    provider = Mock(side_effect=llm.LLMError("provider unavailable"))
    monkeypatch.setattr(generator, "_call_llm", provider)
    reflection = Mock(side_effect=AssertionError("Do not reflect on an error placeholder"))
    monkeypatch.setattr(generator, "reflect_answer", reflection)
    retrieved = RetrievalResult(
        chunks=[Chunk(id="source", text="A supporting source passage.", source_id="course", modality="pdf")],
        subgraph=[], routing_mode="dense",
    )
    result = generator.generate("Question", retrieved, {})
    assert not result.reflection.is_supported
    provider.assert_called_once()
    reflection.assert_not_called()
