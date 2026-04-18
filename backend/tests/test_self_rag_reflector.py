"""
Tests for backend/self_rag/self_rag_reflector.py
"""

import json
import pytest

from backend.schemas import Chunk, RetrievalResult, ReflectionVerdict
from backend.self_rag.self_rag_reflector import (
    reflect_retrieval,
    reflect_answer,
    _format_context,
    _parse_verdict,
    _SAFE_DEFAULT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(text: str, modality: str = "pdf") -> Chunk:
    return Chunk(text=text, source_id="src1", modality=modality, metadata={})


def _make_result(texts: list[str] = None) -> RetrievalResult:
    chunks = [_make_chunk(t) for t in (texts or [])]
    return RetrievalResult(chunks=chunks, subgraph=[], routing_mode="dense")


def _ollama_cfg(model: str = "llama3.2:3b") -> dict:
    return {
        "self_rag": {
            "provider": "ollama",
            "ollama_model": model,
            "ollama_base_url": "http://localhost:11434",
            "max_tokens": 256,
            "max_context_chars": 2000,
        }
    }


def _anthropic_cfg() -> dict:
    return {
        "self_rag": {
            "provider": "anthropic",
            "anthropic_model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "max_context_chars": 2000,
        }
    }


_VALID_VERDICT_JSON = json.dumps({
    "needs_retrieval": True,
    "is_relevant": True,
    "is_supported": False,
    "is_useful": True,
    "reasoning": "Context supports the answer.",
})


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_no_chunks_returns_placeholder(self):
        result = _make_result([])
        assert _format_context(result, 2000) == "(no context retrieved)"

    def test_single_chunk_formatted(self):
        result = _make_result(["Photosynthesis converts light into energy."])
        out = _format_context(result, 2000)
        assert "[1]" in out
        assert "Photosynthesis" in out

    def test_truncation_at_max_chars(self):
        # "[1] (pdf) " = 10 chars, so 1990 'A's fills exactly to 2000 — second chunk overflows
        long_text = "A" * 1990
        result = _make_result([long_text, "Second chunk that should be truncated."])
        out = _format_context(result, 2000)
        assert "truncated" in out
        assert "Second chunk" not in out

    def test_multiple_chunks_numbered(self):
        result = _make_result(["First.", "Second.", "Third."])
        out = _format_context(result, 2000)
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_valid_json_parsed(self):
        verdict = _parse_verdict(_VALID_VERDICT_JSON)
        assert verdict.needs_retrieval is True
        assert verdict.is_relevant is True
        assert verdict.is_supported is False
        assert verdict.is_useful is True
        assert verdict.reasoning == "Context supports the answer."

    def test_markdown_fences_stripped(self):
        fenced = f"```json\n{_VALID_VERDICT_JSON}\n```"
        verdict = _parse_verdict(fenced)
        assert verdict.is_supported is False

    def test_plain_code_fence_stripped(self):
        fenced = f"```\n{_VALID_VERDICT_JSON}\n```"
        verdict = _parse_verdict(fenced)
        assert verdict.needs_retrieval is True

    def test_malformed_json_raises(self):
        with pytest.raises(Exception):
            _parse_verdict("not valid json")

    def test_missing_fields_use_defaults(self):
        minimal = json.dumps({"reasoning": "ok"})
        verdict = _parse_verdict(minimal)
        # All bool fields default to True when missing
        assert verdict.needs_retrieval is True
        assert verdict.is_relevant is True
        assert verdict.is_supported is True
        assert verdict.is_useful is True


# ---------------------------------------------------------------------------
# reflect_retrieval
# ---------------------------------------------------------------------------

class TestReflectRetrieval:
    def test_returns_safe_default_on_llm_failure(self, mocker):
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            side_effect=RuntimeError("network error"),
        )
        result = _make_result(["Some context."])
        verdict = reflect_retrieval("What is photosynthesis?", result, _ollama_cfg())
        assert verdict == _SAFE_DEFAULT

    def test_valid_verdict_returned(self, mocker):
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            return_value=_VALID_VERDICT_JSON,
        )
        result = _make_result(["Some context."])
        verdict = reflect_retrieval("What is photosynthesis?", result, _ollama_cfg())
        assert isinstance(verdict, ReflectionVerdict)
        assert verdict.needs_retrieval is True
        assert verdict.reasoning == "Context supports the answer."

    def test_placeholders_for_is_supported_is_useful(self, mocker):
        """reflect_retrieval prompt forces is_supported=true, is_useful=true."""
        payload = json.dumps({
            "needs_retrieval": True,
            "is_relevant": False,
            "is_supported": True,
            "is_useful": True,
            "reasoning": "Not relevant.",
        })
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            return_value=payload,
        )
        result = _make_result(["Unrelated text about cooking."])
        verdict = reflect_retrieval("Explain entropy.", result, _ollama_cfg())
        assert verdict.is_relevant is False
        assert verdict.is_supported is True
        assert verdict.is_useful is True

    def test_empty_retrieval_result_handled(self, mocker):
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            return_value=_VALID_VERDICT_JSON,
        )
        result = _make_result([])
        verdict = reflect_retrieval("What is entropy?", result, _ollama_cfg())
        assert isinstance(verdict, ReflectionVerdict)

    def test_ollama_provider_called(self, mocker):
        mock_ollama = mocker.patch(
            "backend.self_rag.self_rag_reflector._call_ollama",
            return_value=_VALID_VERDICT_JSON,
        )
        result = _make_result(["Context text."])
        reflect_retrieval("Question?", result, _ollama_cfg())
        mock_ollama.assert_called_once()

    def test_anthropic_provider_called(self, mocker):
        mock_anthropic = mocker.patch(
            "backend.self_rag.self_rag_reflector._call_anthropic",
            return_value=_VALID_VERDICT_JSON,
        )
        result = _make_result(["Context text."])
        reflect_retrieval("Question?", result, _anthropic_cfg())
        mock_anthropic.assert_called_once()


# ---------------------------------------------------------------------------
# reflect_answer
# ---------------------------------------------------------------------------

class TestReflectAnswer:
    def test_returns_safe_default_on_llm_failure(self, mocker):
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            side_effect=ConnectionError("timeout"),
        )
        result = _make_result(["Some context."])
        verdict = reflect_answer("What is ATP?", result, "ATP is energy.", _ollama_cfg())
        assert verdict == _SAFE_DEFAULT

    def test_all_four_fields_populated(self, mocker):
        payload = json.dumps({
            "needs_retrieval": True,
            "is_relevant": True,
            "is_supported": False,
            "is_useful": True,
            "reasoning": "Answer contains unsupported claims.",
        })
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            return_value=payload,
        )
        result = _make_result(["ATP is adenosine triphosphate."])
        verdict = reflect_answer("What is ATP?", result, "ATP flies through space.", _ollama_cfg())
        assert verdict.needs_retrieval is True
        assert verdict.is_relevant is True
        assert verdict.is_supported is False
        assert verdict.is_useful is True
        assert "unsupported" in verdict.reasoning

    def test_empty_retrieval_result_handled(self, mocker):
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            return_value=_VALID_VERDICT_JSON,
        )
        result = _make_result([])
        verdict = reflect_answer("Question?", result, "Answer.", _ollama_cfg())
        assert isinstance(verdict, ReflectionVerdict)

    def test_context_truncated_at_max_chars(self, mocker):
        captured = {}

        def capture_call(prompt, cfg):
            captured["prompt"] = prompt
            return _VALID_VERDICT_JSON

        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            side_effect=capture_call,
        )
        cfg = _ollama_cfg()
        cfg["self_rag"]["max_context_chars"] = 50

        long_text = "X" * 200
        result = _make_result([long_text])
        reflect_answer("Q?", result, "A.", cfg)
        assert "truncated" in captured["prompt"]

    def test_fenced_response_parsed(self, mocker):
        fenced = f"```json\n{_VALID_VERDICT_JSON}\n```"
        mocker.patch(
            "backend.self_rag.self_rag_reflector._call_llm",
            return_value=fenced,
        )
        result = _make_result(["Some context."])
        verdict = reflect_answer("Q?", result, "A.", _ollama_cfg())
        assert isinstance(verdict, ReflectionVerdict)
