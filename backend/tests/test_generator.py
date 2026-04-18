"""
Tests for backend/generation/generator.py and backend/generation/prompts.py
"""

import pytest

from backend.schemas import (
    Chunk,
    GraphTriple,
    GenerationResult,
    ReflectionVerdict,
    RetrievalResult,
)
from backend.generation.generator import (
    _format_chunks,
    _format_relationships,
    _extract_cited_chunk_ids,
    _build_prompt,
    generate,
)
from backend.generation.prompts import (
    DENSE_RAG_PROMPT,
    GRAPH_RAG_PROMPT,
    NO_RETRIEVAL_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, chunk_id: str | None = None, modality: str = "pdf") -> Chunk:
    c = Chunk(text=text, source_id="src1", modality=modality, metadata={})
    if chunk_id:
        object.__setattr__(c, "id", chunk_id)
    return c


def _make_triple(head: str, relation: str, tail: str) -> GraphTriple:
    return GraphTriple(head=head, relation=relation, tail=tail, source_chunk_ids=[])


def _make_result(
    texts: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    triples: list[GraphTriple] | None = None,
    routing_mode: str = "dense",
) -> RetrievalResult:
    texts = texts or []
    chunk_ids = chunk_ids or [None] * len(texts)
    chunks = [_make_chunk(t, cid) for t, cid in zip(texts, chunk_ids)]
    return RetrievalResult(chunks=chunks, subgraph=triples or [], routing_mode=routing_mode)


def _verdict(supported: bool = True, useful: bool = True) -> ReflectionVerdict:
    return ReflectionVerdict(
        needs_retrieval=True,
        is_relevant=True,
        is_supported=supported,
        is_useful=useful,
        reasoning="test verdict",
    )


def _ollama_cfg() -> dict:
    return {
        "generator": {
            "provider": "ollama",
            "ollama_model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "max_tokens": 1024,
            "max_context_chars": 4000,
        },
        "self_rag": {
            "provider": "ollama",
            "ollama_model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "max_tokens": 256,
            "max_context_chars": 2000,
        },
    }


def _anthropic_cfg() -> dict:
    return {
        "generator": {
            "provider": "anthropic",
            "anthropic_model": "claude-haiku-4-5-20251001",
            "max_tokens": 1024,
            "max_context_chars": 4000,
        },
        "self_rag": {
            "provider": "anthropic",
            "anthropic_model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "max_context_chars": 2000,
        },
    }


# ---------------------------------------------------------------------------
# _format_chunks
# ---------------------------------------------------------------------------

class TestFormatChunks:
    def test_no_chunks_returns_placeholder(self):
        result = _make_result([])
        out = _format_chunks(result, 4000)
        assert out == "(no context retrieved)"

    def test_single_chunk_numbered(self):
        result = _make_result(["Photosynthesis converts light to energy."])
        out = _format_chunks(result, 4000)
        assert "[1]" in out
        assert "Photosynthesis" in out

    def test_multiple_chunks_numbered(self):
        result = _make_result(["First.", "Second.", "Third."])
        out = _format_chunks(result, 4000)
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out

    def test_truncation_at_max_chars(self):
        long_text = "A" * 3990
        result = _make_result([long_text, "Second chunk."])
        out = _format_chunks(result, 4000)
        assert "truncated" in out
        assert "Second chunk" not in out

    def test_modality_included(self):
        result = _make_result(["Slide text."])
        result.chunks[0] = _make_chunk("Slide text.", modality="slide")
        out = _format_chunks(result, 4000)
        assert "(slide)" in out


# ---------------------------------------------------------------------------
# _format_relationships
# ---------------------------------------------------------------------------

class TestFormatRelationships:
    def test_no_triples_returns_placeholder(self):
        result = _make_result([])
        out = _format_relationships(result)
        assert out == "(no relationships extracted)"

    def test_triple_formatted_correctly(self):
        result = _make_result(triples=[_make_triple("ATP", "CAUSES", "Energy")])
        out = _format_relationships(result)
        assert "ATP" in out
        assert "CAUSES" in out
        assert "Energy" in out

    def test_multiple_triples(self):
        triples = [
            _make_triple("ATP", "PART_OF", "Cell"),
            _make_triple("Glucose", "CAUSES", "ATP"),
        ]
        result = _make_result(triples=triples)
        out = _format_relationships(result)
        assert "ATP" in out
        assert "Glucose" in out


# ---------------------------------------------------------------------------
# _extract_cited_chunk_ids
# ---------------------------------------------------------------------------

class TestExtractCitedChunkIds:
    def test_single_citation(self):
        result = _make_result(["chunk A"], ["id-A"])
        ids = _extract_cited_chunk_ids("According to [1], ATP is energy.", result)
        assert ids == ["id-A"]

    def test_multiple_citations(self):
        result = _make_result(["A", "B", "C"], ["id-A", "id-B", "id-C"])
        ids = _extract_cited_chunk_ids("[1] and [3] support this.", result)
        assert "id-A" in ids
        assert "id-C" in ids
        assert "id-B" not in ids

    def test_duplicate_citations_deduplicated(self):
        result = _make_result(["A"], ["id-A"])
        ids = _extract_cited_chunk_ids("[1] is cited again in [1].", result)
        assert ids.count("id-A") == 1

    def test_out_of_range_citation_ignored(self):
        result = _make_result(["A"], ["id-A"])
        ids = _extract_cited_chunk_ids("See [99] for details.", result)
        assert ids == []

    def test_no_citations_returns_empty(self):
        result = _make_result(["A"], ["id-A"])
        ids = _extract_cited_chunk_ids("No citations here.", result)
        assert ids == []

    def test_empty_chunks_returns_empty(self):
        result = _make_result([])
        ids = _extract_cited_chunk_ids("[1] cited but no chunks.", result)
        assert ids == []


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_none_mode_uses_no_retrieval_prompt(self):
        result = _make_result(routing_mode="none")
        prompt = _build_prompt("What is RAG?", result, 4000)
        assert "general knowledge" in prompt.lower() or "What is RAG?" in prompt

    def test_empty_chunks_uses_no_retrieval_prompt(self):
        result = _make_result([], routing_mode="dense")
        prompt = _build_prompt("What is RAG?", result, 4000)
        assert "general knowledge" in prompt.lower() or "What is RAG?" in prompt

    def test_dense_mode_uses_dense_prompt(self):
        result = _make_result(["Some context."], routing_mode="dense")
        prompt = _build_prompt("What is RAG?", result, 4000)
        assert "What is RAG?" in prompt
        assert "Some context." in prompt
        # Dense prompt should not contain relationship instructions
        assert "relationships" not in prompt.lower() or "Concept relationships" not in prompt

    def test_graph_mode_uses_graph_prompt(self):
        triples = [_make_triple("RAG", "PART_OF", "LLM")]
        result = _make_result(["Some context."], triples=triples, routing_mode="graph")
        prompt = _build_prompt("What is RAG?", result, 4000)
        assert "What is RAG?" in prompt
        assert "RAG" in prompt
        assert "PART_OF" in prompt

    def test_question_always_in_prompt(self):
        for mode in ["dense", "graph", "none"]:
            result = _make_result(["ctx"] if mode != "none" else [], routing_mode=mode)
            prompt = _build_prompt("My question?", result, 4000)
            assert "My question?" in prompt


# ---------------------------------------------------------------------------
# generate (integration with mocks)
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_generation_result(self, mocker):
        mocker.patch(
            "backend.generation.generator._call_llm",
            return_value="ATP is the energy currency of the cell.",
        )
        mocker.patch(
            "backend.generation.generator.reflect_answer",
            return_value=_verdict(),
        )
        result = _make_result(["ATP context."], ["id-1"])
        out = generate("What is ATP?", result, _ollama_cfg())
        assert isinstance(out, GenerationResult)
        assert out.answer == "ATP is the energy currency of the cell."

    def test_routing_mode_preserved(self, mocker):
        mocker.patch(
            "backend.generation.generator._call_llm",
            return_value="Answer.",
        )
        mocker.patch(
            "backend.generation.generator.reflect_answer",
            return_value=_verdict(),
        )
        result = _make_result(["ctx"], routing_mode="graph")
        out = generate("Question?", result, _ollama_cfg())
        assert out.routing_mode == "graph"

    def test_reflection_attached(self, mocker):
        mocker.patch("backend.generation.generator._call_llm", return_value="Answer.")
        v = _verdict(supported=False)
        mocker.patch("backend.generation.generator.reflect_answer", return_value=v)
        result = _make_result(["ctx"], ["id-1"])
        out = generate("Q?", result, _ollama_cfg())
        assert out.reflection.is_supported is False

    def test_citations_extracted(self, mocker):
        mocker.patch(
            "backend.generation.generator._call_llm",
            return_value="See [1] and [2] for details.",
        )
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["A", "B"], ["id-A", "id-B"])
        out = generate("Q?", result, _ollama_cfg())
        assert "id-A" in out.evidence_chunk_ids
        assert "id-B" in out.evidence_chunk_ids

    def test_llm_failure_returns_fallback_answer(self, mocker):
        mocker.patch(
            "backend.generation.generator._call_llm",
            side_effect=RuntimeError("connection refused"),
        )
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"], ["id-1"])
        out = generate("Q?", result, _ollama_cfg())
        assert "unable to generate" in out.answer.lower()

    def test_ollama_provider_called(self, mocker):
        mock_ollama = mocker.patch(
            "backend.generation.generator._call_ollama",
            return_value="Answer.",
        )
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"])
        generate("Q?", result, _ollama_cfg())
        mock_ollama.assert_called_once()

    def test_anthropic_provider_called(self, mocker):
        mock_anthropic = mocker.patch(
            "backend.generation.generator._call_anthropic",
            return_value="Answer.",
        )
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"])
        generate("Q?", result, _anthropic_cfg())
        mock_anthropic.assert_called_once()

    def test_no_retrieval_mode_no_context_in_prompt(self, mocker):
        captured = {}

        def capture(prompt, cfg):
            captured["prompt"] = prompt
            return "Answer."

        mocker.patch("backend.generation.generator._call_llm", side_effect=capture)
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result([], routing_mode="none")
        generate("What is RAG?", result, _ollama_cfg())
        assert "Retrieved context" not in captured["prompt"]

    def test_graph_mode_includes_relationships_in_prompt(self, mocker):
        captured = {}

        def capture(prompt, cfg):
            captured["prompt"] = prompt
            return "Answer."

        mocker.patch("backend.generation.generator._call_llm", side_effect=capture)
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        triples = [_make_triple("ATP", "CAUSES", "Energy")]
        result = _make_result(["ctx"], triples=triples, routing_mode="graph")
        generate("Q?", result, _ollama_cfg())
        assert "CAUSES" in captured["prompt"]

    def test_evidence_chunk_ids_empty_when_no_citations(self, mocker):
        mocker.patch("backend.generation.generator._call_llm", return_value="No citations here.")
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"], ["id-1"])
        out = generate("Q?", result, _ollama_cfg())
        assert out.evidence_chunk_ids == []
