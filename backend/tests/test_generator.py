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
    _AnswerCitation,
    _AnswerOutput,
    _context_chunks,
    _format_chunks,
    _format_relationships,
    _extract_cited_chunk_ids,
    _build_prompt,
    generate,
)
from backend.utils.text import find_quote_span
from backend.generation.prompts import (
    DENSE_RAG_PROMPT,
    GRAPH_RAG_PROMPT,
    NO_EVIDENCE_ANSWER,
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
        assert out == "(no connections retrieved)"

    def test_triple_formatted_correctly(self):
        result = _make_result(triples=[_make_triple("ATP", "CAUSES", "Energy")])
        out = _format_relationships(result)
        assert "ATP" in out
        assert "CAUSES" not in out and "→" not in out
        assert "(association)" in out
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

    def test_duplicate_and_reversed_semantic_connections_are_one_plain_line(self):
        result = _make_result(triples=[
            _make_triple("ATP", "RELATED_TO", "Energy"),
            _make_triple("Energy", "RELATED_TO", "ATP"),
            _make_triple("Energy", "CAUSES", "ATP"),
        ])
        out = _format_relationships(result)
        assert out == "- ATP — Energy (association)"

    def test_hierarchy_keeps_parent_child_organization(self):
        result = _make_result(triples=[
            _make_triple("Biology", "HAS_SUBTOPIC", "Cells"),
            _make_triple("Cells", "HAS_CONTENT", "ATP"),
        ])
        assert _format_relationships(result) == "- Biology → Cells (hierarchy)\n- Cells → ATP (hierarchy)"


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
        assert "general knowledge" in prompt.lower()

    @pytest.mark.parametrize("mode", ["dense", "graph", "hybrid"])
    def test_empty_sources_keep_the_requested_grounding_prompt(self, mode):
        result = _make_result([], routing_mode=mode)
        prompt = _build_prompt("What is RAG?", result, 4000)
        assert "general knowledge" not in prompt.lower()
        assert "<retrieved_context>" in prompt and "no context retrieved" in prompt

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
        assert "PART_OF" not in prompt
        assert "(association)" in prompt

    @pytest.mark.parametrize("mode", ["graph", "hybrid"])
    def test_associations_never_replace_passage_evidence(self, mode):
        result = _make_result(["Direct passage evidence."],
                              triples=[_make_triple("A", "RELATED_TO", "B")], routing_mode=mode)
        prompt = _build_prompt("Explain A and B.", result, 4000)
        assert "ONLY the retrieved context passages" in prompt
        assert "does not establish causation" in prompt
        assert "retrieved passages support that explanation" in prompt
        assert "[1] (pdf) Direct passage evidence." in prompt

    def test_question_always_in_prompt(self):
        for mode in ["dense", "graph", "none"]:
            result = _make_result(["ctx"] if mode != "none" else [], routing_mode=mode)
            prompt = _build_prompt("My question?", result, 4000)
            assert "My question?" in prompt


# ---------------------------------------------------------------------------
# generate (integration with mocks)
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.mark.parametrize("mode", ["dense", "graph", "hybrid"])
    @pytest.mark.parametrize("query_mode", ["fast", "thorough"])
    def test_no_source_evidence_declines_without_generation_or_reflection(self, mocker, mode, query_mode):
        provider = mocker.patch("backend.generation.generator._call_llm")
        reflector = mocker.patch("backend.generation.generator.reflect_answer")
        result = _make_result([], triples=[_make_triple("A", "RELATED_TO", "B")], routing_mode=mode)
        output = generate("What does the private handbook say?", result, {"query_mode": query_mode})
        assert output.answer == NO_EVIDENCE_ANSWER
        assert output.routing_mode == mode
        assert output.evidence_chunk_ids == output.citations == []
        assert output.reflection.needs_retrieval and not output.reflection.is_supported
        assert not output.reflection.is_relevant and not output.reflection.is_useful
        provider.assert_not_called()
        reflector.assert_not_called()

    def test_returns_generation_result(self, mocker):
        mocker.patch(
            "backend.generation.generator._call_llm",
            return_value=_AnswerOutput(answer="ATP is the energy currency of the cell."),
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
            return_value=_AnswerOutput(answer="Answer."),
        )
        mocker.patch(
            "backend.generation.generator.reflect_answer",
            return_value=_verdict(),
        )
        result = _make_result(["ctx"], routing_mode="graph")
        out = generate("Question?", result, _ollama_cfg())
        assert out.routing_mode == "graph"

    def test_reflection_attached(self, mocker):
        mocker.patch("backend.generation.generator._call_llm", return_value=_AnswerOutput(answer="Answer."))
        v = _verdict(supported=False)
        mocker.patch("backend.generation.generator.reflect_answer", return_value=v)
        result = _make_result(["ctx"], ["id-1"])
        out = generate("Q?", result, _ollama_cfg())
        assert out.reflection.is_supported is False

    def test_citations_extracted(self, mocker):
        mocker.patch(
            "backend.generation.generator._call_llm",
            return_value=_AnswerOutput(answer="See [1] and [2] for details."),
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
            return_value=_AnswerOutput(answer="Answer."),
        )
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"])
        generate("Q?", result, _ollama_cfg())
        mock_ollama.assert_called_once()

    def test_anthropic_provider_called(self, mocker):
        mock_anthropic = mocker.patch(
            "backend.generation.generator._call_anthropic",
            return_value=_AnswerOutput(answer="Answer."),
        )
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"])
        generate("Q?", result, _anthropic_cfg())
        mock_anthropic.assert_called_once()

    def test_no_retrieval_mode_no_context_in_prompt(self, mocker):
        captured = {}

        def capture(prompt, system, cfg):
            captured["prompt"] = prompt
            return _AnswerOutput(answer="Answer.")

        mocker.patch("backend.generation.generator._call_llm", side_effect=capture)
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result([], routing_mode="none")
        generate("What is RAG?", result, _ollama_cfg())
        assert "Retrieved context" not in captured["prompt"]

    def test_graph_mode_includes_relationships_in_prompt(self, mocker):
        captured = {}

        def capture(prompt, system, cfg):
            captured["prompt"] = prompt
            return _AnswerOutput(answer="Answer.")

        mocker.patch("backend.generation.generator._call_llm", side_effect=capture)
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        triples = [_make_triple("ATP", "CAUSES", "Energy")]
        result = _make_result(["ctx"], triples=triples, routing_mode="graph")
        generate("Q?", result, _ollama_cfg())
        assert "CAUSES" not in captured["prompt"]
        assert "ATP — Energy (association)" in captured["prompt"]

    def test_evidence_chunk_ids_empty_when_no_citations(self, mocker):
        mocker.patch("backend.generation.generator._call_llm", return_value=_AnswerOutput(answer="No citations here."))
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        result = _make_result(["ctx"], ["id-1"])
        out = generate("Q?", result, _ollama_cfg())
        assert out.evidence_chunk_ids == []


@pytest.mark.parametrize("text,quote,expected", [
    ("First. Enzymes lower energy. Last.", "Enzymes lower energy.", "Enzymes lower energy."),
    ("αβ. Enzymes\n  lower\tenergy. Ω", "Enzymes lower energy.", "Enzymes\n  lower\tenergy."),
    ("**Enzymes** lower energy.", "Enzymes lower energy.", "**Enzymes** lower energy."),
    ("Enzymes lower energy.", "**Enzymes lower energy.**", "Enzymes lower energy."),
    ("Enzymes do not lower energy.", "Enzymes lower energy.", None),
    ("A fact. A fact.", "A fact.", None),
    ("A fact.\nA   fact.", "A\tfact.", None),
    ("Exact punctuation!", "Exact punctuation.", None),
    ("Identifier x_2 is important.", "Identifier x2 is important.", None),
    ("Passage.", " ", None),
])
def test_quote_span_preserves_real_source_text_and_rejects_ambiguity(text, quote, expected):
    span = find_quote_span(text, quote)
    if expected is None:
        assert span is None
    else:
        assert span is not None
        assert text[span[0]:span[1]] == expected


@pytest.mark.parametrize("marker", ["*", "`", "__"])
def test_quote_span_handles_maximum_malformed_markup_without_backtracking(marker):
    text = (marker + "a ") * (100_000 // (len(marker) + 2))
    assert find_quote_span(text, "A nonexistent exact sentence.") is None


def test_generation_attaches_only_verified_answer_used_quotes(mocker):
    retrieved = _make_result([
        "Intro. Enzymes\n  lower activation energy. More.",
        "The French Revolution began in 1789.",
        "An uncited statement.",
    ], ["bio", "history", "unused"])
    provider = mocker.patch("backend.generation.generator._call_llm", return_value=_AnswerOutput(
        answer="Enzymes lower activation energy [1]; the revolution began in 1789 [2].",
        citations=[
            _AnswerCitation(source_number=1, quote="Enzymes lower activation energy."),
            _AnswerCitation(source_number=2, quote="Enzymes lower activation energy."),
            _AnswerCitation(source_number=3, quote="An uncited statement."),
            _AnswerCitation(source_number=99, quote="Invented source."),
        ],
    ))
    output = generate("Explain", retrieved, {"query_mode": "fast"})
    provider.assert_called_once()
    assert output.evidence_chunk_ids == ["bio", "history"]
    assert len(output.citations) == 1
    citation = output.citations[0]
    assert citation.chunk_id == "bio" and citation.source_number == 1
    assert citation.quote == "Enzymes\n  lower activation energy."
    assert retrieved.chunks[0].text[citation.quote_start:citation.quote_end] == citation.quote
    assert output.answer.endswith("1789 [2].")


def test_generation_ignores_quotes_for_context_not_sent_to_model(mocker):
    retrieved = _make_result(["Visible.", "Hidden." * 100], ["visible", "hidden"])
    mocker.patch("backend.generation.generator._call_llm", return_value=_AnswerOutput(
        answer="Visible [1], hidden [2].",
        citations=[_AnswerCitation(source_number=1, quote="Visible."), _AnswerCitation(source_number=2, quote="Hidden.")],
    ))
    output = generate("Explain", retrieved, {"query_mode": "fast", "generator": {"max_context_chars": 60}})
    assert output.evidence_chunk_ids == ["visible"]
    assert [citation.chunk_id for citation in output.citations] == ["visible"]


def test_generation_invalid_quotes_keep_answer_and_whole_passage_fallback(mocker):
    provider = mocker.patch("backend.generation.generator._call_llm", return_value=_AnswerOutput(
        answer="A grounded answer [1].", citations=[_AnswerCitation(source_number=1, quote="Hallucinated exact quote.")],
    ))
    output = generate("Explain", _make_result(["Actual passage."], ["original"]), {"query_mode": "fast"})
    assert output.answer == "A grounded answer [1]."
    assert output.evidence_chunk_ids == ["original"]
    assert output.citations == []
    provider.assert_called_once()


def test_long_first_passage_supplies_prefix_and_quote_keeps_original_offsets(mocker):
    quote = "Enzymes lower activation energy."
    hidden_quote = "Only hidden in the omitted tail."
    text = ("αβ source intro. " + quote + " " + "Long material. " * 10_000)[:100_000 - len(hidden_quote)] + hidden_quote
    retrieved = _make_result([text, "Later omitted source."], ["long-note", "later"])
    retrieved.chunks[0].modality = "note"

    def provider(prompt, _system, _cfg):
        assert f"[1] (note) αβ source intro. {quote}" in prompt
        assert "truncated" in prompt
        assert text not in prompt and "Later omitted source." not in prompt
        return _AnswerOutput(answer="Enzymes lower activation energy [1], omitted source [2].", citations=[
            _AnswerCitation(source_number=1, quote=hidden_quote),
            _AnswerCitation(source_number=1, quote=quote),
            _AnswerCitation(source_number=2, quote="Later omitted source."),
        ])

    call = mocker.patch("backend.generation.generator._call_llm", side_effect=provider)
    output = generate("Explain", retrieved, {"query_mode": "fast", "generator": {"max_context_chars": 256}})
    call.assert_called_once()
    assert len(output.citations) == 1
    assert output.evidence_chunk_ids == ["long-note"]
    citation = output.citations[0]
    assert citation.chunk_id == "long-note"
    assert citation.quote_start == text.index(quote)
    assert text[citation.quote_start:citation.quote_end] == quote
    assert retrieved.chunks[0].text == text
    assert len(_format_chunks(retrieved, 256)) <= 256
    assert "truncated" in _format_chunks(_make_result([text]), 256)


@pytest.mark.parametrize("budget", [0, 1, 5, 9, 10, 38, 40, 49, 50, 51, 60])
def test_tiny_context_budgets_never_expose_unshown_quote_text(budget, mocker):
    retrieved = _make_result(["A" * 100_000, "Later."], ["long", "later"])
    context = _format_chunks(retrieved, budget)
    assert len(context) <= budget
    visible = _context_chunks(retrieved, budget)
    for index, chunk in enumerate(visible, start=1):
        assert chunk.text and f"[{index}] ({chunk.modality}) {chunk.text}" in context
        assert retrieved.chunks[index - 1].text.startswith(chunk.text)
    assert "Later." not in context
    if not visible:
        provider = mocker.patch("backend.generation.generator._call_llm")
        reflector = mocker.patch("backend.generation.generator.reflect_answer")
        output = generate("Explain", retrieved, {"generator": {"max_context_chars": budget}})
        assert output.answer == NO_EVIDENCE_ANSWER and output.evidence_chunk_ids == []
        provider.assert_not_called()
        reflector.assert_not_called()
