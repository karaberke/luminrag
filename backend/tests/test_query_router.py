"""
Tests for backend/retrieval/query_router.py

LLM calls are always mocked. Heuristic logic is tested directly with no mocking.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.retrieval.query_router import (
    _classify_heuristic,
    _classify_llm,
    _content_words,
    _parse_llm_mode,
    route_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> dict:
    return {
        "query_router": {
            "use_llm_fallback": True,
            "llm_provider": "ollama",
            "llm_ollama_model": "llama3.2:3b",
            "llm_anthropic_model": "claude-haiku-4-5-20251001",
            "llm_max_tokens": 50,
        }
    }


@pytest.fixture
def no_llm_config() -> dict:
    return {"query_router": {"use_llm_fallback": False}}


# ---------------------------------------------------------------------------
# _content_words
# ---------------------------------------------------------------------------

class TestContentWords:
    def test_removes_stopwords(self):
        words = _content_words("What is the enzyme?")
        assert "the" not in words
        assert "is" not in words

    def test_keeps_domain_words(self):
        words = _content_words("What is an enzyme?")
        assert "enzyme" in words

    def test_empty_string(self):
        assert _content_words("") == []

    def test_all_stopwords(self):
        assert _content_words("is the a an") == []


# ---------------------------------------------------------------------------
# _classify_heuristic — none mode
# ---------------------------------------------------------------------------

class TestHeuristicNone:
    def test_empty_string(self):
        assert _classify_heuristic("") == "none"

    def test_too_short(self):
        assert _classify_heuristic("Hi") == "none"

    def test_greeting_hello(self):
        assert _classify_heuristic("Hello, how are you?") == "none"

    def test_greeting_thanks(self):
        assert _classify_heuristic("Thanks for the help!") == "none"

    def test_only_stopwords(self):
        assert _classify_heuristic("is the a") == "none"


# ---------------------------------------------------------------------------
# _classify_heuristic — dense mode
# ---------------------------------------------------------------------------

class TestHeuristicDense:
    def test_what_is(self):
        assert _classify_heuristic("What is an enzyme?") == "dense"

    def test_what_are(self):
        assert _classify_heuristic("What are the products of glycolysis?") == "dense"

    def test_define(self):
        assert _classify_heuristic("Define activation energy.") == "dense"

    def test_who_invented(self):
        assert _classify_heuristic("Who invented the RAG framework?") == "dense"

    def test_when_did(self):
        assert _classify_heuristic("When did Watson and Crick discover DNA?") == "dense"

    def test_list(self):
        assert _classify_heuristic("List the stages of mitosis.") == "dense"

    def test_how_many(self):
        assert _classify_heuristic("How many ATP molecules does glycolysis produce?") == "dense"

    def test_which(self):
        assert _classify_heuristic("Which enzyme catalyses this reaction?") == "dense"

    def test_name(self):
        assert _classify_heuristic("Name the four DNA bases.") == "dense"


# ---------------------------------------------------------------------------
# _classify_heuristic — graph mode
# ---------------------------------------------------------------------------

class TestHeuristicGraph:
    def test_why_prefix(self):
        assert _classify_heuristic("Why does temperature affect enzyme activity?") == "graph"

    def test_how_does_prefix(self):
        assert _classify_heuristic("How does pH affect the rate of enzyme catalysis?") == "graph"

    def test_compare_prefix(self):
        assert _classify_heuristic("Compare RAG and GraphRAG retrieval methods.") == "graph"

    def test_contrast_prefix(self):
        assert _classify_heuristic("Contrast dense retrieval with graph retrieval.") == "graph"

    def test_explain_prefix(self):
        assert _classify_heuristic("Explain the role of cofactors in enzyme function.") == "graph"

    def test_difference_between_signal(self):
        assert _classify_heuristic(
            "What is the difference between competitive and non-competitive inhibition?"
        ) == "graph"

    def test_relationship_between_signal(self):
        assert _classify_heuristic(
            "Describe the relationship between substrate concentration and reaction rate."
        ) == "graph"

    def test_affect_signal(self):
        assert _classify_heuristic(
            "How does temperature affect enzyme kinetics?"
        ) == "graph"

    def test_leads_to_signal(self):
        assert _classify_heuristic(
            "Enzyme inhibition leads to decreased metabolic output."
        ) == "graph"

    def test_in_terms_of_signal(self):
        assert _classify_heuristic(
            "Explain hallucination in terms of retrieval quality."
        ) == "graph"

    def test_graph_beats_dense_prefix(self):
        # "why" should override any dense-looking structure
        assert _classify_heuristic("Why is RAG better than pure LLM generation?") == "graph"


# ---------------------------------------------------------------------------
# _classify_heuristic — ambiguous (returns None)
# ---------------------------------------------------------------------------

class TestHeuristicAmbiguous:
    def test_returns_none_for_ambiguous(self):
        # Doesn't start with any known prefix and has no strong signal
        result = _classify_heuristic("Tell me something about enzymes and metabolism.")
        assert result is None

    def test_returns_none_for_general_statement(self):
        result = _classify_heuristic("Enzymes are proteins that catalyse reactions.")
        assert result is None


# ---------------------------------------------------------------------------
# _parse_llm_mode
# ---------------------------------------------------------------------------

class TestParseLlmMode:
    def test_parses_dense(self):
        assert _parse_llm_mode("dense") == "dense"

    def test_parses_graph(self):
        assert _parse_llm_mode("graph") == "graph"

    def test_parses_none(self):
        assert _parse_llm_mode("none") == "none"

    def test_strips_whitespace(self):
        assert _parse_llm_mode("  graph  ") == "graph"

    def test_case_insensitive(self):
        assert _parse_llm_mode("DENSE") == "dense"

    def test_strips_punctuation(self):
        assert _parse_llm_mode('"graph"') == "graph"

    def test_finds_mode_in_sentence(self):
        assert _parse_llm_mode("The answer is graph because multiple hops needed.") == "graph"

    def test_unrecognisable_falls_back_to_dense(self):
        assert _parse_llm_mode("I cannot classify this.") == "dense"

    def test_empty_falls_back_to_dense(self):
        assert _parse_llm_mode("") == "dense"


# ---------------------------------------------------------------------------
# _classify_llm
# ---------------------------------------------------------------------------

class TestClassifyLlm:
    def test_ollama_path_called(self, base_config):
        cfg = base_config["query_router"]
        with patch("backend.retrieval.query_router._classify_llm_ollama",
                   return_value="graph") as mock:
            result = _classify_llm("some question", cfg)
            mock.assert_called_once()
            assert result == "graph"

    def test_anthropic_path_called(self, base_config):
        cfg = {**base_config["query_router"], "llm_provider": "anthropic"}
        with patch("backend.retrieval.query_router._classify_llm_anthropic",
                   return_value="dense") as mock:
            result = _classify_llm("some question", cfg)
            mock.assert_called_once()
            assert result == "dense"

    def test_llm_failure_returns_dense(self, base_config):
        cfg = base_config["query_router"]
        with patch("backend.retrieval.query_router._classify_llm_ollama",
                   side_effect=RuntimeError("Ollama down")):
            assert _classify_llm("some question", cfg) == "dense"

    def test_unknown_provider_returns_dense(self, base_config):
        cfg = {**base_config["query_router"], "llm_provider": "openai"}
        assert _classify_llm("some question", cfg) == "dense"


# ---------------------------------------------------------------------------
# route_query (full integration)
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def test_empty_question_returns_none(self, base_config):
        assert route_query("", base_config) == "none"

    def test_whitespace_only_returns_none(self, base_config):
        assert route_query("   ", base_config) == "none"

    def test_dense_question_no_llm_call(self, base_config):
        with patch("backend.retrieval.query_router._classify_llm") as mock:
            result = route_query("What is an enzyme?", base_config)
            mock.assert_not_called()
            assert result == "dense"

    def test_graph_question_no_llm_call(self, base_config):
        with patch("backend.retrieval.query_router._classify_llm") as mock:
            result = route_query("Why does pH affect enzyme activity?", base_config)
            mock.assert_not_called()
            assert result == "graph"

    def test_ambiguous_calls_llm_when_enabled(self, base_config):
        with patch("backend.retrieval.query_router._classify_llm",
                   return_value="graph") as mock:
            route_query("Tell me about enzymes and metabolism.", base_config)
            mock.assert_called_once()

    def test_ambiguous_no_llm_call_when_disabled(self, no_llm_config):
        with patch("backend.retrieval.query_router._classify_llm") as mock:
            result = route_query("Tell me about enzymes and metabolism.", no_llm_config)
            mock.assert_not_called()
            assert result == "dense"  # conservative default

    def test_llm_result_returned(self, base_config):
        with patch("backend.retrieval.query_router._classify_llm",
                   return_value="graph"):
            result = route_query("Tell me about enzymes and metabolism.", base_config)
            assert result == "graph"

    def test_routing_mode_is_valid_literal(self, base_config):
        valid = {"dense", "graph", "none"}
        questions = [
            "What is an enzyme?",
            "Why does temperature affect reaction rate?",
            "Hi there!",
        ]
        for q in questions:
            assert route_query(q, base_config) in valid

    # Real-world questions from the raw course materials
    def test_rag_question_routes_dense(self, base_config):
        assert route_query("What is RAG?", base_config) == "dense"

    def test_rag_comparison_routes_graph(self, base_config):
        assert route_query(
            "Compare RAG and GraphRAG and explain the difference between them.",
            base_config
        ) == "graph"

    def test_chemistry_why_routes_graph(self, base_config):
        assert route_query(
            "Why does increasing substrate concentration affect the rate of an enzyme-catalysed reaction?",
            base_config
        ) == "graph"

    def test_chemistry_define_routes_dense(self, base_config):
        assert route_query("Define the Michaelis constant Km.", base_config) == "dense"
