"""Routing decisions and provider fallback without remote model calls."""
from unittest.mock import patch

import pytest

from backend.retrieval.query_router import (
    _classify_heuristic, _classify_llm, _content_words, _parse_llm_mode, route_query,
)


@pytest.fixture
def base_config():
    return {"query_router": {"use_llm_fallback": True, "llm_provider": "ollama",
            "llm_ollama_model": "llama3.2:3b", "llm_anthropic_model": "claude-haiku-4-5-20251001",
            "llm_max_tokens": 50}}


def test_content_words_preserve_domain_terms_and_remove_stopwords():
    words = _content_words("What is the enzyme?")
    assert "enzyme" in words and not {"the", "is"}.intersection(words)
    assert _content_words("") == _content_words("is the a an") == []


@pytest.mark.parametrize("question,expected", [
    ("", "none"), ("Hi", "none"), ("Hello, how are you?", "none"),
    ("Thanks for the help!", "none"), ("is the a", "none"),
    ("What is an enzyme?", "dense"), ("What are the products of glycolysis?", "dense"),
    ("Define activation energy.", "dense"), ("Who invented the RAG framework?", "dense"),
    ("When did Watson and Crick discover DNA?", "dense"), ("List the stages of mitosis.", "dense"),
    ("How many ATP molecules does glycolysis produce?", "dense"),
    ("Which enzyme catalyses this reaction?", "dense"), ("Name the four DNA bases.", "dense"),
    ("Why does temperature affect enzyme activity?", "graph"),
    ("How does pH affect the rate of enzyme catalysis?", "graph"),
    ("Compare RAG and GraphRAG retrieval methods.", "graph"),
    ("Contrast dense retrieval with graph retrieval.", "graph"),
    ("Explain the role of cofactors in enzyme function.", "graph"),
    ("What is the difference between competitive and non-competitive inhibition?", "graph"),
    ("Describe the relationship between substrate concentration and reaction rate.", "graph"),
    ("How does temperature affect enzyme kinetics?", "graph"),
    ("Enzyme inhibition leads to decreased metabolic output.", "graph"),
    ("Explain hallucination in terms of retrieval quality.", "graph"),
    ("Why is RAG better than pure LLM generation?", "graph"),
    ("Tell me something about enzymes and metabolism.", None),
    ("Enzymes are proteins that catalyse reactions.", None),
])
def test_heuristic_routes_question_signals(question, expected):
    assert _classify_heuristic(question) == expected


@pytest.mark.parametrize("response,expected", [
    ("dense", "dense"), ("graph", "graph"), ("none", "none"),
    ("  graph  ", "graph"), ("DENSE", "dense"), ('"graph"', "graph"),
    ("The answer is graph because multiple hops needed.", "graph"),
    ("I cannot classify this.", "dense"), ("", "dense"),
])
def test_parse_llm_mode_accepts_wrapping_and_defaults_unknown_output(response, expected):
    assert _parse_llm_mode(response) == expected


@pytest.mark.parametrize("provider,mode", [("ollama", "graph"), ("anthropic", "dense")])
def test_provider_dispatch_returns_selected_mode(base_config, provider, mode):
    cfg = {**base_config["query_router"], "llm_provider": provider}
    with patch(f"backend.retrieval.query_router._classify_llm_{provider}", return_value=mode) as model:
        assert _classify_llm("some question", cfg) == mode
        model.assert_called_once()


def test_model_failure_defaults_to_dense(base_config):
    with patch("backend.retrieval.query_router._classify_llm_ollama", side_effect=RuntimeError("offline")):
        assert _classify_llm("some question", base_config["query_router"]) == "dense"


def test_unknown_provider_defaults_to_dense(base_config):
    assert _classify_llm("some question", {**base_config["query_router"], "llm_provider": "unknown"}) == "dense"


@pytest.mark.parametrize("question,expected", [
    ("", "none"), ("   ", "none"), ("What is an enzyme?", "dense"),
    ("Why does pH affect enzyme activity?", "graph"),
])
def test_heuristic_routes_never_spend_a_model_call(base_config, question, expected):
    with patch("backend.retrieval.query_router._classify_llm") as model:
        assert route_query(question, base_config) == expected
        model.assert_not_called()


def test_ambiguous_question_returns_model_decision(base_config):
    with patch("backend.retrieval.query_router._classify_llm", return_value="graph") as model:
        assert route_query("Tell me about enzymes and metabolism.", base_config) == "graph"
        model.assert_called_once()


def test_disabled_model_defaults_ambiguous_question_to_dense():
    with patch("backend.retrieval.query_router._classify_llm") as model:
        assert route_query("Tell me about enzymes and metabolism.", {"query_router": {"use_llm_fallback": False}}) == "dense"
        model.assert_not_called()
