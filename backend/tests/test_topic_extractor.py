"""
Tests for backend/graph/topic_extractor.py — Stage 1.2.

The LLM is mocked via `safe_call_json` so no network calls happen. Synonym
swap via embeddings is covered with a tiny fake embedder.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.graph.topic_extractor import extract_topics


@pytest.fixture
def config() -> dict:
    return {"topic_extractor": {"merge_threshold": 0.85, "max_tokens": 256}}


def _fake_embedder(mapping: dict[str, np.ndarray], dim: int = 4) -> MagicMock:
    default = np.zeros(dim, dtype=np.float32)
    default[-1] = 1.0

    def _vec(text: str) -> np.ndarray:
        return mapping.get(text.strip().lower(), default)

    mock = MagicMock()
    mock.embed.side_effect = lambda texts: np.stack([_vec(t) for t in texts])
    mock.embed_one.side_effect = _vec
    mock.dimension = dim
    return mock


class TestExtractTopics:
    def test_parses_topic_proposals(self, config):
        # Use a long-enough preview so the dynamic max_topics cap allows 2
        long_preview = "Graph Theory and Attention Is All You Need " * 40
        with patch("backend.graph.topic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"name": "Graph Theory", "scope": "broad", "summary": "math of graphs"},
                {"name": "Attention Is All You Need", "scope": "narrow", "summary": "paper"},
            ]
            topics = extract_topics(long_preview, ["c1"], config)
        assert len(topics) == 2
        assert topics[0].name == "Graph Theory"
        assert topics[0].scope == "broad"
        assert topics[1].scope == "narrow"

    def test_returns_empty_on_llm_failure(self, config):
        with patch("backend.graph.topic_extractor.safe_call_json", return_value=[]):
            topics = extract_topics("preview", ["c1"], config)
        assert topics == []

    def test_source_chunk_ids_propagated(self, config):
        with patch("backend.graph.topic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"name": "T", "scope": "broad", "summary": ""}]
            topics = extract_topics("p", ["a", "b"], config)
        assert topics[0].source_chunk_ids == ["a", "b"]

    def test_malformed_entries_dropped(self, config):
        with patch("backend.graph.topic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"name": "Valid", "scope": "broad", "summary": "ok"},
                {"missing_name": "x"},
                {"name": "", "scope": "broad"},
            ]
            topics = extract_topics("p", ["c1"], config)
        assert [t.name for t in topics] == ["Valid"]

    def test_invalid_scope_defaults_to_broad(self, config):
        with patch("backend.graph.topic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"name": "T", "scope": "weird", "summary": ""}]
            topics = extract_topics("p", ["c1"], config)
        assert topics[0].scope == "broad"

    def test_synonym_swap_when_embedding_close(self, config):
        # "NLP" is an existing topic; embedder returns near-identical vectors
        nlp_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb = _fake_embedder({
            "natural language processing": nlp_vec,
            "nlp": nlp_vec,
        })
        with patch("backend.graph.topic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"name": "Natural Language Processing", "scope": "broad"}]
            topics = extract_topics(
                "preview", ["c1"], config,
                existing_names=["NLP"], embedder=emb,
            )
        # Should have been swapped to the existing display name "NLP"
        assert topics[0].name == "NLP"

    def test_no_swap_when_below_threshold(self, config):
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # orthogonal
        emb = _fake_embedder({"graph theory": v1, "linguistics": v2})
        with patch("backend.graph.topic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"name": "Graph Theory", "scope": "broad"}]
            topics = extract_topics(
                "preview", ["c1"], config,
                existing_names=["Linguistics"], embedder=emb,
            )
        assert topics[0].name == "Graph Theory"  # unchanged
