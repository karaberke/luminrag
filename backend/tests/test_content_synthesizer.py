"""
Tests for backend/graph/content_synthesizer.py — Stage 1.4.

LLM is mocked. Tests cover single-summary parsing, illustration extraction,
content_type validation, max_per_chunk cap, and evidence wiring.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.graph.content_synthesizer import synthesize_contents
from backend.schemas import Chunk


@pytest.fixture
def config() -> dict:
    return {"content_synthesizer": {"merge_threshold": 0.88, "max_tokens": 1200, "max_per_chunk": 3}}


def _chunk() -> Chunk:
    return Chunk(id="c1", text="Some teachable content here.", source_id="d.pdf", modality="pdf")


class TestSynthesizeContents:
    def test_parses_full_content_entry(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{
                "title": "Self-Attention",
                "content_type": "definition",
                "summary": "Attention is a mechanism that computes a weighted sum of values.",
                "parent_subtopic": "Transformers",
                "illustration_kind": "equation",
                "illustration_hint": "softmax(QK^T/sqrt(d))V",
            }]
            contents = synthesize_contents(
                _chunk(), ["Transformers"], ["NLP"], config,
            )
        assert len(contents) == 1
        c = contents[0]
        assert c.title == "Self-Attention"
        assert c.content_type == "definition"
        assert "weighted sum" in c.summary
        assert c.illustration is not None
        assert c.illustration.kind == "equation"
        assert c.parent_subtopic_names == ["Transformers"]
        assert c.parent_topic_names == ["NLP"]
        assert c.evidence_chunk_ids == ["c1"]

    def test_invalid_content_type_falls_back_to_other(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{
                "title": "X", "content_type": "weird-type",
                "summary": "",
            }]
            contents = synthesize_contents(_chunk(), [], ["T"], config)
        assert contents[0].content_type == "other"

    def test_missing_illustration_yields_none(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"title": "X", "summary": ""}]
            contents = synthesize_contents(_chunk(), [], ["T"], config)
        assert contents[0].illustration is None

    def test_empty_chunk_text_skipped(self, config):
        chunk = Chunk(id="c1", text="", source_id="d.pdf", modality="pdf")
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            contents = synthesize_contents(chunk, [], ["T"], config)
        mock_llm.assert_not_called()
        assert contents == []

    def test_falls_back_to_subtopics_when_parent_empty(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"title": "X", "parent_subtopic": "", "summary": "x"}]
            contents = synthesize_contents(
                _chunk(), ["A", "B"], ["T"], config,
            )
        assert contents[0].parent_subtopic_names == ["A", "B"]

    def test_illegal_illustration_kind_dropped(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{
                "title": "X",
                "illustration_kind": "photograph",
                "illustration_hint": "x",
                "summary": "",
            }]
            contents = synthesize_contents(_chunk(), [], ["T"], config)
        assert contents[0].illustration is None

    def test_max_per_chunk_cap_respected(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"title": f"Unit{i}", "summary": "text"} for i in range(10)
            ]
            contents = synthesize_contents(_chunk(), [], ["T"], config)
        assert len(contents) <= 3

    def test_raw_excerpt_captured_and_trimmed(self, config):
        long_quote = "x" * 500
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{
                "title": "Unit",
                "summary": "s",
                "raw_excerpt": long_quote,
                "key_terms": ["Alpha", "Beta"],
            }]
            contents = synthesize_contents(_chunk(), [], ["T"], config)
        assert contents[0].raw_excerpt == "x" * 300
        assert contents[0].key_terms == ["Alpha", "Beta"]

    def test_key_terms_dedup_and_filter_non_strings(self, config):
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [{
                "title": "Unit",
                "summary": "s",
                "key_terms": ["vmax", "Vmax", "", None, 42, "  Km  "],
            }]
            contents = synthesize_contents(_chunk(), [], ["T"], config)
        # "Vmax" dedup'd vs "vmax" (case-insensitive); only strings kept; trimmed.
        assert contents[0].key_terms == ["vmax", "Km"]

    def test_dynamic_max_per_chunk_uses_length_when_no_override(self):
        cfg = {"content_synthesizer": {"merge_threshold": 0.88, "max_tokens": 1200}}
        # Long chunk (>=3000 chars) → cap 10
        long_chunk = Chunk(
            id="c1", text="word " * 1000, source_id="d.pdf", modality="pdf"
        )
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"title": f"U{i}", "summary": "s"} for i in range(15)
            ]
            contents = synthesize_contents(long_chunk, [], ["T"], cfg)
        assert len(contents) == 10

    def test_dynamic_max_per_chunk_short_chunk_still_has_min_three(self):
        cfg = {"content_synthesizer": {"merge_threshold": 0.88, "max_tokens": 1200}}
        short_chunk = Chunk(
            id="c1", text="short text", source_id="d.pdf", modality="pdf"
        )
        with patch("backend.graph.content_synthesizer.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"title": f"U{i}", "summary": "s"} for i in range(5)
            ]
            contents = synthesize_contents(short_chunk, [], ["T"], cfg)
        assert len(contents) == 3
