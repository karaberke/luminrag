"""
Tests for backend/graph/subtopic_extractor.py — Stage 1.3.

LLM is mocked. The extractor iterates per chunk and carries source_chunk_ids
forward; these tests focus on the parsing + per-chunk attribution logic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.graph.subtopic_extractor import extract_subtopics
from backend.schemas import Chunk


@pytest.fixture
def config() -> dict:
    return {"subtopic_extractor": {"merge_threshold": 0.82, "max_tokens": 256}}


def _chunk(cid: str, text: str = "default") -> Chunk:
    return Chunk(id=cid, text=text, source_id="doc.pdf", modality="pdf")


class TestExtractSubtopics:
    def test_parses_subtopics_and_attributes_chunk(self, config):
        with patch("backend.graph.subtopic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"name": "Transformers", "summary": "attention-based",
                 "parent_topic": "NLP", "parent_subtopic": ""},
            ]
            subs = extract_subtopics(
                [_chunk("c1")], topic_names=["NLP"], config=config,
            )
        assert len(subs) == 1
        assert subs[0].name == "Transformers"
        assert subs[0].source_chunk_ids == ["c1"]
        assert subs[0].parent_topic_names == ["NLP"]

    def test_empty_chunk_text_skipped(self, config):
        with patch("backend.graph.subtopic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"name": "X", "summary": ""}]
            subs = extract_subtopics(
                [_chunk("c1", text="   ")], topic_names=[], config=config,
            )
        mock_llm.assert_not_called()
        assert subs == []

    def test_nesting_parent_subtopic_propagated(self, config):
        with patch("backend.graph.subtopic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"name": "Self-Attention", "summary": "",
                 "parent_topic": "NLP", "parent_subtopic": "Transformers"},
            ]
            subs = extract_subtopics(
                [_chunk("c1")], topic_names=["NLP"], config=config,
            )
        assert subs[0].parent_subtopic_names == ["Transformers"]

    def test_falls_back_to_all_topics_when_parent_topic_missing(self, config):
        with patch("backend.graph.subtopic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"name": "Sub", "summary": "", "parent_topic": "", "parent_subtopic": ""},
            ]
            subs = extract_subtopics(
                [_chunk("c1")], topic_names=["NLP", "ML"], config=config,
            )
        assert subs[0].parent_topic_names == ["NLP", "ML"]

    def test_malformed_entries_dropped(self, config):
        with patch("backend.graph.subtopic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [
                {"name": "Valid", "summary": ""},
                {"missing_name": "x"},
                {"name": "", "summary": ""},
            ]
            subs = extract_subtopics(
                [_chunk("c1")], topic_names=["T"], config=config,
            )
        assert [s.name for s in subs] == ["Valid"]

    def test_iterates_across_multiple_chunks(self, config):
        with patch("backend.graph.subtopic_extractor.safe_call_json") as mock_llm:
            mock_llm.return_value = [{"name": "Sub", "summary": ""}]
            subs = extract_subtopics(
                [_chunk("c1"), _chunk("c2")], topic_names=["T"], config=config,
            )
        # Same "Sub" name from both chunks — deduped into one proposal merging both chunk ids
        assert len(subs) == 1
        assert set(subs[0].source_chunk_ids) == {"c1", "c2"}
