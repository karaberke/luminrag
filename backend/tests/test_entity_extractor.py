"""
Tests for backend/graph/entity_extractor.py

NER backends and LLM calls are always mocked — no model downloads or
API calls during the test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.graph.entity_extractor import (
    VALID_RELATIONS,
    _build_triples,
    _extract_relations,
    _parse_llm_json,
    _build_ner_backend,
    extract_triples,
    GLiNERBackend,
    SpacyBackend,
)
from backend.schemas import Chunk, GraphTriple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> dict:
    return {
        "entity_extractor": {
            "ner_backend": "gliner",
            "gliner_model": "urchade/gliner_medium-v2.1",
            "gliner_labels": ["concept", "process"],
            "gliner_threshold": 0.5,
            "relation_provider": "ollama",
            "relation_ollama_model": "llama3.2:3b",
            "relation_max_tokens": 512,
        }
    }


def _chunk(id="src_chunk_0", text="Enzymes catalyse biochemical reactions."):
    return Chunk(id=id, text=text, source_id="lecture_01", modality="pdf")


# ---------------------------------------------------------------------------
# _parse_llm_json
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    def test_parses_plain_json(self):
        raw = '[{"head": "A", "relation": "CAUSES", "tail": "B"}]'
        result = _parse_llm_json(raw)
        assert result[0]["head"] == "A"

    def test_strips_markdown_code_fence(self):
        raw = '```json\n[{"head": "A", "relation": "CAUSES", "tail": "B"}]\n```'
        result = _parse_llm_json(raw)
        assert len(result) == 1

    def test_strips_plain_code_fence(self):
        raw = '```\n[{"head": "X", "relation": "PART_OF", "tail": "Y"}]\n```'
        result = _parse_llm_json(raw)
        assert result[0]["relation"] == "PART_OF"

    def test_empty_array(self):
        assert _parse_llm_json("[]") == []

    def test_raises_on_invalid_json(self):
        with pytest.raises(Exception):
            _parse_llm_json("not json at all")


# ---------------------------------------------------------------------------
# _build_triples
# ---------------------------------------------------------------------------

class TestBuildTriples:
    def test_valid_triple_returned(self):
        relations = [{"head": "Enzyme", "relation": "CAUSES", "tail": "ReactionRate"}]
        triples = _build_triples("chunk_0", relations)
        assert len(triples) == 1
        assert isinstance(triples[0], GraphTriple)

    def test_source_chunk_id_attached(self):
        relations = [{"head": "A", "relation": "PART_OF", "tail": "B"}]
        triples = _build_triples("my_chunk", relations)
        assert triples[0].source_chunk_ids == ["my_chunk"]

    def test_invalid_relation_filtered(self):
        relations = [{"head": "A", "relation": "HATES", "tail": "B"}]
        assert _build_triples("c", relations) == []

    def test_relation_uppercased(self):
        relations = [{"head": "A", "relation": "causes", "tail": "B"}]
        triples = _build_triples("c", relations)
        assert triples[0].relation == "CAUSES"

    def test_missing_key_skipped(self):
        relations = [{"head": "A", "relation": "CAUSES"}]  # no tail
        assert _build_triples("c", relations) == []

    def test_self_loop_skipped(self):
        relations = [{"head": "Enzyme", "relation": "CAUSES", "tail": "Enzyme"}]
        assert _build_triples("c", relations) == []

    def test_empty_head_skipped(self):
        relations = [{"head": "", "relation": "CAUSES", "tail": "B"}]
        assert _build_triples("c", relations) == []

    def test_all_valid_relations_accepted(self):
        for relation in VALID_RELATIONS:
            relations = [{"head": "A", "relation": relation, "tail": "B"}]
            triples = _build_triples("c", relations)
            assert len(triples) == 1

    def test_multiple_relations(self):
        relations = [
            {"head": "Enzyme", "relation": "CAUSES", "tail": "Rate"},
            {"head": "Rate", "relation": "PART_OF", "tail": "Kinetics"},
        ]
        triples = _build_triples("c", relations)
        assert len(triples) == 2


# ---------------------------------------------------------------------------
# _build_ner_backend
# ---------------------------------------------------------------------------

class TestBuildNerBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown NER backend"):
            _build_ner_backend({"ner_backend": "bert"})

    def test_returns_gliner_when_configured(self):
        with patch("backend.graph.entity_extractor.GLiNERBackend") as MockGLiNER:
            MockGLiNER.return_value = MagicMock(spec=GLiNERBackend)
            backend = _build_ner_backend({
                "ner_backend": "gliner",
                "gliner_model": "urchade/gliner_medium-v2.1",
                "gliner_labels": ["concept"],
                "gliner_threshold": 0.5,
            })
            MockGLiNER.assert_called_once()

    def test_returns_spacy_when_configured(self):
        with patch("backend.graph.entity_extractor.SpacyBackend") as MockSpacy:
            MockSpacy.return_value = MagicMock(spec=SpacyBackend)
            _build_ner_backend({"ner_backend": "spacy", "spacy_model": "en_core_web_sm"})
            MockSpacy.assert_called_once()


# ---------------------------------------------------------------------------
# _extract_relations
# ---------------------------------------------------------------------------

class TestExtractRelations:
    def test_ollama_path_called(self, base_config):
        cfg = base_config["entity_extractor"]
        with patch("backend.graph.entity_extractor._extract_relations_ollama") as mock:
            mock.return_value = []
            _extract_relations("text", ["A", "B"], cfg)
            mock.assert_called_once()

    def test_anthropic_path_called(self, base_config):
        cfg = {**base_config["entity_extractor"], "relation_provider": "anthropic"}
        with patch("backend.graph.entity_extractor._extract_relations_anthropic") as mock:
            mock.return_value = []
            _extract_relations("text", ["A", "B"], cfg)
            mock.assert_called_once()

    def test_failure_returns_empty_list(self, base_config):
        cfg = base_config["entity_extractor"]
        with patch("backend.graph.entity_extractor._extract_relations_ollama",
                   side_effect=RuntimeError("Ollama down")):
            result = _extract_relations("text", ["A", "B"], cfg)
            assert result == []

    def test_unknown_provider_returns_empty_list(self, base_config):
        cfg = {**base_config["entity_extractor"], "relation_provider": "openai"}
        result = _extract_relations("text", ["A", "B"], cfg)
        assert result == []


# ---------------------------------------------------------------------------
# extract_triples (integration, NER + LLM mocked)
# ---------------------------------------------------------------------------

class TestExtractTriples:
    def _run(self, chunks, config, ner_entities, llm_relations):
        mock_ner = MagicMock()
        mock_ner.extract.return_value = ner_entities

        with patch("backend.graph.entity_extractor._build_ner_backend",
                   return_value=mock_ner), \
             patch("backend.graph.entity_extractor._extract_relations",
                   return_value=llm_relations):
            return extract_triples(chunks, config)

    def test_returns_graph_triples(self, base_config):
        chunks = [_chunk()]
        relations = [{"head": "Enzyme", "relation": "CAUSES", "tail": "ReactionRate"}]
        result = self._run(chunks, base_config, ["Enzyme", "ReactionRate"], relations)
        assert all(isinstance(t, GraphTriple) for t in result)

    def test_chunk_with_one_entity_skipped(self, base_config):
        chunks = [_chunk()]
        result = self._run(chunks, base_config, ["OnlyOne"], [])
        assert result == []

    def test_empty_chunk_text_skipped(self, base_config):
        chunks = [Chunk(id="c0", text="   ", source_id="s", modality="pdf")]
        result = self._run(chunks, base_config, [], [])
        assert result == []

    def test_multiple_chunks_aggregated(self, base_config):
        chunks = [_chunk(id=f"c{i}") for i in range(3)]
        relations = [{"head": "A", "relation": "PART_OF", "tail": "B"}]
        result = self._run(chunks, base_config, ["A", "B"], relations)
        assert len(result) == 3  # one triple per chunk

    def test_invalid_llm_output_skipped(self, base_config):
        chunks = [_chunk()]
        bad_relations = [{"head": "A", "relation": "INVALID", "tail": "B"}]
        result = self._run(chunks, base_config, ["A", "B"], bad_relations)
        assert result == []

    def test_empty_chunk_list(self, base_config):
        result = self._run([], base_config, [], [])
        assert result == []
