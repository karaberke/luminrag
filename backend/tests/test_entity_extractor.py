"""
Tests for backend/graph/entity_extractor.py

Only covers the public NER helper surface that remains after the hierarchical
extractor migration. NER backends are mocked — no model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.graph.entity_extractor import (
    BaseNER,
    GLiNERBackend,
    SpacyBackend,
    build_ner_backend,
)


# ---------------------------------------------------------------------------
# build_ner_backend
# ---------------------------------------------------------------------------

class TestBuildNerBackend:
    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown NER backend"):
            build_ner_backend({"entity_extractor": {"ner_backend": "bogus"}})

    def test_accepts_section_passed_directly(self):
        """Caller may pass the entity_extractor sub-section directly."""
        with patch("gliner.GLiNER") as mock_cls:
            mock_cls.from_pretrained = MagicMock(return_value=MagicMock())
            ner = build_ner_backend({"ner_backend": "gliner", "gliner_model": "m"})
        assert isinstance(ner, GLiNERBackend)

    def test_full_config_dict(self):
        with patch("gliner.GLiNER") as mock_cls:
            mock_cls.from_pretrained = MagicMock(return_value=MagicMock())
            ner = build_ner_backend({
                "entity_extractor": {"ner_backend": "gliner", "gliner_model": "m"}
            })
        assert isinstance(ner, GLiNERBackend)


# ---------------------------------------------------------------------------
# GLiNER backend
# ---------------------------------------------------------------------------

class TestGLiNERBackend:
    def test_extract_dedup_case_insensitive(self):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Enzyme"},
            {"text": "enzyme"},
            {"text": "Catalysis"},
        ]
        with patch("gliner.GLiNER") as mock_cls:
            mock_cls.from_pretrained = MagicMock(return_value=mock_model)
            ner = GLiNERBackend("m", ["concept"], 0.5)
        result = ner.extract("Enzymes are catalysts.")
        assert result == ["Enzyme", "Catalysis"]

    def test_split_short_text(self):
        with patch("gliner.GLiNER") as mock_cls:
            mock_cls.from_pretrained = MagicMock(return_value=MagicMock())
            ner = GLiNERBackend("m", ["concept"], 0.5)
        assert ner._split("short text") == ["short text"]

    def test_split_long_text_windows(self):
        with patch("gliner.GLiNER") as mock_cls:
            mock_cls.from_pretrained = MagicMock(return_value=MagicMock())
            ner = GLiNERBackend("m", ["concept"], 0.5)
        long_text = ("Sentence one. " * 200)  # ~2800 chars, well over _MAX_CHARS
        segments = ner._split(long_text)
        assert len(segments) > 1
        assert all(len(s) <= ner._MAX_CHARS + 1 for s in segments)


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

class TestSpacyBackend:
    def test_extract_returns_unique_entities(self):
        ent1 = MagicMock(); ent1.text = "Graph"
        ent2 = MagicMock(); ent2.text = "graph"  # duplicate, case differs
        ent3 = MagicMock(); ent3.text = "Theorem"
        mock_doc = MagicMock(); mock_doc.ents = [ent1, ent2, ent3]
        mock_nlp = MagicMock(return_value=mock_doc)
        with patch("spacy.load", return_value=mock_nlp):
            ner = SpacyBackend("en_core_web_sm")
        assert ner.extract("text") == ["Graph", "Theorem"]


# ---------------------------------------------------------------------------
# BaseNER abstract contract
# ---------------------------------------------------------------------------

class TestBaseNER:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            BaseNER()  # type: ignore[abstract]
