"""
NER helpers used by Stage-1 graph extractors.

The hierarchical pipeline (topic_extractor / subtopic_extractor /
content_synthesizer) treats entity identification as an optional pre-step:
an extractor may seed its prompt with NER output to improve recall.

This module exposes two drop-in NER backends (GLiNER + spaCy) and a
factory that selects between them from the ``entity_extractor`` section
of config/llm.yaml.

Public API:
    ner = build_ner_backend(config)
    entities = ner.extract(text)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseNER(ABC):
    @abstractmethod
    def extract(self, text: str) -> list[str]:
        """Return a deduplicated list of entity strings found in text."""


class GLiNERBackend(BaseNER):
    """Zero-shot NER via GLiNER. No domain-specific training required."""

    # GLiNER's tokenizer truncates at 384 tokens (~1 500 chars).
    # Split long texts into overlapping windows and merge results.
    _MAX_CHARS = 1200
    _OVERLAP_CHARS = 100

    def __init__(self, model_name: str, labels: list[str], threshold: float) -> None:
        from gliner import GLiNER
        logger.info(f"Loading GLiNER model '{model_name}'…")
        self._model = GLiNER.from_pretrained(model_name)
        self._labels = labels
        self._threshold = threshold

    def extract(self, text: str) -> list[str]:
        segments = self._split(text)
        seen: set[str] = set()
        entities: list[str] = []
        for segment in segments:
            for pred in self._model.predict_entities(
                segment, self._labels, threshold=self._threshold
            ):
                name = pred["text"].strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    entities.append(name)
        return entities

    def _split(self, text: str) -> list[str]:
        if len(text) <= self._MAX_CHARS:
            return [text]
        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._MAX_CHARS
            if end < len(text):
                boundary = max(
                    text.rfind(". ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("\n", start, end),
                )
                if boundary > start:
                    end = boundary + 1
            segments.append(text[start:end].strip())
            start = end - self._OVERLAP_CHARS
        return [s for s in segments if s]


class SpacyBackend(BaseNER):
    """Classic NER via spaCy. Requires: python -m spacy download <model>."""

    def __init__(self, model_name: str) -> None:
        import spacy
        logger.info(f"Loading spaCy model '{model_name}'…")
        self._nlp = spacy.load(model_name)

    def extract(self, text: str) -> list[str]:
        doc = self._nlp(text)
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            name = ent.text.strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                entities.append(name)
        return entities


def build_ner_backend(config: dict) -> BaseNER:
    """
    Construct the configured NER backend. Accepts either the full parsed
    ``llm.yaml`` dict (reads ``entity_extractor`` section) or that section
    directly.
    """
    cfg = config.get("entity_extractor", config)
    backend = cfg.get("ner_backend", "gliner")
    if backend == "gliner":
        return GLiNERBackend(
            model_name=cfg.get("gliner_model", "urchade/gliner_medium-v2.1"),
            labels=cfg.get("gliner_labels", ["concept", "process", "theory",
                                              "law", "principle", "entity"]),
            threshold=cfg.get("gliner_threshold", 0.5),
        )
    if backend == "spacy":
        return SpacyBackend(model_name=cfg.get("spacy_model", "en_core_web_sm"))
    raise ValueError(f"Unknown NER backend: '{backend}'. Use 'gliner' or 'spacy'.")
