"""
Stage 2 - Concept Graph Construction: entity_extractor.py

Two-step hybrid pipeline per chunk:
  Step 1 (NER)       — local model identifies entities (GLiNER or spaCy)
  Step 2 (Relations) — small LLM maps relations between those entities

Only the relation step hits the LLM; entity detection is fully local,
keeping API costs minimal.

Valid pedagogical relations:
    PART_OF, PREREQUISITE, CAUSES, EXAMPLE_OF, EXPLAINS, BELONGS_TO_TOPIC

Public API:
    triples = extract_triples(chunks, config)
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from backend.schemas import Chunk, GraphTriple

logger = logging.getLogger(__name__)

VALID_RELATIONS = frozenset({
    "PART_OF", "PREREQUISITE", "CAUSES",
    "EXAMPLE_OF", "EXPLAINS", "BELONGS_TO_TOPIC",
})

_RELATION_PROMPT = """\
You are building a knowledge graph for educational content.

Text chunk:
{text}

Identified entities: {entities}

Task: Find relationships between pairs of the above entities.

Valid relations (use ONLY these exact strings):
- PART_OF: A is a component or subset of B
- PREREQUISITE: understanding A requires knowing B first
- CAUSES: A leads to or produces B
- EXAMPLE_OF: A is a specific instance or example of B
- EXPLAINS: A provides an explanation or mechanism for B
- BELONGS_TO_TOPIC: A is categorised under topic B

Return ONLY a valid JSON array. Each element must have keys "head", "relation", "tail".
Both "head" and "tail" must be from the provided entities list exactly as written.
If no clear relationships exist, return: []

Example: [{{"head": "Enzyme", "relation": "CAUSES", "tail": "ReactionRate"}}]\
"""

# Maximum characters of chunk text sent to the relation LLM.
# Keeps prompt size bounded for large textbook chunks.
_MAX_TEXT_CHARS = 1500


# ---------------------------------------------------------------------------
# NER backends
# ---------------------------------------------------------------------------

class BaseNER(ABC):
    @abstractmethod
    def extract(self, text: str) -> list[str]:
        """Return a deduplicated list of entity strings found in text."""


class GLiNERBackend(BaseNER):
    """Zero-shot NER via GLiNER. No domain-specific training required."""

    def __init__(self, model_name: str, labels: list[str], threshold: float) -> None:
        from gliner import GLiNER
        logger.info(f"Loading GLiNER model '{model_name}'…")
        self._model = GLiNER.from_pretrained(model_name)
        self._labels = labels
        self._threshold = threshold

    # GLiNER's tokenizer truncates at 384 tokens (~1 500 chars).
    # Split long texts into overlapping windows and merge results.
    _MAX_CHARS = 1200
    _OVERLAP_CHARS = 100

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
        """Split text into segments that fit within GLiNER's token limit."""
        if len(text) <= self._MAX_CHARS:
            return [text]
        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._MAX_CHARS
            if end < len(text):
                # Break at the last sentence boundary within the window
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


def _build_ner_backend(cfg: dict) -> BaseNER:
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


# ---------------------------------------------------------------------------
# Relation extraction (LLM)
# ---------------------------------------------------------------------------

def _parse_llm_json(response: str) -> list[dict]:
    """
    Parse a JSON array from an LLM response that may be wrapped in
    markdown code fences (```json ... ```).
    """
    text = response.strip()
    # Strip optional markdown code block
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _extract_relations_ollama(text: str, entities: list[str], cfg: dict) -> list[dict]:
    import httpx

    prompt = _RELATION_PROMPT.format(
        text=text[:_MAX_TEXT_CHARS],
        entities=", ".join(entities),
    )
    payload = {
        "model": cfg.get("relation_ollama_model", "llama3.2:3b"),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_predict": cfg.get("relation_max_tokens", 512)},
    }
    base_url = cfg.get("relation_ollama_base_url", "http://localhost:11434")
    response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    return _parse_llm_json(response.json()["response"])


def _extract_relations_anthropic(text: str, entities: list[str], cfg: dict) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    prompt = _RELATION_PROMPT.format(
        text=text[:_MAX_TEXT_CHARS],
        entities=", ".join(entities),
    )
    message = client.messages.create(
        model=cfg.get("relation_anthropic_model", "claude-haiku-4-5-20251001"),
        max_tokens=cfg.get("relation_max_tokens", 512),
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_llm_json(message.content[0].text)


def _extract_relations(text: str, entities: list[str], cfg: dict) -> list[dict]:
    """Call the configured LLM to extract relations. Returns [] on failure."""
    provider = cfg.get("relation_provider", "ollama")
    try:
        if provider == "ollama":
            return _extract_relations_ollama(text, entities, cfg)
        if provider == "anthropic":
            return _extract_relations_anthropic(text, entities, cfg)
        raise ValueError(f"Unknown relation provider: '{provider}'")
    except Exception as exc:
        logger.warning(f"Relation extraction failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Triple assembly
# ---------------------------------------------------------------------------

def _build_triples(chunk_id: str, relations: list[dict]) -> list[GraphTriple]:
    """
    Convert raw LLM relation dicts into validated GraphTriple objects.
    Filters out entries with missing keys or invalid relation types.
    """
    triples: list[GraphTriple] = []
    for rel in relations:
        try:
            head = str(rel["head"]).strip()
            relation = str(rel["relation"]).strip().upper()
            tail = str(rel["tail"]).strip()
        except (KeyError, TypeError):
            logger.debug(f"Skipping malformed relation: {rel}")
            continue

        if relation not in VALID_RELATIONS:
            logger.debug(f"Skipping unknown relation type: '{relation}'")
            continue
        if not head or not tail or head == tail:
            continue

        triples.append(GraphTriple(
            head=head,
            relation=relation,
            tail=tail,
            source_chunk_ids=[chunk_id],
        ))
    return triples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_triples(chunks: list[Chunk], config: dict) -> list[GraphTriple]:
    """
    Run the full two-step extraction pipeline over a list of Chunks.

    The NER model is loaded once and reused across all chunks.

    Args:
        chunks: Output from any Stage 1 ingestion module.
        config: Parsed contents of config/llm.yaml.

    Returns:
        Flat list of GraphTriple objects across all chunks.
    """
    cfg = config.get("entity_extractor", {})
    ner = _build_ner_backend(cfg)

    all_triples: list[GraphTriple] = []

    for chunk in chunks:
        if not chunk.text.strip():
            continue

        entities = ner.extract(chunk.text)
        if len(entities) < 2:
            logger.debug(f"[{chunk.id}] Only {len(entities)} entity found, skipping.")
            continue

        logger.debug(f"[{chunk.id}] Entities: {entities}")
        raw_relations = _extract_relations(chunk.text, entities, cfg)
        triples = _build_triples(chunk.id, raw_relations)
        logger.debug(f"[{chunk.id}] {len(triples)} triples extracted")
        all_triples.extend(triples)

    logger.info(f"Total triples extracted: {len(all_triples)} from {len(chunks)} chunks")
    return all_triples
