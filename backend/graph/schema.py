"""
Stage 2 — Concept Graph: schema.py

Source of truth for node attributes, edge payloads, and Stage-1 proposal
objects used by the hierarchical Topic → Subtopic → Content graph.

Node types:   topic | subtopic | content
Structural edges:  HAS_SUBTOPIC, HAS_CONTENT
Semantic edges:    RELATED_TO  (carries a free-form label + confidence)

Stage 1 extractors emit the *Proposal classes. Stage 2 (GraphBuilder.apply_proposals)
resolves display names to stable keys and performs dedup/merge against the
existing graph.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


NodeType = Literal["topic", "subtopic", "content"]
TopicScope = Literal["broad", "narrow"]
ContentType = Literal[
    "definition", "theorem", "technique", "example", "question", "figure", "other"
]
IllustrationKind = Literal["diagram", "equation", "code", "image"]

StructuralRelation = Literal["HAS_SUBTOPIC", "HAS_CONTENT"]
SemanticRelation = Literal["RELATED_TO"]

STRUCTURAL_RELATIONS: frozenset[str] = frozenset({"HAS_SUBTOPIC", "HAS_CONTENT"})
SEMANTIC_RELATIONS: frozenset[str] = frozenset({"RELATED_TO"})
ALL_RELATIONS: frozenset[str] = STRUCTURAL_RELATIONS | SEMANTIC_RELATIONS


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------

_SLUG_LEN = 12


def _canonical(name: str) -> str:
    """Whitespace-collapsed, lowercase form used for hashing."""
    return re.sub(r"\s+", " ", name.strip().lower())


def make_key(node_type: NodeType, display_name: str) -> str:
    """
    Deterministic key for a Topic/Subtopic/Content node.
    Form: ``<node_type>:<sha1(canonical_name)[:12]>``.
    """
    if not display_name or not display_name.strip():
        raise ValueError("display_name must be non-empty")
    slug = hashlib.sha1(_canonical(display_name).encode("utf-8")).hexdigest()[:_SLUG_LEN]
    return f"{node_type}:{slug}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Shared structured payloads
# ---------------------------------------------------------------------------

class Illustration(BaseModel):
    kind: IllustrationKind
    hint: str


# ---------------------------------------------------------------------------
# Node attribute classes — written into the NetworkX graph as dicts
# ---------------------------------------------------------------------------

class _NodeAttrsBase(BaseModel):
    """Fields common to every node."""

    name: str
    node_type: NodeType
    origin: Literal["ingested", "manual"] = "ingested"
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    source_ids: list[str] = Field(default_factory=list)

    def to_graph_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=False)


class TopicAttrs(_NodeAttrsBase):
    node_type: Literal["topic"] = "topic"
    summary: str = ""
    scope: TopicScope = "broad"
    illustration: Illustration | None = None


class SubtopicAttrs(_NodeAttrsBase):
    node_type: Literal["subtopic"] = "subtopic"
    summary: str = ""
    illustration: Illustration | None = None
    parent_topic_keys: list[str] = Field(default_factory=list)


class ContentAttrs(_NodeAttrsBase):
    node_type: Literal["content"] = "content"
    content_type: ContentType = "other"
    summary: str = ""
    raw_excerpt: str = ""
    key_terms: list[str] = Field(default_factory=list)
    illustration: Illustration | None = None
    illustration_path: str | None = None
    parent_subtopic_keys: list[str] = Field(default_factory=list)


NodeAttrs = TopicAttrs | SubtopicAttrs | ContentAttrs


# ---------------------------------------------------------------------------
# Stage-1 Proposal classes — LLM extractor outputs, pre-dedup, pre-keying
# ---------------------------------------------------------------------------

class TopicProposal(BaseModel):
    name: str
    summary: str = ""
    scope: TopicScope = "broad"
    illustration: Illustration | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)


class SubtopicProposal(BaseModel):
    name: str
    summary: str = ""
    illustration: Illustration | None = None
    # Parents referenced by display name; Stage 2 resolves to keys.
    parent_topic_names: list[str] = Field(default_factory=list)
    parent_subtopic_names: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class ContentProposal(BaseModel):
    title: str
    content_type: ContentType = "other"
    summary: str = ""
    raw_excerpt: str = ""
    key_terms: list[str] = Field(default_factory=list)
    illustration: Illustration | None = None
    parent_subtopic_names: list[str] = Field(default_factory=list)
    parent_topic_names: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class RelatedToProposal(BaseModel):
    """Semantic cross-link between any two keyed nodes (topic/subtopic/content)."""

    source_key: str
    target_key: str
    label: str
    confidence: float = 0.5
    source_chunk_ids: list[str] = Field(default_factory=list)


class ProposalBundle(BaseModel):
    """
    Everything Stage 1 produces for a single ingestion batch.
    Consumed atomically by GraphBuilder.apply_proposals.
    """

    topics: list[TopicProposal] = Field(default_factory=list)
    subtopics: list[SubtopicProposal] = Field(default_factory=list)
    contents: list[ContentProposal] = Field(default_factory=list)
    related: list[RelatedToProposal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Label normalisation for RELATED_TO edges
# ---------------------------------------------------------------------------

_LABEL_CLEAN = re.compile(r"[^a-z0-9]+")


def normalise_relation_label(label: str) -> str:
    """
    Normalise an LLM-generated relation label to snake_case.
    Empty input returns "related_to" as a safe default.
    """
    if not label:
        return "related_to"
    cleaned = _LABEL_CLEAN.sub("_", label.strip().lower()).strip("_")
    return cleaned or "related_to"
