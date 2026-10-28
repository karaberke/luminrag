"""
Stage 2 — Concept Graph: schema.py

Source of truth for node attributes, edge payloads, and Stage-1 proposal
objects used by the hierarchical Topic → Subtopic → Content graph.

Node types:   topic | subtopic | content
Structural edges:  HAS_SUBTOPIC, HAS_CONTENT
Semantic edges:    RELATED_TO  (plain association with independent backing)

Stage 1 extractors emit the *Proposal classes. Stage 2 (GraphBuilder.apply_proposals)
resolves display names to stable keys and performs dedup/merge against the
existing graph.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.utils.associations import AssociationSupport, MAX_SUPPORTS

MAX_GRAPH_NAME = 500
MAX_GRAPH_TEXT = 100_000
MAX_KEY_TERMS = 200
MAX_GRAPH_PARENTS = 200
MAX_GRAPH_REFERENCES = 20_000
MAX_ILLUSTRATION_HINT = 10_000
NODE_SCHEMA_VERSION = 1
GraphText = Annotated[str, Field(strict=True, max_length=MAX_GRAPH_NAME)]
GraphReference = Annotated[str, Field(strict=True, min_length=1, max_length=1024)]


def bounded_key_terms(values: list[str]) -> list[str]:
    """Keep a bounded, ordered union of generated descriptive terms."""
    result, seen = [], set()
    for term in values:
        if term and term.casefold() not in seen:
            result.append(term)
            seen.add(term.casefold())
            if len(result) == MAX_KEY_TERMS:
                break
    return result


NodeType = Literal["topic", "subtopic", "content", "note"]
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
    hint: str = Field(max_length=MAX_ILLUSTRATION_HINT)


class NodeInput(BaseModel):
    """Shared persisted/editable node boundary; imports are never truncated."""
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=MAX_GRAPH_NAME)
    node_type: NodeType = "note"
    title: GraphText = ""
    body: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    summary: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    raw_excerpt: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    key_terms: list[GraphText] = Field(default_factory=list, max_length=MAX_KEY_TERMS)
    source_ids: list[GraphReference] = Field(default_factory=list, max_length=MAX_GRAPH_REFERENCES)
    parent_topic_keys: list[GraphText] = Field(default_factory=list, max_length=MAX_GRAPH_PARENTS)
    parent_subtopic_keys: list[GraphText] = Field(default_factory=list, max_length=MAX_GRAPH_PARENTS)
    scope: TopicScope = "broad"
    content_type: ContentType = "other"
    illustration: Illustration | None = None
    illustration_path: str | None = Field(default=None, max_length=2048)
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class ChunkReferenceInput(NodeInput):
    """Legacy evidence nodes have no required display name, but typed references."""
    name: str = Field(default="", max_length=MAX_GRAPH_NAME)
    node_type: Literal["chunk_ref"] = "chunk_ref"


class EdgeInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    relation: Literal["HAS_SUBTOPIC", "HAS_CONTENT", "RELATED_TO", "EVIDENCE_OF"]
    label: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    source_chunk_ids: list[GraphReference] = Field(default_factory=list, max_length=MAX_GRAPH_REFERENCES)
    supports: list[AssociationSupport] | None = Field(default=None, min_length=1, max_length=MAX_SUPPORTS)


def validate_node_attrs(attrs: dict) -> NodeInput:
    if attrs.get("node_type") not in {"topic", "subtopic", "content", "note", "chunk_ref"}:
        raise ValueError("Invalid graph node type.")
    model = ChunkReferenceInput if attrs.get("node_type") == "chunk_ref" else NodeInput
    return model.model_validate(attrs)


def legacy_node_schema(version: object) -> bool:
    """Only unversioned graphs predate the shared persisted-node bounds."""
    if version is None:
        return True
    if type(version) is not int or version != NODE_SCHEMA_VERSION:
        raise ValueError("Unsupported graph node schema version.")
    return False


def normalize_legacy_node_attrs(attrs: dict) -> dict:
    """Bound legacy descriptions without repairing or dropping references.

    Keys, sources, parents, coordinates and source-backed passages are retained.
    Incorrect types and reference overflows still fail the shared validator.
    """
    normalized = dict(attrs)
    for field in ("name", "title"):
        if isinstance(normalized.get(field), str):
            normalized[field] = normalized[field][:MAX_GRAPH_NAME]
    terms = normalized.get("key_terms")
    if isinstance(terms, list) and all(isinstance(term, str) for term in terms):
        if len(terms) > MAX_KEY_TERMS or any(len(term) > MAX_GRAPH_NAME for term in terms):
            normalized["key_terms"] = bounded_key_terms([term[:MAX_GRAPH_NAME] for term in terms])
    illustration = normalized.get("illustration")
    if isinstance(illustration, dict) and isinstance(illustration.get("hint"), str):
        normalized["illustration"] = {**illustration, "hint": illustration["hint"][:MAX_ILLUSTRATION_HINT]}
    validate_node_attrs(normalized)
    return normalized


def normalize_legacy_edge_attrs(attrs: dict) -> dict:
    """Bound old display labels while keeping all relationship backing intact."""
    normalized = dict(attrs)
    if isinstance(normalized.get("label"), str):
        normalized["label"] = normalized["label"][:200]
    EdgeInput.model_validate(normalized)
    return normalized


# ---------------------------------------------------------------------------
# Node attribute classes — written into the NetworkX graph as dicts
# ---------------------------------------------------------------------------

class _NodeAttrsBase(BaseModel):
    """Fields common to every node."""

    name: str = Field(min_length=1, max_length=MAX_GRAPH_NAME)
    node_type: NodeType
    origin: Literal["ingested", "manual"] = "ingested"
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    source_ids: list[GraphReference] = Field(default_factory=list, max_length=MAX_GRAPH_REFERENCES)

    def to_graph_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=False)


class TopicAttrs(_NodeAttrsBase):
    node_type: Literal["topic"] = "topic"
    summary: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    scope: TopicScope = "broad"
    illustration: Illustration | None = None


class SubtopicAttrs(_NodeAttrsBase):
    node_type: Literal["subtopic"] = "subtopic"
    summary: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    illustration: Illustration | None = None
    parent_topic_keys: list[GraphText] = Field(default_factory=list, max_length=MAX_GRAPH_PARENTS)


class ContentAttrs(_NodeAttrsBase):
    node_type: Literal["content"] = "content"
    content_type: ContentType = "other"
    summary: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    raw_excerpt: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    key_terms: list[GraphText] = Field(default_factory=list, max_length=MAX_KEY_TERMS)
    illustration: Illustration | None = None
    illustration_path: str | None = Field(default=None, max_length=2048)
    parent_subtopic_keys: list[GraphText] = Field(default_factory=list, max_length=MAX_GRAPH_PARENTS)


class NoteAttrs(_NodeAttrsBase):
    node_type: Literal["note"] = "note"
    origin: Literal["ingested", "manual"] = "manual"
    title: GraphText = ""
    body: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    summary: str = Field(default="", max_length=MAX_GRAPH_TEXT)
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


NodeAttrs = TopicAttrs | SubtopicAttrs | ContentAttrs | NoteAttrs


# ---------------------------------------------------------------------------
# Stage-1 Proposal classes — LLM extractor outputs, pre-dedup, pre-keying
# ---------------------------------------------------------------------------

class _ProposalBase(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def bound_generated_text(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for field in ("name", "title", "summary", "raw_excerpt"):
            if isinstance(value.get(field), str):
                limit = MAX_GRAPH_NAME if field in {"name", "title"} else MAX_GRAPH_TEXT
                value[field] = value[field][:limit]
        if isinstance(value.get("key_terms"), list):
            value["key_terms"] = bounded_key_terms([term[:MAX_GRAPH_NAME] for term in value["key_terms"] if isinstance(term, str)])
        for field in ("parent_topic_names", "parent_subtopic_names"):
            if isinstance(value.get(field), list):
                value[field] = list(dict.fromkeys(name[:MAX_GRAPH_NAME] for name in value[field] if isinstance(name, str)))[:MAX_GRAPH_PARENTS]
        return value


class TopicProposal(_ProposalBase):
    name: str
    summary: str = ""
    scope: TopicScope = "broad"
    illustration: Illustration | None = None
    source_chunk_ids: list[GraphReference] = Field(default_factory=list, max_length=MAX_GRAPH_REFERENCES)


class SubtopicProposal(_ProposalBase):
    name: str
    summary: str = ""
    illustration: Illustration | None = None
    # Parents referenced by display name; Stage 2 resolves to keys.
    parent_topic_names: list[str] = Field(default_factory=list)
    parent_subtopic_names: list[str] = Field(default_factory=list)
    source_chunk_ids: list[GraphReference] = Field(default_factory=list, max_length=MAX_GRAPH_REFERENCES)


class ContentProposal(_ProposalBase):
    title: str
    content_type: ContentType = "other"
    summary: str = ""
    raw_excerpt: str = ""
    key_terms: list[str] = Field(default_factory=list)
    illustration: Illustration | None = None
    parent_subtopic_names: list[str] = Field(default_factory=list)
    parent_topic_names: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[GraphReference] = Field(default_factory=list, max_length=MAX_GRAPH_REFERENCES)


class RelatedToProposal(BaseModel):
    """Semantic cross-link between any two keyed nodes (topic/subtopic/content)."""

    source_key: str
    target_key: str
    label: str | None = None  # Accepted from legacy callers; never used as a predicate.
    confidence: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    source_chunk_ids: list[str] = Field(default_factory=list)
    verified: bool = False


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
