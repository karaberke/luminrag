"""
Core Pydantic data schemas shared across all LuminRAG pipeline stages.
All inter-module data must be passed as instances of these models.
"""

from __future__ import annotations

from typing import Literal
import uuid

from pydantic import BaseModel, Field

MAX_CHUNK_TEXT_CHARACTERS = 100_000

Modality = Literal["video", "slide", "pdf", "image", "audio", "note", "markdown", "office"]
RoutingMode = Literal["dense", "graph", "hybrid", "none"]


class Chunk(BaseModel):
    """A single content chunk produced by any ingestion module."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_id: str
    modality: Modality
    metadata: dict = Field(default_factory=dict)

    # Video-specific metadata keys (stored in metadata dict):
    #   start_time: float       — segment start in seconds
    #   end_time: float         — segment end in seconds
    #   keyframe_path: str|None — path to extracted keyframe image
    #
    # Slide/PDF-specific metadata keys:
    #   page_number: int
    #   slide_title: str|None
    #   page_start / page_end: int — actual PDF passage page span
    #   slide_number: int — original PowerPoint slide number
    #   section_title / original_filename: str|None


class GraphTriple(BaseModel):
    """A subject–predicate–object triple extracted from one or more chunks."""

    head: str
    relation: str
    tail: str
    source_chunk_ids: list[str]


class RetrievalResult(BaseModel):
    """Output of any retrieval module (dense or graph-based)."""

    chunks: list[Chunk]
    subgraph: list[GraphTriple]
    routing_mode: RoutingMode
    # Graph-retriever-only: stable keys of every node in the retrieved subgraph.
    # Empty for dense retrieval. Used by the API layer to build the `hops` list
    # the frontend highlights in the concept graph.
    subgraph_node_keys: list[str] = Field(default_factory=list)


class ReflectionVerdict(BaseModel):
    """Structured self-evaluation output from the Self-RAG reflector."""

    needs_retrieval: bool
    is_relevant: bool
    is_supported: bool
    is_useful: bool
    reasoning: str
    verification_status: Literal["verified", "skipped", "unavailable"] = "verified"


class CitationSpan(BaseModel):
    """Verified quote in one numbered passage; offsets address original chunk text."""

    chunk_id: str
    source_number: int = Field(ge=1)
    quote: str
    quote_start: int = Field(ge=0)
    quote_end: int = Field(ge=0)  # exclusive, Python Unicode character offsets


class GenerationResult(BaseModel):
    """Final output of the generation stage."""

    answer: str
    routing_mode: RoutingMode
    reflection: ReflectionVerdict
    evidence_chunk_ids: list[str]  # chunk ids cited in the answer
    citations: list[CitationSpan] = Field(default_factory=list)
