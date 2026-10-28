"""
Stage 1.3 — Subtopic mapping (per-chunk or per-section).

Given a chunk and the document's Topics, asks the LLM which Subtopics
the chunk introduces or discusses. Each Subtopic optionally carries a
parent Subtopic (for nesting) and always points back to one-or-more
parent Topics. A synonym-merge step deduplicates against existing
Subtopic nodes in the graph. A global cap limits subtopics per chunk
and deduplication removes repeats across chunks.

Public API:
    proposals = extract_subtopics(chunks, topic_names, config,
                                  existing_names=[], embedder=None)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, Field

from backend.graph._llm import SynonymResolver, safe_call_json
from backend.utils.jobs import ingestion_workers, map_bounded
from backend.graph.schema import SubtopicProposal, make_key
from backend.schemas import Chunk

if TYPE_CHECKING:
    from backend.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class _SubtopicPayload(BaseModel):
    name: str = Field(min_length=1)
    summary: str
    parent_topic: str
    parent_subtopic: str


class _SubtopicItems(BaseModel):
    items: list[_SubtopicPayload]


_PROMPT = """\
You are building a hierarchical knowledge graph for educational content.

Document topics: {topics}

Text chunk:
\"\"\"
{text}
\"\"\"

Task: Identify the Subtopics this chunk introduces or discusses. A Subtopic
is an intermediate layer BETWEEN the broad Topic and the leaf Content.
Examples: under Topic "Graph Theory", Subtopics could be "Graph Traversal",
"Shortest Paths", "Spectral Graph Theory".

Extract at most {max_per_chunk} subtopics. Focus on the most significant ones.

For each subtopic write a 2-3 sentence summary that explains what the subtopic
covers, its key ideas, and its relevance within the broader topic.

For math, chemistry, or physics notation, use LaTeX syntax: inline as
$x^2$, block as $$...$$. Use markdown for emphasis, lists, or code.

Return ONLY a JSON array. Each element must have keys:
  "name"           : short display name (1-6 words)
  "summary"        : 2-3 sentence description of what this subtopic covers (LaTeX/markdown allowed)
  "parent_topic"   : name of the parent Topic from the list above, or ""
  "parent_subtopic": name of another subtopic this one nests UNDER, or ""

If the chunk introduces no Subtopics, return: []

Example: [{{"name":"Shortest Paths","summary":"Algorithms for finding minimum-weight paths in graphs. Key methods include Dijkstra and Bellman-Ford, which differ in their handling of negative-weight edges.","parent_topic":"Graph Theory","parent_subtopic":""}}]\
"""


_MAX_TEXT_CHARS = 1800


def _extract_for_chunk(
    chunk: Chunk,
    topic_names: list[str],
    cfg: dict,
    existing_names: list[str],
    embedder: "Embedder | None",
    threshold: float,
    max_per_chunk: int,
    resolver: SynonymResolver,
) -> list[SubtopicProposal]:
    if not chunk.text.strip():
        return []

    prompt = _PROMPT.format(
        topics=", ".join(topic_names) if topic_names else "(none — infer from chunk)",
        text=chunk.text[:_MAX_TEXT_CHARS],
        max_per_chunk=max_per_chunk,
    )
    raw = safe_call_json(prompt, cfg, max_tokens=cfg.get("max_tokens", 600), response_model=_SubtopicItems)
    return subtopic_proposals_from_entries(raw, chunk, topic_names, max_per_chunk, resolver)


def subtopic_proposals_from_entries(
    raw: list[dict], chunk: Chunk, topic_names: list[str], max_per_chunk: int,
    resolver: SynonymResolver,
) -> list[SubtopicProposal]:
    """Shared parsing and exact chunk attribution for separate/combined extraction."""
    proposals: list[SubtopicProposal] = []
    for entry in raw[:max_per_chunk]:
        try:
            name = str(entry["name"]).strip()
            if not name:
                continue
            summary = str(entry.get("summary", "")).strip()
            parent_topic = str(entry.get("parent_topic", "")).strip()
            parent_subtopic = str(entry.get("parent_subtopic", "")).strip()
        except (KeyError, TypeError, ValueError):
            logger.debug(f"Skipping malformed subtopic entry: {entry}")
            continue

        proposals.append(
            SubtopicProposal(
                name=name,
                summary=summary,
                parent_topic_names=[parent_topic] if parent_topic else list(topic_names),
                parent_subtopic_names=[parent_subtopic] if parent_subtopic else [],
                source_chunk_ids=[chunk.id],
            )
        )
    for proposal, name in zip(proposals, resolver.resolve_many([proposal.name for proposal in proposals])):
        proposal.name = name
    return proposals


def extract_subtopics(
    chunks: list[Chunk],
    topic_names: list[str],
    config: dict,
    existing_names: list[str] | None = None,
    embedder: "Embedder | None" = None,
    *,
    on_completed: Callable[[int, int], None] | None = None,
) -> list[SubtopicProposal]:
    """
    Run Stage 1.3 across a batch of chunks for one document.

    Args:
        chunks:         All Chunks for the document.
        topic_names:    Display names of the document's Topics (Stage 1.2 output).
        config:         Parsed llm.yaml.
        existing_names: Display names of existing Subtopic nodes for synonym swap.
        embedder:       Optional Embedder for synonym matching.

    Returns:
        Flat, deduplicated list of SubtopicProposals across all chunks.
        Deduplication merges proposals with the same resolved name, unioning
        their source_chunk_ids (longest summary wins).
    """
    cfg = config.get("subtopic_extractor", {})
    threshold = float(cfg.get("merge_threshold", 0.82))
    max_per_chunk = int(cfg.get("max_per_chunk", 3))
    max_workers = ingestion_workers(cfg)

    _existing = existing_names or []
    resolver = SynonymResolver(_existing, embedder, threshold)

    def extract(chunk: Chunk) -> list[SubtopicProposal]:
        try:
            return _extract_for_chunk(chunk, topic_names, cfg, _existing, embedder,
                                      threshold, max_per_chunk, resolver)
        except Exception as exc:
            logger.warning("Subtopic extraction failed (%s)", type(exc).__name__)
            return []

    all_proposals = [proposal for batch in map_bounded(
        extract, chunks, max_workers=max_workers, on_completed=on_completed,
    ) for proposal in batch]

    # Deduplicate by resolved name — merge source_chunk_ids, keep longest summary
    merged: dict[str, SubtopicProposal] = {}
    for p in all_proposals:
        key = p.name.strip().lower()
        if key not in merged:
            merged[key] = p
        else:
            existing = merged[key]
            if len(p.summary) > len(existing.summary):
                existing.summary = p.summary
            for cid in p.source_chunk_ids:
                if cid not in existing.source_chunk_ids:
                    existing.source_chunk_ids.append(cid)

    result = list(merged.values())
    logger.info(
        f"Extracted {len(result)} unique subtopic(s) across {len(chunks)} chunks "
        f"(before dedup: {len(all_proposals)})"
    )
    return result
