"""
Stage 1.2 — Topic inference (document-level).

Given a short preview of a source document, asks the LLM for 1–5 Topics
the document is primarily about. Each Topic has a name, scope, and short
summary. A synonym-merge step swaps proposed names for existing graph
topic names when embedding similarity exceeds `merge_threshold`, so that
"Natural Language Processing" and "NLP" converge to one key.

Public API:
    proposals = extract_topics(doc_preview, source_chunk_ids, config,
                               existing_names=[], embedder=None)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.graph._llm import safe_call_json
from backend.graph.schema import TopicProposal

if TYPE_CHECKING:
    from backend.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


_PROMPT = """\
You are building a hierarchical knowledge graph for educational content.

Document preview:
\"\"\"
{preview}
\"\"\"

Task: Identify the 1-{max_topics} primary Topics this document is about. A Topic is
the broadest unit of knowledge — examples include "Organic Chemistry",
"Graph Theory", "Transformers Architecture", or a specific paper title
when the document IS that paper (e.g. "Attention Is All You Need").

For each Topic, also decide its scope:
  - "broad"  : covers a whole field or discipline
  - "narrow" : a focused work, paper, or tightly scoped subject

Be conservative — prefer fewer, more accurate topics over many vague ones.

For math, chemistry, or physics notation, use LaTeX syntax: inline as
$x^2$, block as $$...$$. Use markdown for emphasis, lists, or code.

Return ONLY a JSON array. Each element must have keys:
  "name"     : short display name (1-6 words)
  "scope"    : "broad" | "narrow"
  "summary"  : one-sentence description of the Topic (LaTeX/markdown allowed)

If no clear topic, return: []

Example: [{{"name":"Graph Theory","scope":"broad","summary":"Mathematical study of graphs and their properties."}}]\
"""


_MAX_PREVIEW_CHARS = 3000


def _max_topics_for_length(preview_len: int) -> int:
    """Scale max topics with document size: short docs get fewer topics."""
    if preview_len < 500:
        return 1
    if preview_len < 1500:
        return 2
    return 3


def _resolve_synonym(
    proposed_name: str,
    existing_names: list[str],
    embedder: "Embedder | None",
    threshold: float,
) -> str:
    """Swap the proposed name for the nearest existing name if similar enough."""
    if not existing_names or embedder is None:
        return proposed_name
    try:
        candidate = embedder.embed_one(proposed_name)
        pool = embedder.embed(existing_names)
        sims = pool @ candidate  # both L2-normalised
        best_idx = int(sims.argmax())
        if float(sims[best_idx]) >= threshold:
            return existing_names[best_idx]
    except Exception as exc:
        logger.debug(f"Embedding-based synonym lookup failed: {exc}")
    return proposed_name


def extract_topics(
    doc_preview: str,
    source_chunk_ids: list[str],
    config: dict,
    existing_names: list[str] | None = None,
    embedder: "Embedder | None" = None,
) -> list[TopicProposal]:
    """
    Run Stage 1.2 over a single document's preview text.

    Args:
        doc_preview:       Concatenated first ~3k chars of the document
                           (filename + TOC + head chunks).
        source_chunk_ids:  Chunk IDs from the document, recorded as provenance.
        config:            Parsed llm.yaml.
        existing_names:    Display names of Topic nodes already in the graph,
                           used for synonym-swap.
        embedder:          Optional Embedder for synonym matching.

    Returns:
        list[TopicProposal] — empty on LLM failure.
    """
    cfg = config.get("topic_extractor", {})
    threshold = float(cfg.get("merge_threshold", 0.85))
    preview = doc_preview[:_MAX_PREVIEW_CHARS]
    max_topics = int(cfg.get("max_topics", _max_topics_for_length(len(preview))))
    prompt = _PROMPT.format(preview=preview, max_topics=max_topics)

    raw = safe_call_json(prompt, cfg, max_tokens=cfg.get("max_tokens", 512))
    proposals: list[TopicProposal] = []
    for entry in raw[:max_topics]:
        try:
            name = str(entry["name"]).strip()
            if not name:
                continue
            scope = str(entry.get("scope", "broad")).strip().lower()
            if scope not in ("broad", "narrow"):
                scope = "broad"
            summary = str(entry.get("summary", "")).strip()
        except (KeyError, TypeError, ValueError):
            logger.debug(f"Skipping malformed topic entry: {entry}")
            continue

        name = _resolve_synonym(name, existing_names or [], embedder, threshold)
        proposals.append(
            TopicProposal(
                name=name,
                summary=summary,
                scope=scope,  # type: ignore[arg-type]
                source_chunk_ids=list(source_chunk_ids),
            )
        )

    logger.info(f"Extracted {len(proposals)} topic(s) from document preview")
    return proposals
