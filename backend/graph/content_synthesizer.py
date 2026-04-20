"""
Stage 1.4 — Content synthesis (per chunk).

Given a chunk plus the Topics/Subtopics it has been assigned to, asks the
LLM to extract the "teachable units" in the chunk: definitions, theorems,
techniques, worked examples, exam-style questions, figures. For each unit
the LLM produces a single rich summary and an illustration placeholder hint.

Public API:
    proposals = synthesize_contents(chunk, subtopic_names, topic_names,
                                    config, existing_titles=[], embedder=None)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.graph._llm import safe_call_json
from backend.graph.schema import ContentProposal, Illustration
from backend.schemas import Chunk

if TYPE_CHECKING:
    from backend.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


_CONTENT_TYPES = {"definition", "theorem", "technique", "example", "question", "figure", "other"}
_ILLUSTRATION_KINDS = {"diagram", "equation", "code", "image"}


_PROMPT = """\
You are building a hierarchical knowledge graph for educational content.

Document topics:     {topics}
Relevant subtopics:  {subtopics}

Text chunk:
\"\"\"
{text}
\"\"\"

Task: Extract the distinct "teachable units" present in the chunk. A teachable
unit is a self-contained concept card — for example a definition, a theorem,
a technique, a worked example, an exam-style question, or a labeled figure.
Extract at most {max_per_chunk} units. Focus on the most important ones.

For each unit produce a single clear summary (4-8 sentences). The summary
should be comprehensive enough to stand alone: explain what it is, why it
matters, and how it relates to the surrounding topics.

For math, chemistry, or physics notation, use LaTeX syntax: inline as
$x^2$, block as $$...$$. Use markdown for emphasis, lists, or code.

Return ONLY a JSON array. Each element must have keys:
  "title"             : 3-10 word title of the concept card
  "content_type"      : one of: definition, theorem, technique, example, question, figure, other
  "summary"           : comprehensive 4-8 sentence explanation (LaTeX/markdown allowed)
  "raw_excerpt"       : single most information-dense verbatim sentence or formula
                        copied from the chunk (≤300 chars). Use the chunk's exact
                        wording. May be empty if nothing stands out.
  "key_terms"         : array of 3-8 technical terms, variable names, or formula
                        names central to this unit (plain strings, no explanations)
  "parent_subtopic"   : best subtopic from the list above (or "")
  "illustration_kind" : diagram | equation | code | image | "" if none
  "illustration_hint" : 1-line description of what the illustration would show, or ""

Return [] if the chunk has no discrete teachable units.\
"""


_MAX_TEXT_CHARS = 2000
_MAX_EXCERPT_CHARS = 300


def _max_content_for_length(text: str) -> int:
    """
    Dynamic cap on content nodes per chunk based on source length.
    Longer chunks get more content units; minimum 3 for short chunks.
    """
    n = len(text)
    if n < 500:
        return 3
    if n < 1500:
        return 5
    if n < 3000:
        return 7
    return 10


def _parse_key_terms(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out


def _resolve_synonym(
    proposed_title: str,
    existing_titles: list[str],
    embedder: "Embedder | None",
    threshold: float,
) -> str:
    if not existing_titles or embedder is None:
        return proposed_title
    try:
        candidate = embedder.embed_one(proposed_title)
        pool = embedder.embed(existing_titles)
        sims = pool @ candidate
        best_idx = int(sims.argmax())
        if float(sims[best_idx]) >= threshold:
            return existing_titles[best_idx]
    except Exception as exc:
        logger.debug(f"Embedding-based synonym lookup failed: {exc}")
    return proposed_title


def _illustration_from(entry: dict) -> Illustration | None:
    kind = str(entry.get("illustration_kind", "")).strip().lower()
    hint = str(entry.get("illustration_hint", "")).strip()
    if kind not in _ILLUSTRATION_KINDS or not hint:
        return None
    return Illustration(kind=kind, hint=hint)  # type: ignore[arg-type]


def synthesize_contents(
    chunk: Chunk,
    subtopic_names: list[str],
    topic_names: list[str],
    config: dict,
    existing_titles: list[str] | None = None,
    embedder: "Embedder | None" = None,
) -> list[ContentProposal]:
    """
    Run Stage 1.4 over one chunk.

    Args:
        chunk:           The chunk to mine for teachable units.
        subtopic_names:  Display names of subtopics that the chunk has been
                         mapped to in Stage 1.3.
        topic_names:     Display names of the document's Topics.
        config:          Parsed llm.yaml.
        existing_titles: Titles of existing Content nodes, for synonym-swap.
        embedder:        Optional Embedder for synonym matching.

    Returns:
        list[ContentProposal] — empty if the chunk has no teachable units.
    """
    cfg = config.get("content_synthesizer", {})
    threshold = float(cfg.get("merge_threshold", 0.88))
    override = cfg.get("max_per_chunk")
    max_per_chunk = int(override) if override is not None else _max_content_for_length(chunk.text)

    if not chunk.text.strip():
        return []

    prompt = _PROMPT.format(
        topics=", ".join(topic_names) if topic_names else "(none)",
        subtopics=", ".join(subtopic_names) if subtopic_names else "(none)",
        text=chunk.text[:_MAX_TEXT_CHARS],
        max_per_chunk=max_per_chunk,
    )
    raw = safe_call_json(prompt, cfg, max_tokens=cfg.get("max_tokens", 2000))

    proposals: list[ContentProposal] = []
    for entry in raw[:max_per_chunk]:
        try:
            title = str(entry["title"]).strip()
            if not title:
                continue
            content_type = str(entry.get("content_type", "other")).strip().lower()
            if content_type not in _CONTENT_TYPES:
                content_type = "other"
            summary = str(entry.get("summary", "")).strip()
            raw_excerpt = str(entry.get("raw_excerpt", "")).strip()
            if len(raw_excerpt) > _MAX_EXCERPT_CHARS:
                raw_excerpt = raw_excerpt[:_MAX_EXCERPT_CHARS].rstrip()
            key_terms = _parse_key_terms(entry.get("key_terms"))
            parent_subtopic = str(entry.get("parent_subtopic", "")).strip()
        except (KeyError, TypeError, ValueError):
            logger.debug(f"Skipping malformed content entry: {entry}")
            continue

        title = _resolve_synonym(title, existing_titles or [], embedder, threshold)

        proposals.append(
            ContentProposal(
                title=title,
                content_type=content_type,  # type: ignore[arg-type]
                summary=summary,
                raw_excerpt=raw_excerpt,
                key_terms=key_terms,
                illustration=_illustration_from(entry),
                parent_subtopic_names=[parent_subtopic] if parent_subtopic else list(subtopic_names),
                parent_topic_names=list(topic_names),
                evidence_chunk_ids=[chunk.id],
            )
        )

    logger.debug(f"[{chunk.id}] synthesized {len(proposals)} content unit(s)")
    return proposals
