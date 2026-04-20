"""
Stage 1.5 — Semantic linking across hierarchy nodes.

Given a set of keyed (Topic/Subtopic/Content) nodes, finds pairs whose
embeddings are related-but-not-duplicate (cosine in [sim_lo, sim_hi])
and asks the LLM to label each surviving pair with a short free-form
relation verb. Produces RELATED_TO edges that wire cross-tree links
(e.g. Graph Theory ↔ Transformer Attention under "used_in").

Public API:
    proposals = link_nodes(linkables, embedder, config)
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from backend.graph._llm import call_llm, parse_json_list
from backend.graph.schema import RelatedToProposal

if TYPE_CHECKING:
    from backend.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class LinkableNode(TypedDict):
    key: str
    name: str
    text: str       # text used for embedding (usually name + summary)
    node_type: str  # "topic" | "subtopic" | "content"


_PROMPT = """\
You are building a hierarchical knowledge graph for educational content.

Below are candidate pairs of related nodes. For each pair, decide whether
the two concepts are meaningfully related (not duplicates, but linked) and,
if so, suggest a short free-form relation label (a verb or short verb phrase,
1-4 words, lowercase, e.g. "uses", "generalises", "prerequisite for",
"counterexample of", "used in").

Pairs (index: A  ↔  B):
{pairs_block}

Return ONLY a JSON array. For each related pair include:
  "index"      : the pair index above
  "label"      : short verb phrase
  "confidence" : 0.0-1.0

Skip pairs that are unrelated or are near-duplicates. Return [] if none.\
"""


def _build_pairs_block(pairs: list[tuple[int, LinkableNode, LinkableNode]]) -> str:
    lines = []
    for idx, a, b in pairs:
        lines.append(
            f"{idx}: [{a['node_type']}] {a['name']}  ↔  [{b['node_type']}] {b['name']}"
        )
    return "\n".join(lines)


def _candidate_pairs(
    nodes: list[LinkableNode],
    embedder: "Embedder",
    sim_lo: float,
    sim_hi: float,
    max_pairs: int,
) -> list[tuple[int, LinkableNode, LinkableNode, float]]:
    """Pre-filter pairs by embedding cosine. Returns list of (pair_idx, a, b, sim)."""
    if len(nodes) < 2:
        return []

    texts = [n["text"] or n["name"] for n in nodes]
    vectors = embedder.embed(texts)
    # Cosine = inner product because Embedder L2-normalises.
    sim_matrix = vectors @ vectors.T

    candidates: list[tuple[int, LinkableNode, LinkableNode, float]] = []
    for i, j in itertools.combinations(range(len(nodes)), 2):
        sim = float(sim_matrix[i, j])
        if sim_lo <= sim <= sim_hi:
            candidates.append((len(candidates), nodes[i], nodes[j], sim))

    # Sort by similarity descending; the highest-signal pairs go to the LLM first.
    candidates.sort(key=lambda t: t[3], reverse=True)
    # Reindex (indices must be contiguous 0..N for the prompt/response mapping).
    return [(new_idx, a, b, s) for new_idx, (_, a, b, s) in enumerate(candidates[:max_pairs])]


def link_nodes(
    nodes: list[LinkableNode],
    embedder: "Embedder",
    config: dict,
) -> list[RelatedToProposal]:
    """
    Run Stage 1.5 — semantic linking — across a set of linkable nodes.

    Args:
        nodes:    List of LinkableNode dicts (topic/subtopic/content nodes
                  with their already-keyed graph id, display name, and a
                  short text for embedding).
        embedder: Required — used for the embedding pre-filter.
        config:   Parsed llm.yaml. Reads the "semantic_linker" section.

    Returns:
        list[RelatedToProposal] — empty on LLM failure.
    """
    cfg = config.get("semantic_linker", {})
    sim_lo = float(cfg.get("sim_lo", 0.55))
    sim_hi = float(cfg.get("sim_hi", 0.90))
    max_pairs = int(cfg.get("max_candidate_pairs", 80))
    batch_size = int(cfg.get("batch_size", 10))
    min_confidence = float(cfg.get("min_confidence", 0.3))

    candidates = _candidate_pairs(nodes, embedder, sim_lo, sim_hi, max_pairs)
    if not candidates:
        logger.info("Semantic linking: no candidate pairs in [%.2f, %.2f]", sim_lo, sim_hi)
        return []

    proposals: list[RelatedToProposal] = []
    # Batch pairs to keep prompt length bounded.
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        # Re-index batch 0..len(batch)-1 so the LLM indices map cleanly.
        indexed = [(i, a, b) for i, (_, a, b, _) in enumerate(batch)]
        prompt = _PROMPT.format(pairs_block=_build_pairs_block(indexed))
        try:
            raw = call_llm(prompt, cfg, max_tokens=cfg.get("max_tokens", 512))
        except Exception as exc:
            logger.warning(f"Semantic linker LLM call failed on batch: {exc}")
            continue
        try:
            parsed = parse_json_list(raw)
        except json.JSONDecodeError:
            parsed = []

        for entry in parsed:
            try:
                idx = int(entry["index"])
                label = str(entry.get("label", "")).strip()
                confidence = float(entry.get("confidence", 0.5))
            except (KeyError, TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(batch):
                continue
            if not label:
                continue
            if confidence < min_confidence:
                continue
            _, a, b, sim = batch[idx]
            proposals.append(
                RelatedToProposal(
                    source_key=a["key"],
                    target_key=b["key"],
                    label=label,
                    confidence=confidence,
                    source_chunk_ids=[],
                )
            )
            logger.debug(
                f"RELATED_TO proposed: {a['name']} -[{label}]-> {b['name']} "
                f"(sim={sim:.2f}, conf={confidence:.2f})"
            )

    logger.info(
        f"Semantic linker proposed {len(proposals)} edge(s) from "
        f"{len(candidates)} candidate pair(s)"
    )
    return proposals


def build_linkables_from_graph(graph, node_types: list[str] | None = None) -> list[LinkableNode]:
    """
    Helper: extract LinkableNode entries from a live MultiDiGraph for use
    with link_nodes(). Excludes chunk_ref nodes by default.
    """
    allowed = set(node_types or ["topic", "subtopic", "content"])
    linkables: list[LinkableNode] = []
    for key, attrs in graph.nodes(data=True):
        ntype = attrs.get("node_type", "content")
        if ntype not in allowed:
            continue
        name = attrs.get("name", key)
        if ntype == "content":
            text = f"{name}. {attrs.get('summary_intermediate', '')}".strip()
        else:
            text = f"{name}. {attrs.get('summary', '')}".strip()
        linkables.append({
            "key": key,
            "name": name,
            "text": text,
            "node_type": ntype,
        })
    return linkables


# Silence unused-import complaint — numpy is used implicitly through Embedder.
_ = np
