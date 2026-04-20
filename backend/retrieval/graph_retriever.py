"""
Stage 4.2 — Graph Retrieval: graph_retriever.py

Relevance-guided BFS over the hierarchical Topic → Subtopic → Content graph.

Algorithm:
  1. Topic scoping — score Topic nodes against the query; restrict anchor
     search to descendants reachable via HAS_SUBTOPIC / HAS_CONTENT.
  2. Find anchor nodes within the scoped pool — string match first,
     embedding similarity fallback.
  3. Relevance-guided BFS (up to max_hops):
       - traverses HAS_SUBTOPIC, HAS_CONTENT, and RELATED_TO edges
       - skips chunk_ref nodes (leaves) and EVIDENCE_OF edges during navigation
       - keeps top max_nodes_per_hop neighbours above relevance_threshold
  4. Collect source_chunk_ids from every non-chunk_ref node in the subgraph.
  5. Fetch Chunk objects from DocumentStore.
  6. Return RetrievalResult.

Public API:
    result = retrieve_graph(query, builder, store, embedder, config)
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval.embedder import Embedder
from backend.schemas import Chunk, GraphTriple, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node embedding cache (populated at startup / after graph mutations)
# ---------------------------------------------------------------------------

_node_emb_cache: dict[str, np.ndarray] = {}

# Edges that act as true leaves — do not expand through them and exclude them
# from the user-facing subgraph (they point from ChunkRef → Content).
_LEAF_RELATIONS = frozenset({"EVIDENCE_OF"})


def warm_up(graph: nx.MultiDiGraph, embedder: Embedder) -> None:
    """
    Pre-compute L2-normalised embeddings for every Topic/Subtopic/Content
    node. Chunk-ref nodes are skipped — they are never scored against queries.
    """
    global _node_emb_cache
    if graph.number_of_nodes() == 0:
        _node_emb_cache = {}
        return
    keys: list[str] = []
    texts: list[str] = []
    for k, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "chunk_ref":
            continue
        keys.append(k)
        texts.append(_node_text(attrs, k))
    if not keys:
        _node_emb_cache = {}
        return
    embs = embedder.embed(texts)
    _node_emb_cache = dict(zip(keys, embs))
    logger.info(f"Node embedding cache: {len(_node_emb_cache)} nodes precomputed")


def _node_text(attrs: dict, key: str) -> str:
    """Text used to embed a node for anchor scoring. Uses summary when available."""
    name = attrs.get("name", key)
    ntype = attrs.get("node_type", "")
    if ntype == "content":
        summary = attrs.get("summary_intermediate", "") or attrs.get("summary_beginner", "")
    else:
        summary = attrs.get("summary", "")
    return f"{name}. {summary}".strip()


def _embed_keys(keys: list[str], graph: nx.MultiDiGraph, embedder: Embedder) -> np.ndarray:
    """Batch-fetch embeddings from cache or compute fresh if missing."""
    if _node_emb_cache and all(k in _node_emb_cache for k in keys):
        return np.stack([_node_emb_cache[k] for k in keys])
    texts = [_node_text(graph.nodes[k], k) for k in keys]
    return embedder.embed(texts)


# ---------------------------------------------------------------------------
# Topic scoping + anchor finding
# ---------------------------------------------------------------------------

def _topic_descendants(graph: nx.MultiDiGraph, topic_keys: set[str]) -> set[str]:
    """
    Return all nodes reachable from the given topic keys via HAS_SUBTOPIC
    and HAS_CONTENT edges (excluding chunk_ref leaves).
    """
    descendants: set[str] = set(topic_keys)
    frontier = list(topic_keys)
    navigable = {"HAS_SUBTOPIC", "HAS_CONTENT"}
    while frontier:
        next_frontier: list[str] = []
        for node_key in frontier:
            for _, v, edata in graph.out_edges(node_key, data=True):
                if edata.get("relation") in navigable and v not in descendants:
                    if graph.nodes[v].get("node_type") == "chunk_ref":
                        continue
                    descendants.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
    return descendants


def _key_term_hit(attrs: dict, query_lower: str) -> bool:
    """True iff any of the node's key_terms appears (case-insensitive) in the query."""
    terms = attrs.get("key_terms") or []
    for t in terms:
        if isinstance(t, str):
            ts = t.strip().lower()
            if ts and ts in query_lower:
                return True
    return False


def _find_anchors(
    query: str,
    graph: nx.MultiDiGraph,
    embedder: Embedder,
    top_k: int,
    key_term_boost: float = 0.15,
) -> list[str]:
    if graph.number_of_nodes() == 0:
        return []

    q_lower = query.lower()
    all_keys = [
        k for k, a in graph.nodes(data=True)
        if a.get("node_type") != "chunk_ref"
    ]
    if not all_keys:
        return []

    q_emb = embedder.embed_one(query)

    # --- Topic scoping ---
    topic_keys = [k for k in all_keys if graph.nodes[k].get("node_type") == "topic"]
    candidate_pool = all_keys

    if topic_keys:
        topic_embs = _embed_keys(topic_keys, graph, embedder)
        topic_scores = (topic_embs @ q_emb).tolist()
        ranked = sorted(zip(topic_scores, topic_keys), key=lambda x: x[0], reverse=True)
        top_topics: set[str] = {ranked[0][1]}
        if len(ranked) > 1 and ranked[1][0] > 0.25:
            top_topics.add(ranked[1][1])

        logger.debug(
            f"Top topics for '{query[:50]}': "
            f"{[(graph.nodes[k].get('name', k), round(s, 3)) for s, k in ranked[:3]]}"
        )

        members = _topic_descendants(graph, top_topics)
        if len(members) > len(top_topics):
            candidate_pool = [k for k in all_keys if k in members]

    # --- String matches within candidate pool (name OR key_terms) ---
    string_hits = [
        k for k in candidate_pool
        if graph.nodes[k].get("name", k).lower() in q_lower
        or _key_term_hit(graph.nodes[k], q_lower)
    ]
    if len(string_hits) >= top_k:
        return string_hits[:top_k]

    # --- Embedding fallback (with key_terms boost) ---
    remaining = [k for k in candidate_pool if k not in set(string_hits)]
    if (len(remaining) + len(string_hits)) < top_k and candidate_pool is not all_keys:
        extra = [
            k for k in all_keys
            if k not in set(candidate_pool) and k not in set(string_hits)
        ]
        remaining = remaining + extra

    if not remaining:
        return string_hits

    name_embs = _embed_keys(remaining, graph, embedder)
    scores = name_embs @ q_emb
    if key_term_boost > 0:
        boosts = np.array(
            [
                key_term_boost if _key_term_hit(graph.nodes[k], q_lower) else 0.0
                for k in remaining
            ],
            dtype=scores.dtype,
        )
        scores = scores + boosts
    n_needed = top_k - len(string_hits)
    top_idx = np.argsort(scores)[::-1][:n_needed]
    return string_hits + [remaining[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Relevance-guided BFS
# ---------------------------------------------------------------------------

def _bfs(
    anchors: list[str],
    graph: nx.MultiDiGraph,
    query_emb: np.ndarray,
    embedder: Embedder,
    max_hops: int,
    threshold: float,
    max_per_hop: int,
) -> tuple[set[str], list[tuple[str, str, dict]]]:
    visited: set[str] = set(anchors)
    frontier: list[str] = list(anchors)
    subgraph_nodes: set[str] = set(anchors)
    subgraph_edges: list[tuple[str, str, dict]] = []

    for hop in range(max_hops):
        candidate_map: dict[str, list[tuple[str, str, dict]]] = {}

        for node_key in frontier:
            for _, v, edata in graph.out_edges(node_key, data=True):
                if v in visited:
                    continue
                if edata.get("relation") in _LEAF_RELATIONS:
                    continue
                if graph.nodes[v].get("node_type") == "chunk_ref":
                    continue
                candidate_map.setdefault(v, []).append((node_key, v, edata))
            for u, _, edata in graph.in_edges(node_key, data=True):
                if u in visited:
                    continue
                if edata.get("relation") in _LEAF_RELATIONS:
                    continue
                if graph.nodes[u].get("node_type") == "chunk_ref":
                    continue
                candidate_map.setdefault(u, []).append((u, node_key, edata))

        if not candidate_map:
            break

        cand_keys = list(candidate_map.keys())
        cand_embs = _embed_keys(cand_keys, graph, embedder)
        scores = (cand_embs @ query_emb).tolist()

        ranked = sorted(zip(scores, cand_keys), key=lambda x: x[0], reverse=True)
        next_frontier: list[str] = []
        for score, v in ranked[:max_per_hop]:
            if score < threshold:
                break
            visited.add(v)
            subgraph_nodes.add(v)
            for src, tgt, edata in candidate_map[v]:
                subgraph_edges.append((src, tgt, edata))
            next_frontier.append(v)
            logger.debug(
                f"  Hop {hop + 1}: '{graph.nodes[v].get('name', v)}' "
                f"(score={score:.3f})"
            )

        if not next_frontier:
            break
        frontier = next_frontier

    return subgraph_nodes, subgraph_edges


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------

def retrieve_graph(
    query: str,
    builder: GraphBuilder,
    store: DocumentStore,
    embedder: Embedder,
    config: dict,
) -> RetrievalResult:
    cfg = config.get("graph_retriever", {})
    max_hops = int(cfg.get("max_hops", 3))
    threshold = float(cfg.get("relevance_threshold", 0.4))
    max_per_hop = int(cfg.get("max_nodes_per_hop", 10))
    top_k_anchors = int(cfg.get("top_k_anchors", 3))
    key_term_boost = float(cfg.get("key_term_boost", 0.15))

    graph = builder.graph

    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty — returning empty RetrievalResult.")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="graph")

    anchors = _find_anchors(query, graph, embedder, top_k_anchors, key_term_boost)
    if not anchors:
        logger.warning(f"No anchor nodes found for query: '{query[:60]}'")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="graph")

    logger.debug(
        f"Anchors: {[graph.nodes[k].get('name', k) for k in anchors]}"
    )

    q_emb = embedder.embed_one(query)
    subgraph_nodes, subgraph_edges = _bfs(
        anchors, graph, q_emb, embedder, max_hops, threshold, max_per_hop
    )

    triples: list[GraphTriple] = [
        GraphTriple(
            head=graph.nodes[u].get("name", u),
            relation=edata.get("relation", ""),
            tail=graph.nodes[v].get("name", v),
            source_chunk_ids=edata.get("source_chunk_ids", []),
        )
        for u, v, edata in subgraph_edges
    ]

    # Collect chunk IDs from every subgraph node's source_ids attribute.
    all_chunk_ids: set[str] = set()
    for node_key in subgraph_nodes:
        attrs = graph.nodes[node_key]
        if attrs.get("node_type") == "chunk_ref":
            all_chunk_ids.add(node_key)
        else:
            all_chunk_ids.update(attrs.get("source_ids", []))

    chunks: list[Chunk] = [
        c for cid in all_chunk_ids if (c := store.get_chunk(cid))
    ]

    logger.info(
        f"Graph retrieval: {len(subgraph_nodes)} nodes, "
        f"{len(triples)} edges, {len(chunks)} chunks"
    )
    return RetrievalResult(
        chunks=chunks,
        subgraph=triples,
        routing_mode="graph",
        subgraph_node_keys=list(subgraph_nodes),
    )
