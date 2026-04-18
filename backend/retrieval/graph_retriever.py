"""
Stage 4.2 - Graph Retrieval: graph_retriever.py

Relevance-guided BFS over the concept knowledge graph.

Algorithm:
  1. Find anchor nodes — string match first, embedding similarity fallback.
  2. BFS loop (up to max_hops):
       a. Batch-embed all unvisited neighbour names for this hop.
       b. Score each neighbour: cosine(query_emb, neighbour_name_emb).
       c. Keep top max_nodes_per_hop neighbours that exceed relevance_threshold.
       d. Add kept neighbours to subgraph; enqueue for next hop.
  3. Collect all source_chunk_ids from subgraph nodes.
  4. Fetch Chunk objects from DocumentStore.
  5. Return RetrievalResult.

Public API:
    result = retrieve_graph(query, builder, store, embedder, config)
    # -> RetrievalResult(chunks=[...], subgraph=[...], routing_mode="graph")
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval.embedder import Embedder
from backend.schemas import Chunk, GraphTriple, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node embedding cache (populated once at startup via warm_up())
# ---------------------------------------------------------------------------

_node_emb_cache: dict[str, np.ndarray] = {}


def warm_up(graph: nx.MultiDiGraph, embedder: Embedder) -> None:
    """
    Pre-compute L2-normalised embeddings for every node display name.
    Call once at server startup after the graph is loaded.
    Subsequent graph queries skip per-query embedding of node names.
    """
    global _node_emb_cache
    if graph.number_of_nodes() == 0:
        return
    keys = list(graph.nodes())
    names = [graph.nodes[k].get("name", k) for k in keys]
    embs = embedder.embed(names)           # (N, D) — single batched call
    _node_emb_cache = dict(zip(keys, embs))
    logger.info(f"Node embedding cache: {len(_node_emb_cache)} nodes precomputed")


# ---------------------------------------------------------------------------
# Step 1: Anchor node finding
# ---------------------------------------------------------------------------

def _find_anchors(
    query: str,
    graph: nx.MultiDiGraph,
    embedder: Embedder,
    top_k: int,
) -> list[str]:
    """
    Return up to top_k node keys that best match the query.

    Strategy:
      0. Cluster scoping — if topic-cluster nodes exist, score them against
         the query first, then restrict anchor search to members of the
         best-matching cluster(s). This prevents cross-domain contamination
         (e.g. chemistry nodes appearing for NLP queries).
      1. String match   — node display name appears (case-insensitive) in query.
      2. Embedding gap  — fill remaining slots by cosine similarity between
                          the query embedding and each node-name embedding.
    """
    if graph.number_of_nodes() == 0:
        return []

    q_lower = query.lower()
    all_keys = list(graph.nodes())
    q_emb = embedder.embed_one(query)  # computed once; reused for cluster + anchor scoring

    # --- Step 0: cluster-aware candidate scoping ---
    cluster_keys = [
        k for k in all_keys
        if graph.nodes[k].get("node_type") == "cluster"
    ]
    candidate_pool = all_keys  # default: global search

    if cluster_keys:
        if _node_emb_cache:
            cl_embs = np.stack([_node_emb_cache[k] for k in cluster_keys])
        else:
            cl_names = [graph.nodes[k].get("name", k) for k in cluster_keys]
            cl_embs = embedder.embed(cl_names)
        cl_scores = (cl_embs @ q_emb).tolist()

        # Always include the top cluster; add a second if it exceeds 0.25
        ranked = sorted(zip(cl_scores, cluster_keys), key=lambda x: x[0], reverse=True)
        top_clusters: set[str] = {ranked[0][1]}
        if len(ranked) > 1 and ranked[1][0] > 0.25:
            top_clusters.add(ranked[1][1])

        logger.debug(
            f"Top clusters for '{query[:50]}': "
            f"{[(graph.nodes[k].get('name', k), round(s, 3)) for s, k in ranked[:3]]}"
        )

        # Members = nodes with entity → BELONGS_TO_TOPIC → cluster
        members: set[str] = set(top_clusters)
        for k in all_keys:
            for _, v, edata in graph.out_edges(k, data=True):
                if v in top_clusters and edata.get("relation") == "BELONGS_TO_TOPIC":
                    members.add(k)

        if len(members) > len(top_clusters):  # found real concept members
            candidate_pool = list(members)

    # --- Step 1: string match within candidate pool ---
    string_hits = [
        k for k in candidate_pool
        if graph.nodes[k].get("name", k).lower() in q_lower
    ]

    if len(string_hits) >= top_k:
        return string_hits[:top_k]

    # --- Step 2: embedding similarity within candidate pool ---
    remaining = [k for k in candidate_pool if k not in set(string_hits)]

    # If cluster scoping left too few candidates, open up the full graph as fallback
    if (len(remaining) + len(string_hits)) < top_k and candidate_pool is not all_keys:
        extra = [
            k for k in all_keys
            if k not in set(candidate_pool) and k not in set(string_hits)
        ]
        remaining = remaining + extra

    if not remaining:
        return string_hits

    if _node_emb_cache:
        name_embs = np.stack([_node_emb_cache[k] for k in remaining])
    else:
        names = [graph.nodes[k].get("name", k) for k in remaining]
        name_embs = embedder.embed(names)
    scores = name_embs @ q_emb

    n_needed = top_k - len(string_hits)
    top_idx = np.argsort(scores)[::-1][:n_needed]
    embedding_hits = [remaining[i] for i in top_idx]

    return string_hits + embedding_hits


# ---------------------------------------------------------------------------
# Step 2: Relevance-guided BFS
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
    """
    Expand the subgraph from anchors using relevance-gated BFS.

    Traversal is bidirectional (follows both out-edges and in-edges) so that
    BFS can expand from a topic cluster node down to its member entities, and
    vice-versa. BELONGS_TO_TOPIC structural edges are traversed for navigation
    but excluded from the returned subgraph_edges so they don't appear in the
    user-facing evidence.

    Returns:
        subgraph_nodes: set of node keys in the retrieved subgraph
        subgraph_edges: list of (source, target, edge_data) — content edges only
    """
    visited: set[str] = set(anchors)
    frontier: list[str] = list(anchors)
    subgraph_nodes: set[str] = set(anchors)
    # Each entry: (actual_source_key, actual_target_key, edge_data)
    subgraph_edges: list[tuple[str, str, dict]] = []

    for hop in range(max_hops):
        # Collect unvisited neighbours via both out-edges and in-edges
        # candidate_map[candidate_key] = [(src, tgt, edge_data), ...]
        candidate_map: dict[str, list[tuple[str, str, dict]]] = {}

        for node_key in frontier:
            for _, v, edge_data in graph.out_edges(node_key, data=True):
                if v not in visited:
                    candidate_map.setdefault(v, []).append((node_key, v, edge_data))
            for u, _, edge_data in graph.in_edges(node_key, data=True):
                if u not in visited:
                    candidate_map.setdefault(u, []).append((u, node_key, edge_data))

        if not candidate_map:
            break

        # Batch-embed all candidate names
        cand_keys = list(candidate_map.keys())
        if _node_emb_cache:
            cand_embs = np.stack([_node_emb_cache[k] for k in cand_keys])
        else:
            cand_names = [graph.nodes[k].get("name", k) for k in cand_keys]
            cand_embs = embedder.embed(cand_names)
        scores = (cand_embs @ query_emb).tolist()

        ranked = sorted(zip(scores, cand_keys), key=lambda x: x[0], reverse=True)

        next_frontier: list[str] = []
        for score, v in ranked[:max_per_hop]:
            if score < threshold:
                break
            visited.add(v)
            subgraph_nodes.add(v)
            # Record content edges (skip structural BELONGS_TO_TOPIC edges)
            for src, tgt, edge_data in candidate_map[v]:
                if edge_data.get("relation") != "BELONGS_TO_TOPIC":
                    subgraph_edges.append((src, tgt, edge_data))
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
    """
    Run relevance-guided graph BFS and return a RetrievalResult.

    Args:
        query:   Raw user question.
        builder: GraphBuilder containing the concept knowledge graph.
        store:   DocumentStore for fetching full Chunk objects.
        embedder: Shared Embedder instance.
        config:  Parsed config/llm.yaml.

    Returns:
        RetrievalResult with routing_mode="graph", subgraph triples,
        and the Chunks associated with subgraph nodes.
    """
    cfg = config.get("graph_retriever", {})
    max_hops = cfg.get("max_hops", 3)
    threshold = cfg.get("relevance_threshold", 0.4)
    max_per_hop = cfg.get("max_nodes_per_hop", 10)
    top_k_anchors = cfg.get("top_k_anchors", 3)

    graph = builder.graph

    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty — returning empty RetrievalResult.")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="graph")

    # Step 1: Anchors
    anchors = _find_anchors(query, graph, embedder, top_k_anchors)
    if not anchors:
        logger.warning(f"No anchor nodes found for query: '{query[:60]}'")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="graph")

    logger.debug(
        f"Anchors: {[graph.nodes[k].get('name', k) for k in anchors]}"
    )

    # Step 2: BFS
    q_emb = embedder.embed_one(query)
    subgraph_nodes, subgraph_edges = _bfs(
        anchors, graph, q_emb, embedder, max_hops, threshold, max_per_hop
    )

    # Step 3: Build GraphTriple list from collected edges
    triples: list[GraphTriple] = [
        GraphTriple(
            head=graph.nodes[u].get("name", u),
            relation=edge_data.get("relation", ""),
            tail=graph.nodes[v].get("name", v),
            source_chunk_ids=edge_data.get("source_chunk_ids", []),
        )
        for u, v, edge_data in subgraph_edges
    ]

    # Step 4: Fetch chunks from concept nodes only (cluster hub nodes have no content)
    all_chunk_ids: set[str] = set()
    for node_key in subgraph_nodes:
        if graph.nodes[node_key].get("node_type") != "cluster":
            all_chunk_ids.update(graph.nodes[node_key].get("source_ids", []))

    chunks: list[Chunk] = [
        c for cid in all_chunk_ids if (c := store.get_chunk(cid))
    ]

    logger.info(
        f"Graph retrieval: {len(subgraph_nodes)} nodes, "
        f"{len(triples)} edges, {len(chunks)} chunks"
    )
    return RetrievalResult(chunks=chunks, subgraph=triples, routing_mode="graph")
