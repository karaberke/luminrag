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
import threading
from weakref import WeakKeyDictionary

import networkx as nx
import numpy as np

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval.embedder import Embedder
from backend.schemas import Chunk, GraphTriple, RetrievalResult
from backend.utils.query_intent import multi_document_source_limit
from backend.utils.resources import reserve_model_resource
from backend.utils.text import node_text as _node_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node embedding cache (populated at startup / after graph mutations)
# ---------------------------------------------------------------------------

# Caches follow each graph object's lifetime. Identical node keys in different
# workspaces never share embeddings or text. The one lock only gates inference.
_graph_caches: WeakKeyDictionary = WeakKeyDictionary()
_node_emb_lock = threading.RLock()


def reserve_node_embeddings():
    return reserve_model_resource(_node_emb_lock)


def clear_node_embeddings(graph: nx.MultiDiGraph | None = None) -> None:
    with reserve_node_embeddings():
        if graph is None:
            _graph_caches.clear()
        else:
            _graph_caches.pop(graph, None)


def copy_node_embeddings(source: nx.MultiDiGraph, target: nx.MultiDiGraph) -> None:
    """Carry valid vectors into an explicit graph clone without sharing state.

    Node IDs alone never authorize reuse: both graphs must still match the
    cached text, and subsequent reads continue checking the embedder identity.
    """
    # Some callers clone immediately after committing storage. An optional
    # cache optimization must never introduce a retryable post-commit failure.
    if not _node_emb_lock.acquire(blocking=False):
        return
    try:
        cache = _graph_caches.get(source)
        if cache is None:
            return
        embedder, vectors, cached_texts = cache
        copied_vectors, copied_texts = {}, {}
        for key, vector in vectors.items():
            if key not in source or key not in target:
                continue
            text = cached_texts.get(key)
            if text != _node_text(source.nodes[key], key) or text != _node_text(target.nodes[key], key):
                continue
            copied = vector.copy()
            copied.flags.writeable = False
            copied_vectors[key] = copied
            copied_texts[key] = text
        _graph_caches[target] = (embedder, copied_vectors, copied_texts)
    finally:
        _node_emb_lock.release()


_LEAF_RELATIONS = frozenset({"EVIDENCE_OF"})


def warm_up(graph: nx.MultiDiGraph, embedder: Embedder) -> None:
    with reserve_node_embeddings():
        keys = [key for key, attrs in graph.nodes(data=True) if attrs.get("node_type") != "chunk_ref"]
        cache = _graph_caches.get(graph)
        if cache:
            current = set(keys)
            for key in set(cache[1]) - current:
                cache[1].pop(key, None)
                cache[2].pop(key, None)
        if keys:
            _get_node_embeddings(keys, graph, embedder)


def _embed_keys(keys: list[str], graph: nx.MultiDiGraph, embedder: Embedder) -> np.ndarray:
    return get_node_embeddings(keys, graph, embedder)


def get_node_embeddings(keys: list[str], graph: nx.MultiDiGraph, embedder: Embedder) -> np.ndarray:
    with reserve_node_embeddings():
        return _get_node_embeddings(keys, graph, embedder)


def _get_node_embeddings(keys: list[str], graph: nx.MultiDiGraph, embedder: Embedder) -> np.ndarray:
    cache = _graph_caches.get(graph)
    if cache is None or cache[0] is not embedder:
        cache = (embedder, {}, {})
        _graph_caches[graph] = cache
    _, vectors, cached_texts = cache
    if not keys:
        return np.empty((0, getattr(embedder, "dimension", 0)), dtype=np.float32)
    texts = {key: _node_text(graph.nodes[key], key) for key in keys}
    missing = [key for key in dict.fromkeys(keys) if key not in vectors or cached_texts.get(key) != texts[key]]
    if missing:
        embeddings = embedder.embed([texts[key] for key in missing])
        for key, embedding in zip(missing, embeddings):
            vectors[key] = embedding
            cached_texts[key] = texts[key]
    return np.stack([vectors[key] for key in keys])


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
    relevance_threshold: float = 0.4,
    query_emb: np.ndarray | None = None,
) -> list[str]:
    if graph.number_of_nodes() == 0 or top_k <= 0:
        return []

    q_lower = query.lower()
    all_keys = [
        k for k, a in graph.nodes(data=True)
        if a.get("node_type") != "chunk_ref"
    ]
    if not all_keys:
        return []

    q_emb = query_emb if query_emb is not None else getattr(embedder, "embed_query", embedder.embed_one)(query)

    # --- Topic scoping ---
    topic_keys = [k for k in all_keys if graph.nodes[k].get("node_type") == "topic"]
    candidate_pool = all_keys

    if topic_keys:
        topic_embs = _embed_keys(topic_keys, graph, embedder)
        topic_scores = (topic_embs @ q_emb).tolist()
        ranked = sorted(zip(topic_scores, topic_keys), key=lambda x: x[0], reverse=True)
        top_topics = {key for score, key in ranked[:2] if score >= relevance_threshold}

        logger.debug(
            f"Top topics for '{query[:50]}': "
            f"{[(graph.nodes[k].get('name', k), round(s, 3)) for s, k in ranked[:3]]}"
        )

        if top_topics:
            members = _topic_descendants(graph, top_topics)
            # Standalone authored notes also remain searchable. Do not pad
            # a small relevant branch with unrelated disconnected topics,
            # even when its only member is the topic itself.
            candidate_pool = [k for k in all_keys if k in members or graph.nodes[k].get("node_type") == "note"]

    # --- String matches within candidate pool (name OR key_terms) ---
    string_hits = [
        k for k in candidate_pool
        if (graph.nodes[k].get("name", k).strip() and graph.nodes[k].get("name", k).lower() in q_lower)
        or _key_term_hit(graph.nodes[k], q_lower)
    ]
    if len(string_hits) >= top_k:
        return string_hits[:top_k]

    # --- Embedding fallback (with key_terms boost) ---
    string_hit_keys = set(string_hits)
    remaining = [k for k in candidate_pool if k not in string_hit_keys]
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
    top_idx = [i for i in np.argsort(scores)[::-1] if scores[i] >= relevance_threshold][:n_needed]
    return string_hits + [remaining[i] for i in top_idx]


def _node_chunk_ids(graph: nx.MultiDiGraph, node_key: str, limit: int | None = None) -> list[str]:
    """A node's own evidence chunk IDs (source_ids, then legacy chunk-ref leaves)."""
    ids: dict[str, None] = {}
    for chunk_id in graph.nodes[node_key].get("source_ids") or []:
        if limit is not None and len(ids) >= limit:
            return list(ids)
        if isinstance(chunk_id, str):
            ids[chunk_id] = None
    for source, _, data in graph.in_edges(node_key, data=True):
        if limit is not None and len(ids) >= limit:
            break
        if data.get("relation") == "EVIDENCE_OF" and graph.nodes[source].get("node_type") == "chunk_ref":
            ids[source] = None
    return list(ids)


def _find_source_diverse_anchors(
    query: str,
    graph: nx.MultiDiGraph,
    embedder: Embedder,
    store: DocumentStore,
    top_k: int,
    source_limit: int,
    key_term_boost: float,
    relevance_threshold: float,
    query_emb: np.ndarray,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Anchors for an explicit multi-document question.

    Topic scoping keeps only the top two topics, which can all belong to one
    document. Instead rank every graph node above the relevance threshold,
    take the best node of each distinct stored source (up to source_limit),
    then fill the remaining top_k slots in score order. Source groups come
    from the stored Chunk.source_id of every relevant node's full evidence,
    in one batched metadata lookup; no score shortlist precedes grouping, so
    many strong nodes of one document cannot hide another's relevant node.
    Returns (anchors, chunk_id -> source_id, node -> evidence IDs) so
    retrieval can seed per-source evidence without repeating the lookup.
    With fewer than two represented sources this falls back to the ordinary
    anchor search with empty maps.
    """
    if top_k <= 0 or graph.number_of_nodes() == 0:
        return [], {}, {}
    keys = [k for k, a in graph.nodes(data=True) if a.get("node_type") != "chunk_ref"]
    if not keys:
        return [], {}, {}
    q_lower = query.lower()
    scores = _embed_keys(keys, graph, embedder) @ query_emb
    if key_term_boost > 0:
        scores = scores + np.array(
            [key_term_boost if _key_term_hit(graph.nodes[k], q_lower) else 0.0 for k in keys],
            dtype=scores.dtype,
        )
    ranked = sorted((-score, key) for score, key in zip(scores.tolist(), keys) if score >= relevance_threshold)
    ranked_keys = [key for _, key in ranked]
    node_ids = {key: _node_chunk_ids(graph, key) for key in ranked_keys}
    chunk_sources = store.get_chunk_sources(dict.fromkeys(cid for key in ranked_keys for cid in node_ids[key]))

    anchors: list[str] = []
    represented: list[str] = []
    for key in ranked_keys:
        if len(anchors) >= top_k or len(represented) >= source_limit:
            break
        sources = dict.fromkeys(chunk_sources[cid] for cid in node_ids[key] if cid in chunk_sources)
        new = [source for source in sources if source not in represented]
        if new:
            anchors.append(key)
            represented.extend(new[:source_limit - len(represented)])
    if len(represented) < 2:
        fallback = _find_anchors(query, graph, embedder, top_k, key_term_boost, relevance_threshold,
                                 query_emb=query_emb)
        return fallback, {}, {}
    chosen = set(anchors)
    anchors.extend([key for key in ranked_keys if key not in chosen][:top_k - len(anchors)])
    return anchors, chunk_sources, node_ids


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

    for hop in range(max_hops):
        candidates: set[str] = set()

        for node_key in frontier:
            for _, v, edata in graph.out_edges(node_key, data=True):
                if v in visited:
                    continue
                if edata.get("relation") in _LEAF_RELATIONS:
                    continue
                if graph.nodes[v].get("node_type") == "chunk_ref":
                    continue
                candidates.add(v)
            for u, _, edata in graph.in_edges(node_key, data=True):
                if u in visited:
                    continue
                if edata.get("relation") in _LEAF_RELATIONS:
                    continue
                if graph.nodes[u].get("node_type") == "chunk_ref":
                    continue
                candidates.add(u)

        if not candidates:
            break

        cand_keys = sorted(candidates)
        cand_embs = _embed_keys(cand_keys, graph, embedder)
        scores = (cand_embs @ query_emb).tolist()

        ranked = sorted(zip(scores, cand_keys), key=lambda x: x[0], reverse=True)
        next_frontier: list[str] = []
        for score, v in ranked[:max_per_hop]:
            if score < threshold:
                break
            visited.add(v)
            subgraph_nodes.add(v)
            next_frontier.append(v)
            logger.debug(
                f"  Hop {hop + 1}: '{graph.nodes[v].get('name', v)}' "
                f"(score={score:.3f})"
            )

        if not next_frontier:
            break
        frontier = next_frontier

    # BFS chooses relevant nodes, but discovery edges alone omit relations
    # between anchors and between nodes selected on the same/earlier hop.
    # Visit only the selected nodes' outgoing adjacency to retain all induced
    # relations, including distinct parallel edges, without scanning the graph.
    subgraph_edges = [
        (source, target, data)
        for source in sorted(subgraph_nodes)
        for _, target, data in graph.out_edges(source, data=True)
        if target in subgraph_nodes
        and data.get("relation") not in _LEAF_RELATIONS
        and graph.nodes[source].get("node_type") != "chunk_ref"
        and graph.nodes[target].get("node_type") != "chunk_ref"
    ]
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
    max_chunks = max(1, int(cfg.get("max_chunks", 80)))
    key_term_boost = float(cfg.get("key_term_boost", 0.15))

    graph = builder.graph

    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty — returning empty RetrievalResult.")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="graph")

    q_emb = getattr(embedder, "embed_query", embedder.embed_one)(query)
    source_limit = multi_document_source_limit(query)
    chunk_sources: dict[str, str] = {}
    node_ids: dict[str, list[str]] = {}
    if source_limit:
        anchors, chunk_sources, node_ids = _find_source_diverse_anchors(
            query, graph, embedder, store, top_k_anchors, source_limit, key_term_boost, threshold, q_emb)
    else:
        anchors = _find_anchors(query, graph, embedder, top_k_anchors, key_term_boost, threshold, query_emb=q_emb)
    if not anchors:
        logger.warning(f"No anchor nodes found for query: '{query[:60]}'")
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="graph")

    logger.debug(
        f"Anchors: {[graph.nodes[k].get('name', k) for k in anchors]}"
    )

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
    all_chunk_ids: dict[str, None] = {}

    def add_ids(chunk_ids) -> None:
        for chunk_id in chunk_ids:
            if len(all_chunk_ids) >= max_chunks:
                break
            all_chunk_ids[chunk_id] = None

    if source_limit:
        # One owned chunk per represented source first, from the stored
        # metadata: a shared anchor may list many A IDs before its only B ID.
        seeded: set[str] = set()
        for node_key in anchors:
            for chunk_id in node_ids.get(node_key, ()):
                source = chunk_sources.get(chunk_id)
                if source is not None and source not in seeded and len(seeded) < source_limit:
                    seeded.add(source)
                    add_ids([chunk_id])
        # Then each anchor's own evidence so many nodes from one document
        # cannot exhaust max_chunks before another requested source.
        share = max(1, max_chunks // len(anchors))
        for node_key in anchors:
            add_ids(_node_chunk_ids(graph, node_key, share))

    # Specific content and relationship evidence should not be crowded out by
    # broad topics whose source_ids may cover the entire corpus.
    priority = {"content": 0, "subtopic": 1, "topic": 2, "chunk_ref": 3}
    added_relation_ids = False
    for node_key in sorted(subgraph_nodes, key=lambda key: (priority.get(graph.nodes[key].get("node_type"), 3), key)):
        attrs = graph.nodes[node_key]
        if attrs.get("node_type") != "content" and not added_relation_ids:
            for _, _, data in subgraph_edges:
                add_ids(data.get("source_chunk_ids") or [])
            added_relation_ids = True
        if attrs.get("node_type") == "chunk_ref":
            add_ids([node_key])
        else:
            add_ids(attrs.get("source_ids") or [])
            # Persisted legacy graphs ground content using incoming chunk-ref
            # leaves. They remain evidence even though BFS never traverses them.
            for source, _, data in graph.in_edges(node_key, data=True):
                if data.get("relation") == "EVIDENCE_OF" and graph.nodes[source].get("node_type") == "chunk_ref":
                    add_ids([source])
    if not added_relation_ids:
        for _, _, data in subgraph_edges:
            add_ids(data.get("source_chunk_ids") or [])

    chunks = [chunk.model_copy(update={"metadata": {**chunk.metadata, "retrieval_source": "graph"}})
              for chunk in store.get_chunks_by_ids(all_chunk_ids)]

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
