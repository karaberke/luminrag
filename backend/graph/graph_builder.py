"""
Stage 2 - Concept Graph Construction: graph_builder.py

Builds and persists a NetworkX MultiDiGraph from GraphTriple objects.

Graph structure:
  Nodes  — one per unique concept (keyed by normalised lowercase name)
           Attributes: name (str), source_ids (list[str])
  Edges  — directed (head -> tail)
           Attributes: relation (str), source_chunk_ids (list[str])
           Multiple distinct relations between the same node pair are kept
           as separate edges (MultiDiGraph). Duplicate (head, relation, tail)
           triples are merged by accumulating source_chunk_ids.

Persistence: JSON via nx.node_link_data → graph_db.path in config/db.yaml.

Public API:
    builder = GraphBuilder(graph_path)
    builder.add_triples(triples)
    builder.save()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

from backend.schemas import GraphTriple

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Incremental builder for the concept knowledge graph.

    Safe to call add_triples() multiple times (e.g. once per document).
    If graph_path already exists it is loaded on construction, so the
    graph accumulates across ingestion runs.
    """

    def __init__(self, graph_path: str | Path) -> None:
        self._path = Path(graph_path)
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(name: str) -> str:
        """Normalised node key: stripped lowercase."""
        return name.strip().lower()

    def _ensure_node(self, key: str, display_name: str) -> None:
        if key not in self._graph:
            self._graph.add_node(key, name=display_name, source_ids=[])

    def _merge_source_ids(self, container: list[str], new_ids: list[str]) -> None:
        existing = set(container)
        for cid in new_ids:
            if cid not in existing:
                container.append(cid)
                existing.add(cid)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_triples(self, triples: list[GraphTriple]) -> None:
        """
        Merge a list of GraphTriples into the graph.

        - Creates nodes for head and tail if they don't exist.
        - Accumulates source_ids on nodes.
        - Merges edges that share the same (head, relation, tail) by
          accumulating source_chunk_ids; otherwise adds a new edge.
        """
        for triple in triples:
            head_key = self._key(triple.head)
            tail_key = self._key(triple.tail)

            self._ensure_node(head_key, triple.head)
            self._ensure_node(tail_key, triple.tail)

            # Accumulate chunk IDs on both nodes
            self._merge_source_ids(
                self._graph.nodes[head_key]["source_ids"], triple.source_chunk_ids
            )
            self._merge_source_ids(
                self._graph.nodes[tail_key]["source_ids"], triple.source_chunk_ids
            )

            # Try to merge into an existing edge with the same relation
            existing = self._graph.get_edge_data(head_key, tail_key) or {}
            merged = False
            for edge_data in existing.values():
                if edge_data.get("relation") == triple.relation:
                    self._merge_source_ids(
                        edge_data["source_chunk_ids"], triple.source_chunk_ids
                    )
                    merged = True
                    break

            if not merged:
                self._graph.add_edge(
                    head_key,
                    tail_key,
                    relation=triple.relation,
                    source_chunk_ids=list(triple.source_chunk_ids),
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialise the graph to JSON at graph_path."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Graph saved to {self._path} "
            f"({self.node_count()} nodes, {self.edge_count()} edges)"
        )

    def load(self) -> None:
        """Load a previously saved graph from graph_path."""
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data, directed=True, multigraph=True)
        logger.info(
            f"Graph loaded from {self._path} "
            f"({self.node_count()} nodes, {self.edge_count()} edges)"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def get_node(self, name: str) -> dict | None:
        """Return node attribute dict for a concept name, or None."""
        key = self._key(name)
        return dict(self._graph.nodes[key]) if key in self._graph else None

    def add_source_cluster(self, cluster_name: str, entity_keys: list[str]) -> None:
        """
        Create a topic cluster node and link all provided entity nodes to it
        via BELONGS_TO_TOPIC edges.

        The cluster node acts as a domain hub: the graph retriever scores
        cluster nodes against the query first, then restricts BFS anchors to
        members of the best-matching cluster(s), preventing cross-domain
        contamination (e.g. chemistry nodes appearing for NLP queries).

        Args:
            cluster_name:  Human-readable topic name (e.g. "NLP Lecture").
            entity_keys:   Normalised (lowercase) keys of entity nodes that
                           belong to this cluster.
        """
        cluster_key = self._key(cluster_name)
        if cluster_key not in self._graph:
            self._graph.add_node(
                cluster_key,
                name=cluster_name,
                source_ids=[],
                node_type="cluster",
            )
        else:
            self._graph.nodes[cluster_key]["node_type"] = "cluster"

        for ek in entity_keys:
            if ek not in self._graph or ek == cluster_key:
                continue
            existing = self._graph.get_edge_data(ek, cluster_key) or {}
            already = any(
                ed.get("relation") == "BELONGS_TO_TOPIC"
                for ed in existing.values()
            )
            if not already:
                self._graph.add_edge(
                    ek,
                    cluster_key,
                    relation="BELONGS_TO_TOPIC",
                    source_chunk_ids=[],
                )

    def remove_node(self, key: str) -> bool:
        """Remove a node and all its incident edges. Returns True if it existed."""
        key = self._key(key)
        if key not in self._graph:
            return False
        self._graph.remove_node(key)
        return True

    def remove_edge(self, source_key: str, relation: str, target_key: str) -> bool:
        """Remove the first edge matching (source, relation, target). Returns True if removed."""
        source_key = self._key(source_key)
        target_key = self._key(target_key)
        existing = self._graph.get_edge_data(source_key, target_key) or {}
        for edge_key, edge_data in existing.items():
            if edge_data.get("relation") == relation:
                self._graph.remove_edge(source_key, target_key, key=edge_key)
                return True
        return False

    def get_neighbours(self, name: str) -> list[dict]:
        """
        Return all outgoing edges from a concept as
        [{target_name, relation, source_chunk_ids}].
        """
        key = self._key(name)
        if key not in self._graph:
            return []
        neighbours = []
        for _, target_key, edge_data in self._graph.out_edges(key, data=True):
            neighbours.append({
                "target": self._graph.nodes[target_key]["name"],
                "relation": edge_data["relation"],
                "source_chunk_ids": edge_data["source_chunk_ids"],
            })
        return neighbours
