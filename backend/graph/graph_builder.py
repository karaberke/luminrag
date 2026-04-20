"""
Stage 2 — Concept Graph Construction: graph_builder.py

Builds and persists a NetworkX MultiDiGraph carrying a hierarchical
Topic → Subtopic → Content graph.

Node types:   topic | subtopic | content
Structural edges:  HAS_SUBTOPIC, HAS_CONTENT   (validated)
Semantic edges:    RELATED_TO {label: str, confidence: float}  (free-form label)

Polyhierarchy is achieved purely through edges — a single Content node may
receive HAS_CONTENT from multiple Subtopics. MultiDiGraph supports parallel
edges so distinct RELATED_TO labels coexist between the same pair.

Public API:
    builder = GraphBuilder(graph_path)
    builder.apply_proposals(bundle)   # Stage-1 output → graph writes
    builder.prune_disconnected()      # remove isolated nodes (no edges)
    builder.save()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from backend.graph.schema import (
    ALL_RELATIONS,
    STRUCTURAL_RELATIONS,
    ContentAttrs,
    ContentProposal,
    NodeAttrs,
    ProposalBundle,
    SubtopicAttrs,
    SubtopicProposal,
    TopicAttrs,
    TopicProposal,
    _now_iso,
    make_key,
    normalise_relation_label,
)

logger = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    """Summary of an apply_proposals() run — useful for ingest response payloads."""

    topics_added: int = 0
    topics_merged: int = 0
    subtopics_added: int = 0
    subtopics_merged: int = 0
    contents_added: int = 0
    contents_merged: int = 0
    structural_edges_added: int = 0
    related_edges_added: int = 0
    related_edges_merged: int = 0
    nodes_pruned: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["warnings"] = list(self.warnings)
        return data


class GraphBuilder:
    """
    Incremental builder for the hierarchical concept graph.

    Safe to call apply_proposals() repeatedly; duplicates are merged via
    deterministic keys and longest-wins summary rules. If graph_path exists,
    it is loaded on construction so the graph accumulates across runs.
    """

    def __init__(self, graph_path: str | Path) -> None:
        self._path = Path(graph_path)
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Graph saved to {self._path} "
            f"({self.node_count()} nodes, {self.edge_count()} edges)"
        )

    def load(self) -> None:
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

    def get_node(self, key: str) -> dict | None:
        return dict(self._graph.nodes[key]) if key in self._graph else None

    # ------------------------------------------------------------------
    # Low-level node insertion
    # ------------------------------------------------------------------

    def add_node(self, attrs: NodeAttrs) -> str:
        """
        Insert or merge a typed node. Returns the resolved key.

        Merge rules (per node_type):
          topic / subtopic:   keep original name + created_at; longer summary wins;
                              source_ids union; updated_at refreshed.
          content:            keep original title + created_at; longer summary wins;
                              illustration preserved if set; parent_subtopic_keys union.
        """
        key = make_key(attrs.node_type, attrs.name)

        if key not in self._graph:
            self._graph.add_node(key, **attrs.to_graph_dict())
            return key

        existing = self._graph.nodes[key]
        now = _now_iso()
        existing["updated_at"] = now
        _merge_source_ids(existing.setdefault("source_ids", []), attrs.source_ids)

        if isinstance(attrs, (TopicAttrs, SubtopicAttrs, ContentAttrs)):
            if len(attrs.summary) > len(existing.get("summary", "")):
                existing["summary"] = attrs.summary
            if attrs.illustration and not existing.get("illustration"):
                existing["illustration"] = attrs.illustration.model_dump(mode="json")

        if isinstance(attrs, TopicAttrs):
            if existing.get("scope") != "narrow" and attrs.scope == "narrow":
                existing["scope"] = "narrow"

        if isinstance(attrs, SubtopicAttrs):
            existing_parents = existing.setdefault("parent_topic_keys", [])
            _merge_source_ids(existing_parents, attrs.parent_topic_keys)

        if isinstance(attrs, ContentAttrs):
            if attrs.content_type != "other" and existing.get("content_type") == "other":
                existing["content_type"] = attrs.content_type
            existing_parents = existing.setdefault("parent_subtopic_keys", [])
            _merge_source_ids(existing_parents, attrs.parent_subtopic_keys)

            # raw_excerpt: keep the longer verbatim quote
            if len(attrs.raw_excerpt) > len(existing.get("raw_excerpt", "")):
                existing["raw_excerpt"] = attrs.raw_excerpt

            # key_terms: union, preserve order, case-insensitive dedup
            existing_terms = existing.setdefault("key_terms", [])
            seen = {t.lower() for t in existing_terms}
            for term in attrs.key_terms:
                if term and term.lower() not in seen:
                    existing_terms.append(term)
                    seen.add(term.lower())

            # illustration_path: never overwrite a generated image
            if attrs.illustration_path and not existing.get("illustration_path"):
                existing["illustration_path"] = attrs.illustration_path

        return key

    def patch_node(self, key: str, updates: dict) -> bool:
        """
        Partially update mutable attributes of an existing node.
        Returns False if the key does not exist.
        Immutable fields (node_type, created_at) and the key itself are ignored.
        """
        if key not in self._graph:
            return False
        allowed = {"name", "summary", "scope", "content_type", "raw_excerpt", "key_terms", "source_ids"}
        node = self._graph.nodes[key]
        for field, value in updates.items():
            if field in allowed:
                node[field] = value
        node["updated_at"] = _now_iso()
        return True

    def set_illustration_path(self, key: str, path: str) -> bool:
        """Record the filesystem path of a generated illustration image."""
        if key not in self._graph:
            return False
        self._graph.nodes[key]["illustration_path"] = path
        self._graph.nodes[key]["updated_at"] = _now_iso()
        return True

    # ------------------------------------------------------------------
    # Low-level edge insertion
    # ------------------------------------------------------------------

    def add_structural_edge(
        self,
        source_key: str,
        relation: str,
        target_key: str,
        source_chunk_ids: list[str] | None = None,
    ) -> bool:
        """
        Add or merge a structural edge (HAS_SUBTOPIC / HAS_CONTENT).
        Returns True if a new edge was created, False if it merged into an existing one.
        Raises ValueError on unknown relation or missing endpoints.
        """
        if relation not in STRUCTURAL_RELATIONS:
            raise ValueError(f"Not a structural relation: '{relation}'")
        if source_key not in self._graph or target_key not in self._graph:
            raise ValueError(
                f"Cannot add edge — missing node: "
                f"source={source_key!r}, target={target_key!r}"
            )

        chunk_ids = list(source_chunk_ids or [])
        existing = self._graph.get_edge_data(source_key, target_key) or {}
        for edge_data in existing.values():
            if edge_data.get("relation") == relation:
                _merge_source_ids(edge_data.setdefault("source_chunk_ids", []), chunk_ids)
                return False

        self._graph.add_edge(
            source_key,
            target_key,
            relation=relation,
            source_chunk_ids=chunk_ids,
        )
        return True

    def add_related_to(
        self,
        source_key: str,
        target_key: str,
        label: str,
        confidence: float = 0.5,
        source_chunk_ids: list[str] | None = None,
    ) -> bool:
        """
        Add or merge a RELATED_TO semantic edge. The label is normalised to
        snake_case; same (source, target, normalised_label) merges by
        accumulating source_chunk_ids and taking max(confidence).
        Returns True if a new edge was created, False if it merged.
        """
        if source_key not in self._graph or target_key not in self._graph:
            raise ValueError(
                f"Cannot add RELATED_TO — missing node: "
                f"source={source_key!r}, target={target_key!r}"
            )

        norm = normalise_relation_label(label)
        chunk_ids = list(source_chunk_ids or [])
        existing = self._graph.get_edge_data(source_key, target_key) or {}
        for edge_data in existing.values():
            if (
                edge_data.get("relation") == "RELATED_TO"
                and edge_data.get("label") == norm
            ):
                edge_data["confidence"] = max(
                    float(edge_data.get("confidence", 0.0)), float(confidence)
                )
                _merge_source_ids(edge_data.setdefault("source_chunk_ids", []), chunk_ids)
                return False

        self._graph.add_edge(
            source_key,
            target_key,
            relation="RELATED_TO",
            label=norm,
            confidence=float(confidence),
            source_chunk_ids=chunk_ids,
        )
        return True

    # ------------------------------------------------------------------
    # Mutations (used by API endpoints)
    # ------------------------------------------------------------------

    def remove_node(self, key: str) -> bool:
        if key not in self._graph:
            return False
        self._graph.remove_node(key)
        return True

    def remove_edge(
        self,
        source_key: str,
        relation: str,
        target_key: str,
        label: str | None = None,
    ) -> bool:
        """
        Remove an edge matching (source, relation, target). For RELATED_TO
        edges, `label` narrows the match to a specific normalised label.
        Returns True if an edge was removed.
        """
        existing = self._graph.get_edge_data(source_key, target_key) or {}
        norm_label = normalise_relation_label(label) if label else None
        for edge_key, edge_data in existing.items():
            if edge_data.get("relation") != relation:
                continue
            if relation == "RELATED_TO" and norm_label is not None:
                if edge_data.get("label") != norm_label:
                    continue
            self._graph.remove_edge(source_key, target_key, key=edge_key)
            return True
        return False

    def prune_disconnected(self) -> int:
        """
        Remove all nodes that have no edges (in-degree + out-degree == 0).
        Returns the number of nodes removed.
        """
        isolated = [n for n in self._graph.nodes() if self._graph.degree(n) == 0]
        for n in isolated:
            self._graph.remove_node(n)
        if isolated:
            logger.info(f"Pruned {len(isolated)} disconnected node(s)")
        return len(isolated)

    def prune_to_ratio(self) -> dict[str, int]:
        """
        Enforce the invariant:
            # content nodes > # subtopic nodes > # topic nodes

        Removes the *least-connected* nodes of whichever type is
        over-represented, iterating until the invariant holds (or
        until we cannot remove more without emptying a tier).

        After this call you should run prune_disconnected() again because
        removing a topic cascades to orphan any subtopics that had only
        that topic as a parent.

        Returns a dict with the count removed per node type.
        """
        removed: dict[str, int] = {"topic": 0, "subtopic": 0}

        def _by_type(ntype: str) -> list[tuple[str, int]]:
            """Return [(node_id, degree), …] sorted ascending by degree."""
            return sorted(
                [
                    (n, self._graph.degree(n))
                    for n, a in self._graph.nodes(data=True)
                    if a.get("node_type") == ntype
                ],
                key=lambda x: x[1],
            )

        for _ in range(20):
            topics    = _by_type("topic")
            subtopics = _by_type("subtopic")
            contents  = _by_type("content")

            n_t, n_s, n_c = len(topics), len(subtopics), len(contents)
            changed = False

            # --- Rule 1: n_subtopic > n_topic ---
            if n_t > 0 and n_s > 0 and n_t >= n_s:
                # Need n_t < n_s  →  remove (n_t - n_s + 1) least-connected topics
                to_remove = n_t - n_s + 1
                for node_id, _ in topics[:to_remove]:
                    self._graph.remove_node(node_id)
                    removed["topic"] += 1
                changed = True
                continue  # re-count before checking rule 2

            # --- Rule 2: n_content > n_subtopic ---
            if n_s > 0 and n_c > 0 and n_s >= n_c:
                to_remove = n_s - n_c + 1
                for node_id, _ in subtopics[:to_remove]:
                    self._graph.remove_node(node_id)
                    removed["subtopic"] += 1
                changed = True
                continue

            if not changed:
                break

        if any(removed.values()):
            logger.info(
                f"prune_to_ratio: removed {removed['topic']} topic(s), "
                f"{removed['subtopic']} subtopic(s)"
            )
        return removed

    # ------------------------------------------------------------------
    # Stage-1 → Stage-2: the single entry point ingestion uses
    # ------------------------------------------------------------------

    def apply_proposals(self, bundle: ProposalBundle) -> ApplyResult:
        """
        Apply a full Stage-1 proposal bundle to the graph atomically.

        Order is deliberate:
          1. Insert / merge Topics.      (display_name → key map)
          2. Insert / merge Subtopics.   (resolves parent topics)
          3. Insert / merge Contents.    (resolves parent subtopics + topics)
          4. Wire structural edges (HAS_SUBTOPIC, HAS_CONTENT).
          5. Wire semantic RELATED_TO edges.

        Caller is responsible for subsequently invoking save() /
        export_graph() / warm_graph_retriever().
        """
        result = ApplyResult()

        # 1. Topics ---------------------------------------------------------
        topic_key_by_name: dict[str, str] = {}
        for tp in bundle.topics:
            key, was_new = self._apply_topic(tp)
            topic_key_by_name[_norm_lookup(tp.name)] = key
            if was_new:
                result.topics_added += 1
            else:
                result.topics_merged += 1

        # 2. Subtopics ------------------------------------------------------
        subtopic_key_by_name: dict[str, str] = {}
        pending_subtopic_parents: list[tuple[str, list[str], list[str]]] = []
        for sp in bundle.subtopics:
            key, was_new = self._apply_subtopic(sp, topic_key_by_name)
            subtopic_key_by_name[_norm_lookup(sp.name)] = key
            pending_subtopic_parents.append(
                (key, sp.parent_topic_names, sp.parent_subtopic_names)
            )
            if was_new:
                result.subtopics_added += 1
            else:
                result.subtopics_merged += 1

        # 3. Contents -------------------------------------------------------
        content_keys: list[tuple[ContentProposal, str]] = []
        for cp in bundle.contents:
            key, was_new = self._apply_content(cp, subtopic_key_by_name, topic_key_by_name)
            content_keys.append((cp, key))
            if was_new:
                result.contents_added += 1
            else:
                result.contents_merged += 1

        # 4. Structural edges ----------------------------------------------
        # 4a. Topic → Subtopic
        for sub_key, parent_topic_names, parent_subtopic_names in pending_subtopic_parents:
            for tname in parent_topic_names:
                tkey = topic_key_by_name.get(_norm_lookup(tname)) or _lookup_existing(
                    self._graph, "topic", tname
                )
                if tkey is None:
                    result.warnings.append(
                        f"Subtopic '{sub_key}' parent topic '{tname}' not found"
                    )
                    continue
                if self.add_structural_edge(tkey, "HAS_SUBTOPIC", sub_key):
                    result.structural_edges_added += 1

            # 4b. Subtopic → Subtopic (nesting)
            for sname in parent_subtopic_names:
                pkey = subtopic_key_by_name.get(_norm_lookup(sname)) or _lookup_existing(
                    self._graph, "subtopic", sname
                )
                if pkey is None or pkey == sub_key:
                    continue
                if self.add_structural_edge(pkey, "HAS_SUBTOPIC", sub_key):
                    result.structural_edges_added += 1

        # 4c. Subtopic/Topic → Content
        for cp, ckey in content_keys:
            attached = False
            for sname in cp.parent_subtopic_names:
                skey = subtopic_key_by_name.get(_norm_lookup(sname)) or _lookup_existing(
                    self._graph, "subtopic", sname
                )
                if skey is None:
                    result.warnings.append(
                        f"Content '{ckey}' parent subtopic '{sname}' not found"
                    )
                    continue
                if self.add_structural_edge(
                    skey, "HAS_CONTENT", ckey, source_chunk_ids=cp.evidence_chunk_ids
                ):
                    result.structural_edges_added += 1
                attached = True

            if not attached:
                # Fall back to Topic → Content when no subtopic is available
                for tname in cp.parent_topic_names:
                    tkey = topic_key_by_name.get(_norm_lookup(tname)) or _lookup_existing(
                        self._graph, "topic", tname
                    )
                    if tkey is None:
                        continue
                    if self.add_structural_edge(
                        tkey, "HAS_CONTENT", ckey, source_chunk_ids=cp.evidence_chunk_ids
                    ):
                        result.structural_edges_added += 1

        # 5. Semantic edges -------------------------------------------------
        for rp in bundle.related:
            if rp.source_key not in self._graph or rp.target_key not in self._graph:
                result.warnings.append(
                    f"RELATED_TO skipped — missing node(s): "
                    f"{rp.source_key} -> {rp.target_key}"
                )
                continue
            if rp.source_key == rp.target_key:
                continue
            added = self.add_related_to(
                rp.source_key,
                rp.target_key,
                rp.label,
                rp.confidence,
                rp.source_chunk_ids,
            )
            if added:
                result.related_edges_added += 1
            else:
                result.related_edges_merged += 1

        return result

    # ------------------------------------------------------------------
    # Per-type application helpers
    # ------------------------------------------------------------------

    def _apply_topic(self, tp: TopicProposal) -> tuple[str, bool]:
        key = make_key("topic", tp.name)
        was_new = key not in self._graph
        attrs = TopicAttrs(
            name=tp.name,
            summary=tp.summary,
            scope=tp.scope,
            illustration=tp.illustration,
            source_ids=list(tp.source_chunk_ids),
        )
        self.add_node(attrs)
        return key, was_new

    def _apply_subtopic(
        self, sp: SubtopicProposal, topic_key_by_name: dict[str, str]
    ) -> tuple[str, bool]:
        key = make_key("subtopic", sp.name)
        was_new = key not in self._graph

        parent_topic_keys: list[str] = []
        for tname in sp.parent_topic_names:
            tkey = topic_key_by_name.get(_norm_lookup(tname)) or _lookup_existing(
                self._graph, "topic", tname
            )
            if tkey:
                parent_topic_keys.append(tkey)

        attrs = SubtopicAttrs(
            name=sp.name,
            summary=sp.summary,
            illustration=sp.illustration,
            parent_topic_keys=parent_topic_keys,
            source_ids=list(sp.source_chunk_ids),
        )
        self.add_node(attrs)
        return key, was_new

    def _apply_content(
        self,
        cp: ContentProposal,
        subtopic_key_by_name: dict[str, str],
        topic_key_by_name: dict[str, str],
    ) -> tuple[str, bool]:
        key = make_key("content", cp.title)
        was_new = key not in self._graph

        parent_subtopic_keys: list[str] = []
        for sname in cp.parent_subtopic_names:
            skey = subtopic_key_by_name.get(_norm_lookup(sname)) or _lookup_existing(
                self._graph, "subtopic", sname
            )
            if skey:
                parent_subtopic_keys.append(skey)

        attrs = ContentAttrs(
            name=cp.title,
            content_type=cp.content_type,
            summary=cp.summary,
            raw_excerpt=cp.raw_excerpt,
            key_terms=list(cp.key_terms),
            illustration=cp.illustration,
            parent_subtopic_keys=parent_subtopic_keys,
            source_ids=list(cp.evidence_chunk_ids),
        )
        self.add_node(attrs)
        return key, was_new


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _merge_source_ids(container: list[str], new_ids: list[str]) -> None:
    existing = set(container)
    for cid in new_ids:
        if cid and cid not in existing:
            container.append(cid)
            existing.add(cid)


def _norm_lookup(name: str) -> str:
    return name.strip().lower()


def _lookup_existing(graph: nx.MultiDiGraph, node_type: str, display_name: str) -> str | None:
    """Resolve a display name to an existing node key of the given type, if present."""
    if not display_name:
        return None
    candidate = make_key(node_type, display_name)  # type: ignore[arg-type]
    return candidate if candidate in graph else None


# Relation allowlist exposed for validation in the API layer.
__all__ = [
    "GraphBuilder",
    "ApplyResult",
    "ALL_RELATIONS",
]
