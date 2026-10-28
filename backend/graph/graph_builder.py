"""
Stage 2 — Concept Graph Construction: graph_builder.py

Builds and persists a NetworkX MultiDiGraph carrying a hierarchical
Topic → Subtopic → Content graph.

Node types:   topic | subtopic | content
Structural edges:  HAS_SUBTOPIC, HAS_CONTENT   (validated)
Semantic edges:    RELATED_TO with independent source/manual/note backing

Polyhierarchy is achieved purely through edges — a single Content node may
receive HAS_CONTENT from multiple Subtopics. MultiDiGraph supports parallel
edges for directed hierarchy, while each unordered concept pair has one association.

Public API:
    builder = GraphBuilder(graph_path)
    builder.apply_proposals(bundle)   # Stage-1 output → graph writes
    builder.prune_disconnected()      # remove isolated nodes (no edges)
    builder.save()
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from backend.utils.files import atomic_write_text
from backend.utils.associations import (
    add_association, normalize_association_pair, normalize_associations,
    remove_association, remove_association_supports,
)

from backend.graph.schema import (
    ALL_RELATIONS,
    STRUCTURAL_RELATIONS,
    ContentAttrs,
    ContentProposal,
    EdgeInput,
    MAX_GRAPH_PARENTS,
    MAX_GRAPH_REFERENCES,
    NodeInput,
    NODE_SCHEMA_VERSION,
    NodeAttrs,
    ProposalBundle,
    SubtopicAttrs,
    SubtopicProposal,
    TopicAttrs,
    TopicProposal,
    _now_iso,
    make_key,
    bounded_key_terms,
    legacy_node_schema,
    normalize_legacy_node_attrs,
    normalize_legacy_edge_attrs,
    validate_node_attrs,
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
        self._graph.graph["node_schema_version"] = NODE_SCHEMA_VERSION
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        normalize_associations(self._graph)
        self._graph.graph["node_schema_version"] = NODE_SCHEMA_VERSION
        data = nx.node_link_data(self._graph)
        atomic_write_text(self._path, json.dumps(data, indent=2))
        logger.info(
            f"Graph saved to {self._path} "
            f"({self.node_count()} nodes, {self.edge_count()} edges)"
        )

    def clone(self, graph_path: str | Path) -> "GraphBuilder":
        """Independent candidate graph, including nested node/edge attributes."""
        candidate = deepcopy(self)
        candidate._path = Path(graph_path)
        return candidate

    def load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data, directed=True, multigraph=True)
        legacy = legacy_node_schema(graph.graph.get("node_schema_version"))
        for _, attrs in graph.nodes(data=True):
            if legacy:
                attrs.update(normalize_legacy_node_attrs(attrs))
            else:
                validate_node_attrs(attrs)
        for _, _, attrs in graph.edges(data=True):
            if legacy:
                attrs.update(normalize_legacy_edge_attrs(attrs))
            else:
                EdgeInput.model_validate(attrs)
        normalize_associations(graph)
        graph.graph["node_schema_version"] = NODE_SCHEMA_VERSION
        self._graph = graph
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
        NodeInput.model_validate(attrs.to_graph_dict())
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
            _merge_source_ids(existing_parents, attrs.parent_topic_keys, MAX_GRAPH_PARENTS)

        if isinstance(attrs, ContentAttrs):
            if attrs.content_type != "other" and existing.get("content_type") == "other":
                existing["content_type"] = attrs.content_type
            existing_parents = existing.setdefault("parent_subtopic_keys", [])
            _merge_source_ids(existing_parents, attrs.parent_subtopic_keys, MAX_GRAPH_PARENTS)

            # raw_excerpt: keep the longer verbatim quote
            if len(attrs.raw_excerpt) > len(existing.get("raw_excerpt", "")):
                existing["raw_excerpt"] = attrs.raw_excerpt

            # key_terms: union, preserve order, case-insensitive dedup
            existing["key_terms"] = bounded_key_terms([*existing.get("key_terms", []), *attrs.key_terms])

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
        NodeInput.model_validate({**node, **{field: value for field, value in updates.items() if field in allowed}})
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
        EdgeInput.model_validate({"relation": relation, "source_chunk_ids": chunk_ids})
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
        label: str | None = None,
        confidence: float = 0.5,
        source_chunk_ids: list[str] | None = None,
        *,
        kind: str = "manual",
        owner: str | None = None,
        verified: bool = False,
    ) -> bool:
        """
        Add plain, bidirectionally traversable relatedness. Legacy labels are
        accepted but discarded; independent backing survives later cleanup.
        """
        if source_key not in self._graph or target_key not in self._graph:
            raise ValueError(
                f"Cannot add RELATED_TO — missing node: "
                f"source={source_key!r}, target={target_key!r}"
            )

        return add_association(self._graph, source_key, target_key, kind=kind, owner=owner,
                               confidence=confidence, source_chunk_ids=source_chunk_ids, verified=verified)

    # ------------------------------------------------------------------
    # Mutations (used by API endpoints)
    # ------------------------------------------------------------------

    def remove_node(self, key: str) -> bool:
        if key not in self._graph:
            return False
        self._graph.remove_node(key)
        return True

    def merge_nodes(self, keep: str, drop: str) -> str:
        """Merge attributes and rewire incident edges, preserving evidence."""
        if keep == drop or keep not in self._graph or drop not in self._graph:
            raise ValueError("Choose two different existing nodes to merge.")
        incident_pairs = {tuple(sorted((u, v))) for u, v, attrs in
                          (*self._graph.in_edges(drop, data=True), *self._graph.out_edges(drop, data=True))
                          if attrs.get("relation") == "RELATED_TO"}
        for source, target in incident_pairs:
            normalize_association_pair(self._graph, source, target)
        retained, removed = self._graph.nodes[keep], self._graph.nodes[drop]
        previous_note_text = (retained.get("title") or retained.get("name"), retained.get("body", ""))
        merged = deepcopy(retained)
        for field in ("source_ids", "parent_topic_keys", "parent_subtopic_keys"):
            _merge_source_ids(merged.setdefault(field, []), removed.get(field, []),
                              MAX_GRAPH_REFERENCES if field == "source_ids" else MAX_GRAPH_PARENTS)
        merged["key_terms"] = bounded_key_terms([*retained.get("key_terms", []), *removed.get("key_terms", [])])
        for field in ("summary", "raw_excerpt", "body"):
            if len(removed.get(field) or "") > len(merged.get(field) or ""):
                merged[field] = removed[field]
        NodeInput.model_validate(merged)
        retained.update(merged)
        invalidated = {"note:" + drop} if removed.get("node_type") == "note" else set()
        if retained.get("node_type") == "note" and previous_note_text != (retained.get("title") or retained.get("name"), retained.get("body", "")):
            invalidated.add("note:" + keep)
        if invalidated:
            # A checked passage cannot be renamed to a different note body
            # while keeping its previous verification claim.
            for key in (keep, drop):
                remove_association_supports(self._graph, incident_to=key, invalidated_chunk_ids=invalidated)
        edges = [(u, v, deepcopy(attrs)) for u, v, attrs in (*self._graph.in_edges(drop, data=True), *self._graph.out_edges(drop, data=True))]
        self._graph.remove_node(drop)
        for source, target, attrs in edges:
            source = keep if source == drop else source
            target = keep if target == drop else target
            if attrs.get("relation") == "RELATED_TO":
                if source == target:
                    continue
                records = attrs.get("supports") or [{"kind": "legacy", "owner": None,
                    "source_chunk_ids": attrs.get("source_chunk_ids", []), "confidence": attrs.get("confidence", 0.5), "verified": False}]
                for support in records:
                    add_association(self._graph, source, target, kind=support["kind"],
                                    owner=keep if support.get("owner") == drop else support.get("owner"),
                                    source_chunk_ids=["note:" + keep if value == "note:" + drop else value for value in support.get("source_chunk_ids", [])],
                                    confidence=support["confidence"], verified=support.get("verified", False))
                continue
            existing = self._graph.get_edge_data(source, target) or {}
            match = next((item for item in existing.values() if item.get("relation") == attrs.get("relation") and item.get("label") == attrs.get("label")), None)
            if match is not None:
                _merge_source_ids(match.setdefault("source_chunk_ids", []), attrs.get("source_chunk_ids", []))
                if "confidence" in attrs:
                    match["confidence"] = max(float(match.get("confidence", 0)), float(attrs["confidence"]))
            else:
                self._graph.add_edge(source, target, **attrs)
        for _, attrs in self._graph.nodes(data=True):
            for field in ("parent_topic_keys", "parent_subtopic_keys"):
                if field in attrs:
                    attrs[field] = list(dict.fromkeys(keep if key == drop else key for key in attrs[field]))
        retained["updated_at"] = _now_iso()
        return keep

    def remove_edge(
        self,
        source_key: str,
        relation: str,
        target_key: str,
        label: str | None = None,
    ) -> bool:
        """
        Remove a directed structural edge or the complete plain association.
        Returns True if an edge was removed.
        """
        if relation == "RELATED_TO":
            return remove_association(self._graph, source_key, target_key)
        existing = self._graph.get_edge_data(source_key, target_key) or {}
        for edge_key, edge_data in existing.items():
            if edge_data.get("relation") != relation:
                continue
            self._graph.remove_edge(source_key, target_key, key=edge_key)
            return True
        return False

    def prune_disconnected(self) -> int:
        """
        Remove generated nodes that have no edges (in-degree + out-degree == 0).
        Manual nodes can be deliberate standalone notes and must survive ingestion.
        Returns the number of nodes removed.
        """
        isolated = [
            n for n, attrs in self._graph.nodes(data=True)
            if self._graph.degree(n) == 0 and attrs.get("origin") != "manual"
        ]
        for n in isolated:
            self._graph.remove_node(n)
        if isolated:
            logger.info(f"Pruned {len(isolated)} disconnected node(s)")
        return len(isolated)

    def prune_to_ratio(self) -> dict[str, int]:
        """
        Report (never enforce) the soft shape heuristic:
            # content nodes > # subtopic nodes > # topic nodes

        Earlier versions deleted the least-connected topics/subtopics until
        the heuristic held, which destroyed valid, connected concepts (a
        1-topic/1-subtopic/1-content graph vanished entirely). The ratio is
        aesthetic, not a correctness invariant, so a violation is only logged.

        Kept for API compatibility: returns the removed count per node type,
        which is always zero.
        """
        counts = {"topic": 0, "subtopic": 0, "content": 0}
        for _, a in self._graph.nodes(data=True):
            ntype = a.get("node_type")
            if ntype in counts:
                counts[ntype] += 1

        n_t, n_s, n_c = counts["topic"], counts["subtopic"], counts["content"]
        if (n_t and n_s and n_t >= n_s) or (n_s and n_c and n_s >= n_c):
            logger.info(
                f"prune_to_ratio: hierarchy shape {n_t} topic(s) / {n_s} subtopic(s) / "
                f"{n_c} content(s) does not satisfy content > subtopic > topic; "
                "leaving nodes intact"
            )
        return {"topic": 0, "subtopic": 0}

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
                kind="automatic",
                verified=rp.verified,
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

def _merge_source_ids(container: list[str], new_ids: list[str], max_items: int = MAX_GRAPH_REFERENCES) -> None:
    merged = list(dict.fromkeys([*container, *(cid for cid in new_ids if cid)]))
    if len(merged) > max_items:
        raise ValueError("Graph reference limit exceeded; citations and parents were not truncated.")
    container[:] = merged


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
