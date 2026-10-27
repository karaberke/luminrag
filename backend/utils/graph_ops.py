"""Validated graph editing, sparse inverse operations and note wikilinks."""
from __future__ import annotations

import math
import re
from typing import Literal
from copy import deepcopy

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import ALL_RELATIONS, Illustration, NodeType, _now_iso, make_key
from backend.schemas import Chunk
from backend.utils.budgets import JsonWorkBudget
from backend.utils.associations import (
    AssociationSupport, MAX_SUPPORTS, association_id, normalize_association_pair, remove_association_supports,
)

_WIKILINK = re.compile(r"\[\[([^\[\]\n]{1,500})\]\]")
MAX_INVERSE_ITEMS = 100_000
MAX_INVERSE_BYTES = 16 * 1024 * 1024


def inverse_budget() -> JsonWorkBudget:
    # The compatibility response exposes the inverse under three names.
    return JsonWorkBudget(MAX_INVERSE_ITEMS, MAX_INVERSE_BYTES, copies=3)


class NodeInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=500)
    node_type: NodeType = "note"
    title: str = Field(default="", max_length=500)
    body: str = Field(default="", max_length=100_000)
    summary: str = Field(default="", max_length=100_000)
    raw_excerpt: str = Field(default="", max_length=100_000)
    key_terms: list[str] = Field(default_factory=list, max_length=200)
    source_ids: list[str] = Field(default_factory=list, max_length=20_000)
    parent_topic_keys: list[str] = Field(default_factory=list, max_length=200)
    parent_subtopic_keys: list[str] = Field(default_factory=list, max_length=200)
    scope: Literal["broad", "narrow"] = "broad"
    content_type: Literal["definition", "theorem", "technique", "example", "question", "figure", "other"] = "other"
    illustration: Illustration | None = None
    illustration_path: str | None = Field(default=None, max_length=2048)
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class EdgeInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    relation: Literal["HAS_SUBTOPIC", "HAS_CONTENT", "RELATED_TO", "EVIDENCE_OF"]
    label: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    source_chunk_ids: list[str] = Field(default_factory=list, max_length=20_000)
    supports: list[AssociationSupport] | None = Field(default=None, min_length=1, max_length=MAX_SUPPORTS)


def note_chunk_id(key: str) -> str:
    return "note:" + key


def note_chunks(graph: nx.MultiDiGraph) -> list[Chunk]:
    chunks = []
    for key, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "note":
            continue
        cid = note_chunk_id(key)
        attrs["source_ids"] = list(dict.fromkeys([*attrs.get("source_ids", []), cid]))
        chunks.append(Chunk(id=cid, source_id=key, modality="note", text=f"{attrs.get('title') or attrs.get('name', '')}\n\n{attrs.get('body', '')}", metadata={"node_id": key, "title": attrs.get("title") or attrs.get("name")}))
    return chunks


def resolve_wikilinks(builder: GraphBuilder, key: str) -> None:
    graph = builder.graph
    attrs = graph.nodes[key]
    remove_association_supports(graph, kind="wikilink", owner=key)
    if attrs.get("node_type") != "note":
        return
    from backend.graph._llm import resolve_synonym
    names = [data.get("name", "") for node, data in graph.nodes(data=True) if node != key]
    by_name = {data.get("name", "").strip().casefold(): node for node, data in graph.nodes(data=True) if node != key}
    for match in _WIKILINK.findall(attrs.get("body", "")):
        name = match.split("|", 1)[0].split("#", 1)[0].strip()
        resolved = resolve_synonym(name, names, None, 0.9)
        target = by_name.get((resolved or name).strip().casefold())
        if target:
            builder.add_related_to(key, target, confidence=1.0, kind="wikilink", owner=key,
                                   source_chunk_ids=[note_chunk_id(key)])


def _node_snapshot(graph, keys: list[str], budget: JsonWorkBudget) -> dict:
    unique = list(dict.fromkeys(keys))
    budget.reserve({"op": "restore_nodes", "ids": unique}, items=len(unique))
    nodes, edges, seen = [], [], set()
    for key in unique:
        if key not in graph:
            continue
        attrs = graph.nodes[key]
        budget.reserve({"id": key, "attrs": attrs})
        nodes.append((key, attrs))
        # A small update visits only incident edges. Deduplicate self-loops
        # and edges joining multiple selected nodes before accounting/copying.
        for incident in (graph.in_edges(key, keys=True, data=True), graph.out_edges(key, keys=True, data=True)):
            for u, v, edge_key, data in incident:
                identity = (u, v, edge_key)
                if identity in seen:
                    continue
                budget.reserve({"source": u, "target": v, "key": edge_key, "attrs": data})
                seen.add(identity)
                edges.append((u, v, edge_key, data))
    return {"op": "restore_nodes", "ids": unique,
            "nodes": [{"id": key, "attrs": deepcopy(attrs)} for key, attrs in nodes],
            "edges": [{"source": u, "target": v, "key": key, "attrs": deepcopy(attrs)} for u, v, key, attrs in edges]}


def _edge_snapshot(graph, source, target, budget: JsonWorkBudget, *, symmetric=False) -> dict:
    pairs = list(dict.fromkeys(((source, target), (target, source)))) if symmetric else [(source, target)]
    edges = [(left, right, key, attrs) for left, right in pairs for key, attrs in (graph.get_edge_data(left, right) or {}).items()]
    budget.reserve({"op": "restore_edges", "source": source, "target": target, "symmetric": symmetric}, items=0)
    for left, right, key, attrs in edges:
        budget.reserve({"source": left, "target": right, "key": key, "attrs": attrs})
    return {"op": "restore_edges", "source": source, "target": target, "symmetric": symmetric,
            "edges": [{"source": left, "target": right, "key": key, "attrs": deepcopy(attrs)} for left, right, key, attrs in edges]}


def _id(value) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError("A valid node or edge ID is required.")
    return value


def apply_op(builder: GraphBuilder, operation: dict, budget: JsonWorkBudget | None = None) -> dict:
    """Apply one operation, returning an inverse containing only affected data."""
    graph = builder.graph
    if budget is None:
        budget = inverse_budget()
    kind = operation.get("op", operation.get("type"))
    if kind == "add_node":
        raw = dict(operation.get("node", operation.get("data", {})))
        raw["name"] = str(raw.get("name") or raw.get("title") or "New note").strip()
        attrs = NodeInput.model_validate(raw).model_dump(exclude_none=True)
        key = _id(raw.get("id") or make_key(attrs["node_type"], attrs["name"]))
        if key in graph:
            raise ValueError("A node with this ID already exists.")
        inverse = _node_snapshot(graph, [key], budget)
        attrs.update(origin="manual", created_at=_now_iso(), updated_at=_now_iso())
        if attrs["node_type"] == "note":
            attrs["title"] = attrs["title"] or attrs["name"]
        graph.add_node(key, **attrs)
        for parent in attrs.get("parent_topic_keys", []):
            builder.add_structural_edge(parent, "HAS_SUBTOPIC" if attrs["node_type"] == "subtopic" else "HAS_CONTENT", key)
        for parent in attrs.get("parent_subtopic_keys", []):
            builder.add_structural_edge(parent, "HAS_CONTENT", key)
        resolve_wikilinks(builder, key)
        return inverse
    if kind in {"update_node", "delete_node"}:
        key = _id(operation.get("id", operation.get("node_id")))
        if key not in graph:
            raise ValueError("Node not found.")
        inverse = _node_snapshot(graph, [key], budget)
        if kind == "delete_node":
            graph.remove_node(key)
        else:
            updates = operation.get("updates", operation.get("patch", operation.get("node", {})))
            merged = {**graph.nodes[key], **updates}
            validated = NodeInput.model_validate(merged).model_dump(exclude_none=True)
            previous = graph.nodes[key]
            if previous.get("node_type") == "note" and (validated.get("node_type") != "note" or
                    (previous.get("title") or previous.get("name"), previous.get("body", "")) !=
                    (validated.get("title") or validated.get("name"), validated.get("body", ""))):
                # Only incident backing is touched, so the node inverse above
                # includes every changed record and work stays degree-bounded.
                remove_association_supports(graph, incident_to=key, invalidated_chunk_ids={note_chunk_id(key)})
            graph.nodes[key].update(validated, updated_at=_now_iso())
            resolve_wikilinks(builder, key)
        return inverse
    if kind in {"add_edge", "delete_edge"}:
        raw = operation.get("edge", operation)
        source, target = raw.get("source"), raw.get("target")
        if kind == "delete_edge" and (not source or not target):
            edge_id = raw.get("id")
            found = next(((u, v, attrs) for u, v, attrs in graph.edges(data=True) if attrs.get("id") == edge_id or
                          (attrs.get("relation") == "RELATED_TO" and association_id(u, v) == edge_id) or
                          f"{u}__{attrs.get('label') or attrs.get('relation')}__{v}" == edge_id), None)
            if found is None:
                raise ValueError("Edge not found.")
            source, target, data = found
            raw = {**raw, "relation": data["relation"], "label": data.get("label")}
        source, target = _id(source), _id(target)
        if source not in graph or target not in graph:
            raise ValueError("Edge endpoints must exist.")
        relation = str(raw.get("relation", "RELATED_TO")).upper()
        if relation not in ALL_RELATIONS:
            raise ValueError("Unknown edge relation.")
        inverse = _edge_snapshot(graph, source, target, budget, symmetric=relation == "RELATED_TO")
        if kind == "delete_edge":
            if not builder.remove_edge(source, relation, target, raw.get("label")):
                raise ValueError("Edge not found.")
        elif relation == "RELATED_TO":
            confidence = float(raw.get("confidence", 1.0))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Edge confidence must be between zero and one.")
            builder.add_related_to(source, target, confidence=confidence, source_chunk_ids=raw.get("source_chunk_ids", []))
        else:
            builder.add_structural_edge(source, relation, target, raw.get("source_chunk_ids", []))
        return inverse
    if kind == "merge_nodes":
        keep, drop = _id(operation.get("keep")), _id(operation.get("drop"))
        # Parent references in other nodes are affected by merge too.
        touched = [keep, drop, *(key for key, attrs in graph.nodes(data=True) if any(drop in attrs.get(field, []) for field in ("parent_topic_keys", "parent_subtopic_keys")))]
        inverse = _node_snapshot(graph, touched, budget)
        builder.merge_nodes(keep, drop)
        resolve_wikilinks(builder, keep)
        return inverse
    if kind == "set_positions":
        positions = operation.get("positions", [])
        if isinstance(positions, dict):
            positions = [{"id": key, **attrs} for key, attrs in positions.items()]
        if not isinstance(positions, list) or len(positions) > 30_000:
            raise ValueError("Invalid positions.")
        inverse = {"op": "set_positions", "positions": []}
        for entry in positions:
            key = _id(entry.get("id"))
            if key not in graph:
                continue
            attrs = graph.nodes[key]
            previous = {"id": key, "x": attrs.get("x"), "y": attrs.get("y")}
            budget.reserve(previous)
            inverse["positions"].append(previous)
            for axis in ("x", "y"):
                value = entry.get(axis)
                if value is None:
                    attrs.pop(axis, None)
                elif isinstance(value, (int, float)) and math.isfinite(value):
                    attrs[axis] = float(value)
                else:
                    raise ValueError("Positions must be finite numbers.")
        return inverse
    if kind == "restore_edges":
        source, target = _id(operation.get("source")), _id(operation.get("target"))
        symmetric = operation.get("symmetric") is True
        pairs = list(dict.fromkeys(((source, target), (target, source)))) if symmetric else [(source, target)]
        inverse = _edge_snapshot(graph, source, target, budget, symmetric=symmetric)
        for left, right in pairs:
            for key in list((graph.get_edge_data(left, right) or {}).keys()):
                graph.remove_edge(left, right, key)
        for entry in operation.get("edges", []):
            left, right = _id(entry.get("source", source)), _id(entry.get("target", target))
            if (left, right) not in pairs or left not in graph or right not in graph or entry["attrs"].get("relation") not in ALL_RELATIONS:
                raise ValueError("Invalid inverse edge.")
            EdgeInput.model_validate(entry["attrs"])
            graph.add_edge(left, right, key=entry["key"], **deepcopy(entry["attrs"]))
        normalize_association_pair(graph, source, target)
        return inverse
    if kind == "restore_nodes":
        ids = [_id(key) for key in operation.get("ids", [])]
        inverse = _node_snapshot(graph, ids, budget)
        graph.remove_nodes_from(ids)
        for entry in operation.get("nodes", []):
            key = _id(entry["id"])
            if key not in ids:
                raise ValueError("Inverse node must belong to affected IDs.")
            NodeInput.model_validate(entry["attrs"])
            graph.add_node(key, **deepcopy(entry["attrs"]))
        for entry in operation.get("edges", []):
            source, target = entry["source"], entry["target"]
            if source not in graph or target not in graph or (source not in ids and target not in ids) or entry["attrs"].get("relation") not in ALL_RELATIONS:
                raise ValueError("Invalid inverse edge.")
            EdgeInput.model_validate(entry["attrs"])
            graph.add_edge(source, target, key=entry["key"], **deepcopy(entry["attrs"]))
        for source, target in {tuple(sorted((entry["source"], entry["target"]))) for entry in operation.get("edges", []) if entry["attrs"].get("relation") == "RELATED_TO"}:
            normalize_association_pair(graph, source, target)
        return inverse
    raise ValueError("Unknown graph operation.")
