"""Canonical, undirected concept associations with independent backing records.

Hierarchy edges remain directed. An association only says the concepts belong
together; its backing records retain why it exists without inventing a verb.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Annotated, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator

SUPPORT_KINDS = frozenset({"automatic", "manual", "wikilink", "legacy"})
MAX_SUPPORTS = 20_000
MAX_SOURCE_IDS = 20_000
MAX_CHUNK_IDENTIFIER_LENGTH = 1024


class AssociationSupport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["automatic", "manual", "wikilink", "legacy"]
    owner: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    source_chunk_ids: list[Annotated[str, Field(min_length=1, max_length=MAX_CHUNK_IDENTIFIER_LENGTH)]] = Field(default_factory=list, max_length=MAX_SOURCE_IDS)
    confidence: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    verified: bool = False

    @model_validator(mode="after")
    def valid_provenance(self):
        if self.kind == "wikilink" and self.owner is None:
            raise ValueError("A note connection requires its owning note.")
        if self.verified and self.kind != "automatic":
            raise ValueError("Only source-checked automatic backing can be verified.")
        if self.verified and not self.source_chunk_ids:
            raise ValueError("Source-checked backing requires supporting passages.")
        return self


def association_pair(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def association_id(source: str, target: str) -> str:
    encoded = json.dumps(list(association_pair(source, target)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "assoc:" + hashlib.sha256(encoded).hexdigest()[:24]


def _source_ids(values: object) -> list[str]:
    if not isinstance(values, list) or len(values) > MAX_SOURCE_IDS:
        raise ValueError("Invalid connection source references.")
    if any(not isinstance(value, str) or not value or len(value) > MAX_CHUNK_IDENTIFIER_LENGTH for value in values):
        raise ValueError("Invalid connection source references.")
    return sorted(set(values))


def _support(record: dict) -> dict:
    validated = AssociationSupport.model_validate(record).model_dump()
    validated["source_chunk_ids"] = _source_ids(validated["source_chunk_ids"])
    return validated


def _merge_supports(records: Iterable[dict]) -> list[dict]:
    merged = {}
    for raw in records:
        record = _support(raw)
        key = (record["kind"], record["owner"] or "", tuple(record["source_chunk_ids"]), record["verified"])
        previous = merged.get(key)
        if previous is None:
            if len(merged) >= MAX_SUPPORTS:
                raise ValueError("Connection backing limit exceeded.")
            merged[key] = record
        else:
            previous["confidence"] = max(previous["confidence"], record["confidence"])
    return [merged[key] for key in sorted(merged)]


def _edge_supports(graph: nx.MultiDiGraph, source: str, target: str, attrs: dict) -> list[dict]:
    if "supports" in attrs:
        records = attrs["supports"]
        if not isinstance(records, list) or not records or len(records) > MAX_SUPPORTS:
            raise ValueError("Invalid connection backing records.")
        return records
    kind = attrs.get("origin") if attrs.get("origin") in SUPPORT_KINDS else "legacy"
    owner = None
    sources = list(attrs.get("source_chunk_ids", []))
    if attrs.get("wikilink"):
        kind = "wikilink"
        owner = next((key for key in (source, target) if graph.nodes[key].get("node_type") == "note"), source)
        if not sources:
            sources = ["note:" + owner]
    # Old free-form labels never imply that a source-context check occurred.
    return [{"kind": kind, "owner": owner, "source_chunk_ids": sources,
             "confidence": attrs.get("confidence") if attrs.get("confidence") is not None else 0.5, "verified": False}]


def association_edges(graph: nx.MultiDiGraph, source: str, target: str) -> list[tuple[str, str, object, dict]]:
    result = []
    for left, right in dict.fromkeys(((source, target), (target, source))):
        for key, attrs in (graph.get_edge_data(left, right) or {}).items():
            if attrs.get("relation") == "RELATED_TO":
                result.append((left, right, key, attrs))
    return result


def _attributes(source: str, target: str, records: Iterable[dict], extra: dict | None = None) -> dict:
    supports = _merge_supports(records)
    if not supports:
        raise ValueError("A connection requires at least one backing record.")
    if any(record["kind"] == "wikilink" and record["owner"] not in {source, target} for record in supports):
        raise ValueError("A note connection must include its owning note.")
    attributes = {key: deepcopy(value) for key, value in (extra or {}).items()
                  if key not in {"id", "relation", "label", "confidence", "source_chunk_ids", "supports", "origin", "wikilink", "verified"}}
    attributes.update(id=association_id(source, target), relation="RELATED_TO", label=None, supports=supports,
                      confidence=max(record["confidence"] for record in supports),
                      source_chunk_ids=sorted({value for record in supports for value in record["source_chunk_ids"]}))
    return attributes


def _normalize_pair(graph: nx.MultiDiGraph, source: str, target: str, edges=None) -> bool:
    pair = association_pair(source, target)
    edges = association_edges(graph, *pair) if edges is None else edges
    if not edges:
        return False
    records = [record for left, right, _, attrs in edges for record in _edge_supports(graph, left, right, attrs)]
    attrs = _attributes(*pair, records, extra=edges[0][3])
    key = association_id(*pair)
    if len(edges) == 1 and edges[0][:3] == (*pair, key) and edges[0][3] == attrs:
        return False
    for left, right, edge_key, _ in edges:
        graph.remove_edge(left, right, edge_key)
    graph.add_edge(*pair, key=key, **attrs)
    return True


def normalize_associations(graph: nx.MultiDiGraph) -> int:
    """Idempotently migrate legacy/reversed semantic edges; never call a model."""
    groups = {}
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        if attrs.get("relation") == "RELATED_TO":
            groups.setdefault(association_pair(source, target), []).append((source, target, key, attrs))
    return sum(_normalize_pair(graph, *pair, edges) for pair, edges in groups.items())


def normalize_association_pair(graph: nx.MultiDiGraph, source: str, target: str) -> bool:
    return _normalize_pair(graph, source, target)


def add_association(graph: nx.MultiDiGraph, source: str, target: str, *, kind: str = "manual", owner: str | None = None,
                    source_chunk_ids: list[str] | None = None, confidence: float = 1.0, verified: bool = False) -> bool:
    if source not in graph or target not in graph:
        raise ValueError("Connection endpoints must exist.")
    if kind == "automatic" and source == target:
        return False
    pair = association_pair(source, target)
    edges = association_edges(graph, *pair)
    records = [record for left, right, _, attrs in edges for record in _edge_supports(graph, left, right, attrs)]
    records.append({"kind": kind, "owner": owner, "source_chunk_ids": list(source_chunk_ids or []),
                    "confidence": confidence, "verified": verified})
    attrs = _attributes(*pair, records, extra=edges[0][3] if edges else None)
    for left, right, key, _ in edges:
        graph.remove_edge(left, right, key)
    graph.add_edge(*pair, key=association_id(*pair), **attrs)
    return not edges


def remove_association(graph: nx.MultiDiGraph, source: str, target: str) -> bool:
    edges = association_edges(graph, source, target)
    for left, right, key, _ in edges:
        graph.remove_edge(left, right, key)
    return bool(edges)


def remove_association_supports(graph: nx.MultiDiGraph, *, kind: str | None = None, owner: str | None = None,
                                removed_chunk_ids: set[str] | None = None,
                                invalidated_chunk_ids: set[str] | None = None, incident_to: str | None = None) -> int:
    """Remove only matching provenance, leaving independently backed links."""
    changed = 0
    removed = set(removed_chunk_ids or ())
    invalidated = set(invalidated_chunk_ids or ())
    scoped_node = incident_to if incident_to is not None else owner if kind == "wikilink" else None
    if scoped_node is not None:
        incident = [*graph.in_edges(scoped_node, keys=True, data=True), *graph.out_edges(scoped_node, keys=True, data=True)] if scoped_node in graph else []
        candidates = {(left, right, key): attrs for left, right, key, attrs in incident}
    else:
        candidates = {(left, right, key): attrs for left, right, key, attrs in graph.edges(keys=True, data=True)}
    for (source, target, key), attrs in candidates.items():
        if attrs.get("relation") != "RELATED_TO":
            continue
        records = _merge_supports(_edge_supports(graph, source, target, attrs))
        retained = []
        for record in records:
            if (kind is not None or owner is not None) and (kind is None or record["kind"] == kind) and (owner is None or record["owner"] == owner):
                continue
            sources = set(record["source_chunk_ids"])
            if record["verified"] and sources & invalidated:
                continue
            if sources & removed:
                if record["kind"] == "wikilink" or (record["kind"] == "automatic" and record["verified"]):
                    continue
                record = {**record, "source_chunk_ids": sorted(sources - removed)}
                if record["kind"] != "manual" and not record["source_chunk_ids"]:
                    continue
            retained.append(record)
        if retained == records:
            continue
        changed += 1
        graph.remove_edge(source, target, key)
        if retained:
            pair = association_pair(source, target)
            graph.add_edge(*pair, key=association_id(*pair), **_attributes(*pair, retained, extra=attrs))
    return changed


def prune_association_supports(graph: nx.MultiDiGraph, removed_chunk_ids: set[str]) -> int:
    return remove_association_supports(graph, removed_chunk_ids=removed_chunk_ids)
