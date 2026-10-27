"""
Stage 2 — Concept Graph: graph_export.py

Serialises the MultiDiGraph to a D3/Cytoscape-ready JSON the frontend
mindmap consumes. Emits node_type on every node and distinguishes the
directed hierarchy from plain, symmetric RELATED_TO concept connections.

Output structure:
{
  "nodes": [
    {
      "id": "topic:591566947860",
      "name": "Graph Theory",
      "node_type": "topic",
      "source_ids": ["chunk_0"],
      "summary": "…",
      "scope": "broad",
      "illustration": null,
      "content_type": null,
      "parent_topic_keys": [],
      "parent_subtopic_keys": [],
      "origin": "ingested",
      "created_at": "2026-04-18T…",
      "updated_at": "2026-04-18T…"
    }
  ],
  "edges": [
    {
      "source": "topic:591566947860",
      "target": "subtopic:…",
      "relation": "HAS_SUBTOPIC",
      "label": null,
      "confidence": null,
      "source_chunk_ids": []
    }
  ]
}

Public API:
    result = export_graph(graph, output_path)
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import networkx as nx

from backend.utils.files import atomic_write_text
from backend.utils.associations import association_id, normalize_associations

logger = logging.getLogger(__name__)


_NODE_FIELDS = (
    "title",
    "body",
    "x",
    "y",
    "origin",
    "created_at",
    "updated_at",
    "summary",
    "scope",
    "illustration",
    "illustration_path",
    "content_type",
    "raw_excerpt",
    "key_terms",
    "parent_topic_keys",
    "parent_subtopic_keys",
)


def _node_payload(node_id: str, attrs: dict) -> dict:
    node_type = attrs.get("node_type", "content")
    payload = {
        "id": node_id,
        "name": attrs.get("name", node_id),
        "node_type": node_type,
        "source_ids": list(attrs.get("source_ids", [])),
    }
    for field in _NODE_FIELDS:
        if field in attrs:
            payload[field] = attrs[field]
    return payload


def _edge_payload(u: str, v: str, attrs: dict) -> dict:
    payload = {
        "source": u,
        "target": v,
        "relation": attrs.get("relation", ""),
        "source_chunk_ids": list(attrs.get("source_chunk_ids", [])),
    }
    if attrs.get("relation") == "RELATED_TO":
        payload["id"] = association_id(u, v)
        payload["label"] = None
        payload["confidence"] = attrs.get("confidence")
        payload["supports"] = deepcopy(attrs.get("supports", []))
    return payload


def export_graph(graph: nx.MultiDiGraph, output_path: str | Path) -> dict:
    """
    Serialise the concept graph to a visualisation-ready JSON file.
    Skips chunk_ref nodes (legacy) if any remain in the graph.

    Args:
        graph:       The NetworkX MultiDiGraph from GraphBuilder.
        output_path: Destination JSON file path.

    Returns:
        The exported dict (same content as the written file).
    """
    normalize_associations(graph)
    nodes = [
        _node_payload(n, a)
        for n, a in graph.nodes(data=True)
        if a.get("node_type") != "chunk_ref"
    ]
    node_ids = {n["id"] for n in nodes}
    edges = [
        _edge_payload(u, v, a)
        for u, v, a in graph.edges(data=True)
        if u in node_ids and v in node_ids
    ]

    result = {"association_version": 1, "nodes": nodes, "edges": edges}

    output_path = Path(output_path)
    atomic_write_text(output_path, json.dumps(result, indent=2))

    logger.info(
        f"Graph exported to {output_path} "
        f"({len(nodes)} nodes, {len(edges)} edges)"
    )
    return result
