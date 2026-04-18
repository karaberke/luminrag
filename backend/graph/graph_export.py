"""
Stage 2 - Concept Graph Construction: graph_export.py

Exports a NetworkX MultiDiGraph to a D3 / Cytoscape-ready JSON format.
This JSON is what the frontend mindmap visualization consumes.

Output structure:
{
  "nodes": [
    {"id": "enzyme", "name": "Enzyme", "source_ids": ["chunk_0", "chunk_1"]}
  ],
  "edges": [
    {
      "source": "enzyme",
      "target": "reaction_rate",
      "relation": "CAUSES",
      "source_chunk_ids": ["chunk_0"]
    }
  ]
}

Node "id" is the normalised key used internally by the graph.
Node "name" is the original display-cased name.

Public API:
    result = export_graph(graph, output_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def export_graph(graph: nx.MultiDiGraph, output_path: str | Path) -> dict:
    """
    Serialise the concept graph to a visualisation-ready JSON file.

    Args:
        graph:       The NetworkX MultiDiGraph from GraphBuilder.
        output_path: Destination JSON file path.

    Returns:
        The exported dict (same content as the written file).
    """
    nodes = [
        {
            "id": node_id,
            "name": attrs.get("name", node_id),
            "source_ids": attrs.get("source_ids", []),
            "node_type": attrs.get("node_type", "concept"),
        }
        for node_id, attrs in graph.nodes(data=True)
    ]

    edges = [
        {
            "source": u,
            "target": v,
            "relation": attrs.get("relation", ""),
            "source_chunk_ids": attrs.get("source_chunk_ids", []),
        }
        for u, v, attrs in graph.edges(data=True)
    ]

    result = {"nodes": nodes, "edges": edges}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Graph exported to {output_path} "
        f"({len(nodes)} nodes, {len(edges)} edges)"
    )
    return result
