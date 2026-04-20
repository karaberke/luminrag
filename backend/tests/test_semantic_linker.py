"""
Tests for backend/graph/semantic_linker.py — Stage 1.5.

Uses a substring-based mock embedder so candidate pair discovery is
deterministic. LLM is patched out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from backend.graph.semantic_linker import (
    LinkableNode,
    build_linkables_from_graph,
    link_nodes,
)


def _embedder(pairs_high: list[tuple[str, str]]) -> MagicMock:
    """
    Mock embedder: returns (1,0,0) for any text whose presence-set matches
    one side of a pair in `pairs_high`, and (cos 0.6 with first vec) for the
    other side. This engineers candidate similarity in-range for the linker.
    """
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mid = np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
    mid = mid / np.linalg.norm(mid)
    far = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    related_terms: set[str] = set()
    for a, b in pairs_high:
        related_terms.add(a.lower())
        related_terms.add(b.lower())

    def _vec(text: str) -> np.ndarray:
        t = text.lower()
        if any(term in t for term in related_terms):
            # First related term gets `base`, rest get `mid` so cosine ~ 0.7.
            marker = next(term for term in related_terms if term in t)
            if marker == sorted(related_terms)[0]:
                return base
            return mid
        return far

    mock = MagicMock()
    mock.embed.side_effect = lambda texts: np.stack([_vec(t) for t in texts])
    mock.embed_one.side_effect = _vec
    mock.dimension = 4
    return mock


@pytest.fixture
def config() -> dict:
    return {
        "semantic_linker": {
            "sim_lo": 0.55,
            "sim_hi": 0.98,
            "max_candidate_pairs": 80,
            "batch_size": 5,
            "min_confidence": 0.3,
            "max_tokens": 256,
        }
    }


class TestLinkNodes:
    def test_no_pairs_returns_empty(self, config):
        emb = _embedder([])
        # Only one node — no pairs possible
        nodes: list[LinkableNode] = [
            {"key": "topic:a", "name": "NLP", "text": "NLP", "node_type": "topic"},
        ]
        assert link_nodes(nodes, emb, config) == []

    def test_out_of_range_pairs_filtered(self, config):
        # Two unrelated nodes — embedder returns orthogonal vectors → sim ≈ 0
        emb = _embedder([])
        nodes: list[LinkableNode] = [
            {"key": "t1", "name": "X", "text": "X", "node_type": "topic"},
            {"key": "t2", "name": "Y", "text": "Y", "node_type": "topic"},
        ]
        with patch("backend.graph.semantic_linker.call_llm") as mock_llm:
            result = link_nodes(nodes, emb, config)
        mock_llm.assert_not_called()  # no candidates → no LLM calls
        assert result == []

    def test_in_range_pair_sent_to_llm(self, config):
        emb = _embedder([("nlp", "graph theory")])
        nodes: list[LinkableNode] = [
            {"key": "t1", "name": "NLP", "text": "NLP field",
             "node_type": "topic"},
            {"key": "t2", "name": "Graph Theory", "text": "Graph Theory math",
             "node_type": "topic"},
        ]
        with patch("backend.graph.semantic_linker.call_llm") as mock_llm:
            mock_llm.return_value = (
                '[{"index": 0, "label": "uses", "confidence": 0.8}]'
            )
            proposals = link_nodes(nodes, emb, config)
        assert len(proposals) == 1
        assert proposals[0].source_key in {"t1", "t2"}
        assert proposals[0].target_key in {"t1", "t2"}
        assert proposals[0].label == "uses"
        assert proposals[0].confidence == 0.8

    def test_low_confidence_dropped(self, config):
        emb = _embedder([("nlp", "graph theory")])
        nodes: list[LinkableNode] = [
            {"key": "t1", "name": "NLP", "text": "NLP", "node_type": "topic"},
            {"key": "t2", "name": "Graph Theory", "text": "Graph Theory",
             "node_type": "topic"},
        ]
        with patch("backend.graph.semantic_linker.call_llm") as mock_llm:
            mock_llm.return_value = (
                '[{"index": 0, "label": "uses", "confidence": 0.1}]'
            )
            proposals = link_nodes(nodes, emb, config)
        assert proposals == []

    def test_llm_failure_swallowed(self, config):
        emb = _embedder([("nlp", "graph theory")])
        nodes: list[LinkableNode] = [
            {"key": "t1", "name": "NLP", "text": "NLP", "node_type": "topic"},
            {"key": "t2", "name": "Graph Theory", "text": "Graph Theory",
             "node_type": "topic"},
        ]
        with patch("backend.graph.semantic_linker.call_llm", side_effect=Exception("boom")):
            proposals = link_nodes(nodes, emb, config)
        assert proposals == []


class TestBuildLinkablesFromGraph:
    def test_excludes_chunk_refs_by_default(self):
        g = nx.MultiDiGraph()
        g.add_node("topic:a", name="T", node_type="topic", summary="s")
        g.add_node("c1", name="chunk", node_type="chunk_ref")
        linkables = build_linkables_from_graph(g)
        assert [l["key"] for l in linkables] == ["topic:a"]

    def test_content_uses_summary_intermediate(self):
        g = nx.MultiDiGraph()
        g.add_node("content:a", name="C", node_type="content",
                   summary_intermediate="mid")
        linkables = build_linkables_from_graph(g)
        assert "mid" in linkables[0]["text"]

    def test_node_types_filter(self):
        g = nx.MultiDiGraph()
        g.add_node("topic:a", name="T", node_type="topic")
        g.add_node("subtopic:b", name="S", node_type="subtopic")
        g.add_node("content:c", name="C", node_type="content")
        linkables = build_linkables_from_graph(g, node_types=["topic"])
        assert [l["key"] for l in linkables] == ["topic:a"]
