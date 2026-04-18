"""
Tests for backend/retrieval/graph_retriever.py

Uses a mock Embedder with controlled cosine scores so tests are
deterministic and don't require model inference.

The mock embedder assigns unit vectors such that:
  - "relevant" nodes have high cosine similarity with the query
  - "irrelevant" nodes have low (or negative) cosine similarity

Cosine = dot product of L2-normalised vectors, so we just use
orthogonal / aligned unit vectors as embeddings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval.graph_retriever import _bfs, _find_anchors, retrieve_graph
from backend.schemas import Chunk, GraphTriple, RetrievalResult


# ---------------------------------------------------------------------------
# Mock Embedder factory
# ---------------------------------------------------------------------------

def _mock_embedder(embedding_map: dict[str, np.ndarray]) -> MagicMock:
    """
    Returns a mock Embedder whose embed/embed_one return vectors from
    embedding_map by matching the input text.

    Unrecognised texts return a zero vector (cosine = 0 with everything).
    """
    dim = next(iter(embedding_map.values())).shape[0]
    zero = np.zeros(dim, dtype=np.float32)

    def _embed(texts: list[str]) -> np.ndarray:
        return np.stack([embedding_map.get(t, zero) for t in texts])

    def _embed_one(text: str) -> np.ndarray:
        return embedding_map.get(text, zero)

    mock = MagicMock()
    mock.embed.side_effect = _embed
    mock.embed_one.side_effect = _embed_one
    mock.dimension = dim
    return mock


def _unit(index: int, dim: int = 8) -> np.ndarray:
    """Return a unit vector with a 1 at position `index`."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph(tmp_path: Path) -> GraphBuilder:
    """
    Graph:
        enzyme --CAUSES--> reaction_rate
        reaction_rate --EXPLAINS--> kinetics
        unrelated_node  (isolated concept with low relevance)
    """
    builder = GraphBuilder(tmp_path / "graph.json")
    builder.add_triples([
        GraphTriple(
            head="Enzyme", relation="CAUSES", tail="ReactionRate",
            source_chunk_ids=["chunk_enzyme"]
        ),
        GraphTriple(
            head="ReactionRate", relation="EXPLAINS", tail="Kinetics",
            source_chunk_ids=["chunk_rate"]
        ),
        GraphTriple(
            head="UnrelatedNode", relation="PART_OF", tail="SomeTopic",
            source_chunk_ids=["chunk_unrelated"]
        ),
    ])
    return builder


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    chunks = [
        Chunk(id="chunk_enzyme", text="Enzymes catalyse reactions.", source_id="s", modality="pdf"),
        Chunk(id="chunk_rate", text="Reaction rate depends on enzyme concentration.", source_id="s", modality="pdf"),
        Chunk(id="chunk_unrelated", text="Unrelated content.", source_id="s", modality="pdf"),
    ]
    with DocumentStore(tmp_path / "store.db") as s:
        s.save_chunks(chunks)
        yield s


@pytest.fixture
def embedder(simple_graph) -> MagicMock:
    """
    Assigns:
      query       → e0  (index 0)
      "Enzyme"    → e0  (cosine 1.0 with query — anchor by embedding)
      "ReactionRate" → e0 * 0.9 + e1 * small  (relevant, score > 0.4)
      "Kinetics"  → e0 * 0.6  (relevant, score > 0.4)
      "UnrelatedNode" → e2   (cosine 0.0 with query — irrelevant)
      "SomeTopic" → e2   (cosine 0.0 — irrelevant)
    """
    q = _unit(0)
    relevant_high = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    relevant_high /= np.linalg.norm(relevant_high)
    relevant_mid = np.array([0.7, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    relevant_mid /= np.linalg.norm(relevant_mid)

    mapping = {
        # query and anchor
        "enzyme kinetics query": q,
        "Enzyme": q,                    # string match will catch this
        # neighbours
        "ReactionRate": relevant_high,  # score ~0.95 > threshold
        "Kinetics": relevant_mid,       # score ~0.70 > threshold
        # irrelevant branch
        "UnrelatedNode": _unit(2),      # score 0.0 < threshold
        "SomeTopic": _unit(2),          # score 0.0 < threshold
    }
    return _mock_embedder(mapping)


@pytest.fixture
def base_config() -> dict:
    return {
        "graph_retriever": {
            "max_hops": 3,
            "relevance_threshold": 0.4,
            "max_nodes_per_hop": 10,
            "top_k_anchors": 2,
        }
    }


# ---------------------------------------------------------------------------
# _find_anchors
# ---------------------------------------------------------------------------

class TestFindAnchors:
    def test_string_match_finds_anchor(self, simple_graph, embedder):
        # "Enzyme" appears in the query string
        anchors = _find_anchors(
            "Enzyme kinetics query", simple_graph.graph, embedder, top_k=2
        )
        assert "enzyme" in anchors

    def test_embedding_fallback_used_when_no_string_match(self, simple_graph, embedder):
        # Query mentions no node names directly
        anchors = _find_anchors(
            "biochemical catalysis", simple_graph.graph, embedder, top_k=1
        )
        assert len(anchors) >= 1

    def test_top_k_respected(self, simple_graph, embedder):
        anchors = _find_anchors(
            "enzyme kinetics query", simple_graph.graph, embedder, top_k=1
        )
        assert len(anchors) <= 1

    def test_empty_graph_returns_empty(self, tmp_path, embedder):
        empty = GraphBuilder(tmp_path / "empty.json")
        anchors = _find_anchors("anything", empty.graph, embedder, top_k=3)
        assert anchors == []


# ---------------------------------------------------------------------------
# _bfs
# ---------------------------------------------------------------------------

class TestBfs:
    def _run(self, builder, embedder, cfg):
        graph = builder.graph
        q_emb = embedder.embed_one("enzyme kinetics query")
        anchors = ["enzyme"]
        return _bfs(
            anchors, graph, q_emb, embedder,
            max_hops=cfg["max_hops"],
            threshold=cfg["relevance_threshold"],
            max_per_hop=cfg["max_nodes_per_hop"],
        )

    def test_anchor_always_in_subgraph(self, simple_graph, embedder, base_config):
        nodes, _ = self._run(simple_graph, embedder, base_config["graph_retriever"])
        assert "enzyme" in nodes

    def test_relevant_neighbour_expanded(self, simple_graph, embedder, base_config):
        nodes, _ = self._run(simple_graph, embedder, base_config["graph_retriever"])
        assert "reactionrate" in nodes

    def test_irrelevant_neighbour_pruned(self, simple_graph, embedder, base_config):
        nodes, _ = self._run(simple_graph, embedder, base_config["graph_retriever"])
        assert "unrelatednode" not in nodes

    def test_deep_relevant_node_reached(self, simple_graph, embedder, base_config):
        nodes, _ = self._run(simple_graph, embedder, base_config["graph_retriever"])
        assert "kinetics" in nodes

    def test_max_hops_zero_returns_only_anchors(self, simple_graph, embedder, base_config):
        cfg = {**base_config["graph_retriever"], "max_hops": 0}
        nodes, edges = self._run(simple_graph, embedder, cfg)
        assert nodes == {"enzyme"}
        assert edges == []

    def test_max_nodes_per_hop_limits_expansion(self, simple_graph, embedder, base_config):
        cfg = {**base_config["graph_retriever"], "max_nodes_per_hop": 1}
        nodes, _ = self._run(simple_graph, embedder, cfg)
        # Only 1 neighbour allowed per hop; expansion is bounded
        assert len(nodes) <= 1 + 1 + 1  # anchor + 1/hop × 2 hops max

    def test_edges_connect_known_nodes(self, simple_graph, embedder, base_config):
        nodes, edges = self._run(simple_graph, embedder, base_config["graph_retriever"])
        for u, v, _ in edges:
            assert u in nodes
            assert v in nodes


# ---------------------------------------------------------------------------
# retrieve_graph (full integration)
# ---------------------------------------------------------------------------

class TestRetrieveGraph:
    def test_returns_retrieval_result(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        assert isinstance(result, RetrievalResult)

    def test_routing_mode_is_graph(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        assert result.routing_mode == "graph"

    def test_subgraph_contains_graph_triples(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        assert all(isinstance(t, GraphTriple) for t in result.subgraph)

    def test_chunks_fetched_from_store(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        assert all(isinstance(c, Chunk) for c in result.chunks)

    def test_relevant_chunks_returned(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        ids = {c.id for c in result.chunks}
        assert "chunk_enzyme" in ids

    def test_irrelevant_chunks_excluded(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        ids = {c.id for c in result.chunks}
        assert "chunk_unrelated" not in ids

    def test_empty_graph_returns_empty_result(self, tmp_path, store, embedder, base_config):
        empty = GraphBuilder(tmp_path / "empty.json")
        result = retrieve_graph("any query", empty, store, embedder, base_config)
        assert result.chunks == []
        assert result.subgraph == []

    def test_subgraph_edges_have_relation(self, simple_graph, store, embedder, base_config):
        result = retrieve_graph(
            "enzyme kinetics query", simple_graph, store, embedder, base_config
        )
        assert all(t.relation != "" for t in result.subgraph)
