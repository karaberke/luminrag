"""
Tests for backend/retrieval/graph_retriever.py

Uses a substring-based mock Embedder so scoring is deterministic and
independent of any real model. The graph shape follows the hierarchical
schema: Topic → Subtopic → Content (no ChunkRef nodes).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import (
    ContentProposal,
    ProposalBundle,
    SubtopicProposal,
    TopicProposal,
    make_key,
)
from backend.retrieval.graph_retriever import (
    _bfs,
    _find_anchors,
    retrieve_graph,
)
from backend.schemas import Chunk, GraphTriple, RetrievalResult


# ---------------------------------------------------------------------------
# Mock Embedder — keyword-based scoring
# ---------------------------------------------------------------------------

def _mock_embedder(relevant_keywords: list[str]) -> MagicMock:
    dim = 4
    relevant_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    irrelevant_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def _vec(text: str) -> np.ndarray:
        lower = text.lower()
        return relevant_vec if any(kw in lower for kw in relevant_keywords) else irrelevant_vec

    def _embed(texts: list[str]) -> np.ndarray:
        return np.stack([_vec(t) for t in texts])

    def _embed_one(text: str) -> np.ndarray:
        return _vec(text)

    mock = MagicMock()
    mock.embed.side_effect = _embed
    mock.embed_one.side_effect = _embed_one
    mock.dimension = dim
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hierarchical_graph(tmp_path: Path) -> GraphBuilder:
    """
    Biochemistry (topic)
    └─ Enzyme Kinetics (subtopic)
       ├─ Michaelis-Menten (content, chunk c1)
       └─ Rate Law (content, chunk c2)

    Nuclear Physics (topic)
    └─ Fusion Physics (subtopic)
       └─ Fusion Equation (content, chunk c3)
    """
    builder = GraphBuilder(tmp_path / "graph.json")
    bundle = ProposalBundle(
        topics=[
            TopicProposal(name="Biochemistry", summary="study of enzymes",
                          source_chunk_ids=["c1", "c2"]),
            TopicProposal(name="Nuclear Physics", summary="study of atoms",
                          source_chunk_ids=["c3"]),
        ],
        subtopics=[
            SubtopicProposal(name="Enzyme Kinetics",
                             summary="rate of enzyme-catalysed reactions",
                             parent_topic_names=["Biochemistry"],
                             source_chunk_ids=["c1", "c2"]),
            SubtopicProposal(name="Fusion Physics",
                             summary="nuclear fusion reactions",
                             parent_topic_names=["Nuclear Physics"],
                             source_chunk_ids=["c3"]),
        ],
        contents=[
            ContentProposal(title="Michaelis-Menten",
                            summary="rate law for enzymes",
                            parent_subtopic_names=["Enzyme Kinetics"],
                            parent_topic_names=["Biochemistry"],
                            evidence_chunk_ids=["c1"]),
            ContentProposal(title="Rate Law",
                            summary="general reaction rate",
                            parent_subtopic_names=["Enzyme Kinetics"],
                            parent_topic_names=["Biochemistry"],
                            evidence_chunk_ids=["c2"]),
            ContentProposal(title="Fusion Equation",
                            summary="energy released in fusion",
                            parent_subtopic_names=["Fusion Physics"],
                            parent_topic_names=["Nuclear Physics"],
                            evidence_chunk_ids=["c3"]),
        ],
    )
    builder.apply_proposals(bundle)
    return builder


@pytest.fixture
def store(tmp_path: Path):
    chunks = [
        Chunk(id="c1", text="Enzymes catalyse reactions.", source_id="bio.pdf", modality="pdf"),
        Chunk(id="c2", text="Rate law for enzyme reactions.", source_id="bio.pdf", modality="pdf"),
        Chunk(id="c3", text="Fusion reactions in stars.", source_id="nuke.pdf", modality="pdf"),
    ]
    with DocumentStore(tmp_path / "store.db") as s:
        s.save_chunks(chunks)
        yield s


@pytest.fixture
def enzyme_embedder() -> MagicMock:
    return _mock_embedder(["enzyme", "kinetics", "michaelis", "rate", "biochem"])


@pytest.fixture
def base_config() -> dict:
    return {
        "graph_retriever": {
            "max_hops": 3,
            "relevance_threshold": 0.4,
            "max_nodes_per_hop": 10,
            "top_k_anchors": 3,
        }
    }


# ---------------------------------------------------------------------------
# _find_anchors
# ---------------------------------------------------------------------------

class TestFindAnchors:
    def test_string_match_finds_anchor(self, hierarchical_graph, enzyme_embedder):
        anchors = _find_anchors(
            "what is michaelis-menten kinetics",
            hierarchical_graph.graph, enzyme_embedder, top_k=3,
        )
        assert make_key("content", "Michaelis-Menten") in anchors

    def test_embedding_fallback(self, hierarchical_graph, enzyme_embedder):
        anchors = _find_anchors(
            "enzyme catalysis",
            hierarchical_graph.graph, enzyme_embedder, top_k=2,
        )
        assert len(anchors) >= 1
        names = [hierarchical_graph.graph.nodes[k]["name"].lower() for k in anchors]
        assert any(
            any(kw in n for kw in ("enzyme", "kinetics", "michaelis", "rate", "biochem"))
            for n in names
        )

    def test_chunk_refs_excluded_from_anchors(self, hierarchical_graph, enzyme_embedder):
        anchors = _find_anchors(
            "enzymes", hierarchical_graph.graph, enzyme_embedder, top_k=10,
        )
        for k in anchors:
            assert hierarchical_graph.graph.nodes[k].get("node_type") != "chunk_ref"

    def test_empty_graph_returns_empty(self, tmp_path, enzyme_embedder):
        empty = GraphBuilder(tmp_path / "empty.json")
        anchors = _find_anchors("anything", empty.graph, enzyme_embedder, top_k=3)
        assert anchors == []

    def test_key_term_string_hit_anchors(self, hierarchical_graph, enzyme_embedder):
        # Inject a key_term on a content node whose display name does NOT
        # contain the searched term. Only the key_term match should pull it in.
        ck = make_key("content", "Rate Law")
        hierarchical_graph.graph.nodes[ck]["key_terms"] = ["Vmax"]
        # Query is scoped into Biochemistry via the "enzyme" keyword so Rate
        # Law is in the candidate pool; "vmax" then matches via key_terms.
        anchors = _find_anchors(
            "enzyme vmax", hierarchical_graph.graph, enzyme_embedder, top_k=3,
        )
        assert ck in anchors

    def test_key_term_embedding_boost_flips_ranking(self, hierarchical_graph):
        # Embedder that rates everything as equally-irrelevant. The key_term
        # boost should be the only signal differentiating candidates.
        flat = MagicMock()
        flat.embed.side_effect = lambda texts: np.stack(
            [np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)] * len(texts)
        )
        flat.embed_one.side_effect = lambda _: np.array(
            [0.5, 0.5, 0.0, 0.0], dtype=np.float32
        )
        flat.dimension = 4

        ck = make_key("content", "Rate Law")
        hierarchical_graph.graph.nodes[ck]["key_terms"] = ["km"]

        anchors = _find_anchors(
            "describe km", hierarchical_graph.graph, flat, top_k=1,
            key_term_boost=0.5,
        )
        assert anchors[0] == ck


# ---------------------------------------------------------------------------
# _bfs — relevance-guided expansion
# ---------------------------------------------------------------------------

class TestBfs:
    def _run(self, builder, embedder, anchors, cfg):
        q_emb = embedder.embed_one("enzymes")
        return _bfs(
            anchors, builder.graph, q_emb, embedder,
            max_hops=cfg["max_hops"],
            threshold=cfg["relevance_threshold"],
            max_per_hop=cfg["max_nodes_per_hop"],
        )

    def test_anchor_always_in_subgraph(self, hierarchical_graph, enzyme_embedder, base_config):
        anchor = make_key("subtopic", "Enzyme Kinetics")
        nodes, _ = self._run(hierarchical_graph, enzyme_embedder, [anchor],
                             base_config["graph_retriever"])
        assert anchor in nodes

    def test_relevant_neighbours_expanded(self, hierarchical_graph, enzyme_embedder, base_config):
        anchor = make_key("subtopic", "Enzyme Kinetics")
        nodes, _ = self._run(hierarchical_graph, enzyme_embedder, [anchor],
                             base_config["graph_retriever"])
        assert make_key("content", "Michaelis-Menten") in nodes
        assert make_key("content", "Rate Law") in nodes

    def test_irrelevant_branch_pruned(self, hierarchical_graph, enzyme_embedder, base_config):
        anchor = make_key("subtopic", "Enzyme Kinetics")
        nodes, _ = self._run(hierarchical_graph, enzyme_embedder, [anchor],
                             base_config["graph_retriever"])
        assert make_key("content", "Fusion Equation") not in nodes

    def test_chunk_refs_not_in_results(self, hierarchical_graph, enzyme_embedder, base_config):
        anchor = make_key("content", "Michaelis-Menten")
        nodes, _ = self._run(hierarchical_graph, enzyme_embedder, [anchor],
                             base_config["graph_retriever"])
        for k in nodes:
            assert hierarchical_graph.graph.nodes[k].get("node_type") != "chunk_ref"

    def test_max_hops_zero_returns_only_anchors(self, hierarchical_graph, enzyme_embedder, base_config):
        anchor = make_key("subtopic", "Enzyme Kinetics")
        cfg = {**base_config["graph_retriever"], "max_hops": 0}
        nodes, edges = self._run(hierarchical_graph, enzyme_embedder, [anchor], cfg)
        assert nodes == {anchor}
        assert edges == []


# ---------------------------------------------------------------------------
# retrieve_graph — full integration
# ---------------------------------------------------------------------------

class TestRetrieveGraph:
    def test_returns_retrieval_result(self, hierarchical_graph, store, enzyme_embedder, base_config):
        result = retrieve_graph("enzyme kinetics", hierarchical_graph, store,
                                enzyme_embedder, base_config)
        assert isinstance(result, RetrievalResult)

    def test_routing_mode_is_graph(self, hierarchical_graph, store, enzyme_embedder, base_config):
        result = retrieve_graph("enzyme kinetics", hierarchical_graph, store,
                                enzyme_embedder, base_config)
        assert result.routing_mode == "graph"

    def test_subgraph_is_list_of_graph_triples(self, hierarchical_graph, store, enzyme_embedder, base_config):
        result = retrieve_graph("enzyme kinetics", hierarchical_graph, store,
                                enzyme_embedder, base_config)
        assert all(isinstance(t, GraphTriple) for t in result.subgraph)

    def test_subgraph_node_keys_populated(self, hierarchical_graph, store, enzyme_embedder, base_config):
        result = retrieve_graph("enzyme kinetics", hierarchical_graph, store,
                                enzyme_embedder, base_config)
        assert len(result.subgraph_node_keys) > 0

    def test_relevant_chunks_fetched(self, hierarchical_graph, store, enzyme_embedder, base_config):
        result = retrieve_graph("enzyme kinetics", hierarchical_graph, store,
                                enzyme_embedder, base_config)
        ids = {c.id for c in result.chunks}
        assert ids & {"c1", "c2"}

    def test_irrelevant_chunks_excluded(self, hierarchical_graph, store, enzyme_embedder, base_config):
        result = retrieve_graph("enzyme kinetics", hierarchical_graph, store,
                                enzyme_embedder, base_config)
        ids = {c.id for c in result.chunks}
        assert "c3" not in ids

    def test_empty_graph_returns_empty(self, tmp_path, store, enzyme_embedder, base_config):
        empty = GraphBuilder(tmp_path / "empty.json")
        result = retrieve_graph("any", empty, store, enzyme_embedder, base_config)
        assert result.chunks == []
        assert result.subgraph == []
        assert result.subgraph_node_keys == []
