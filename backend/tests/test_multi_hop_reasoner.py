"""
Tests for backend/self_rag/multi_hop_reasoner.py
"""

import json
import pytest

from backend.schemas import Chunk, GraphTriple, ReflectionVerdict, RetrievalResult
from backend.self_rag.multi_hop_reasoner import _decompose, _merge_results, reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, chunk_id: str | None = None) -> Chunk:
    c = Chunk(text=text, source_id="src1", modality="pdf", metadata={})
    if chunk_id:
        object.__setattr__(c, "id", chunk_id)
    return c


def _make_triple(head: str, relation: str, tail: str) -> GraphTriple:
    return GraphTriple(head=head, relation=relation, tail=tail, source_chunk_ids=[])


def _make_result(
    texts: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    triples: list[GraphTriple] | None = None,
    routing_mode: str = "dense",
) -> RetrievalResult:
    texts = texts or []
    chunk_ids = chunk_ids or [None] * len(texts)
    chunks = [_make_chunk(t, cid) for t, cid in zip(texts, chunk_ids)]
    return RetrievalResult(
        chunks=chunks,
        subgraph=triples or [],
        routing_mode=routing_mode,
    )


def _verdict(relevant: bool = True) -> ReflectionVerdict:
    return ReflectionVerdict(
        needs_retrieval=True,
        is_relevant=relevant,
        is_supported=True,
        is_useful=True,
        reasoning="test",
    )


def _ollama_cfg(max_sub: int = 4) -> dict:
    return {
        "multi_hop_reasoner": {
            "provider": "ollama",
            "ollama_model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "max_tokens": 256,
            "max_sub_questions": max_sub,
        },
        "query_router": {"use_llm_fallback": False},
        "self_rag": {"provider": "ollama"},
        "vector_retriever": {"top_k": 3},
        "graph_retriever": {
            "top_k_anchors": 2,
            "max_hops": 2,
            "relevance_threshold": 0.4,
            "max_nodes_per_hop": 5,
        },
    }


# ---------------------------------------------------------------------------
# _decompose
# ---------------------------------------------------------------------------

class TestDecompose:
    def test_valid_json_returns_sub_questions(self, mocker):
        payload = json.dumps({"sub_questions": ["What is ATP?", "How is ATP produced?"]})
        mocker.patch("backend.self_rag.multi_hop_reasoner._call_llm", return_value=payload)
        cfg = {"provider": "ollama", "max_sub_questions": 4}
        result = _decompose("Explain ATP and its production.", cfg)
        assert result == ["What is ATP?", "How is ATP produced?"]

    def test_fallback_on_llm_failure(self, mocker):
        mocker.patch(
            "backend.self_rag.multi_hop_reasoner._call_llm",
            side_effect=RuntimeError("timeout"),
        )
        cfg = {"provider": "ollama", "max_sub_questions": 4}
        result = _decompose("What is entropy?", cfg)
        assert result == ["What is entropy?"]

    def test_fallback_on_malformed_json(self, mocker):
        mocker.patch("backend.self_rag.multi_hop_reasoner._call_llm", return_value="not json")
        cfg = {"provider": "ollama", "max_sub_questions": 4}
        result = _decompose("What is entropy?", cfg)
        assert result == ["What is entropy?"]

    def test_clamps_to_max_sub(self, mocker):
        payload = json.dumps({"sub_questions": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]})
        mocker.patch("backend.self_rag.multi_hop_reasoner._call_llm", return_value=payload)
        cfg = {"provider": "ollama", "max_sub_questions": 3}
        result = _decompose("Complex question.", cfg)
        assert len(result) == 3

    def test_empty_list_falls_back(self, mocker):
        payload = json.dumps({"sub_questions": []})
        mocker.patch("backend.self_rag.multi_hop_reasoner._call_llm", return_value=payload)
        cfg = {"provider": "ollama", "max_sub_questions": 4}
        result = _decompose("What is ATP?", cfg)
        assert result == ["What is ATP?"]

    def test_markdown_fences_stripped(self, mocker):
        payload = f"```json\n{json.dumps({'sub_questions': ['Q1', 'Q2']})}\n```"
        mocker.patch("backend.self_rag.multi_hop_reasoner._call_llm", return_value=payload)
        cfg = {"provider": "ollama", "max_sub_questions": 4}
        result = _decompose("Multi-part question.", cfg)
        assert result == ["Q1", "Q2"]

    def test_empty_strings_filtered(self, mocker):
        payload = json.dumps({"sub_questions": ["Valid?", "", "  ", "Also valid?"]})
        mocker.patch("backend.self_rag.multi_hop_reasoner._call_llm", return_value=payload)
        cfg = {"provider": "ollama", "max_sub_questions": 4}
        result = _decompose("Question.", cfg)
        assert result == ["Valid?", "Also valid?"]


# ---------------------------------------------------------------------------
# _merge_results
# ---------------------------------------------------------------------------

class TestMergeResults:
    def test_chunks_deduplicated_by_id(self):
        chunk = _make_chunk("shared text", "id-1")
        r1 = _make_result(["text A"], ["id-A"])
        r1.chunks.append(chunk)
        r2 = _make_result(["text B"], ["id-B"])
        r2.chunks.append(chunk)  # duplicate
        merged = _merge_results([r1, r2])
        ids = [c.id for c in merged.chunks]
        assert ids.count("id-1") == 1

    def test_triples_deduplicated_by_head_relation_tail(self):
        t = _make_triple("ATP", "PART_OF", "Cell")
        r1 = _make_result(triples=[t])
        r2 = _make_result(triples=[t])  # same triple
        merged = _merge_results([r1, r2])
        assert len(merged.subgraph) == 1

    def test_routing_mode_graph_if_any_graph(self):
        r1 = _make_result(routing_mode="dense")
        r2 = _make_result(routing_mode="graph")
        merged = _merge_results([r1, r2])
        assert merged.routing_mode == "graph"

    def test_routing_mode_dense_if_all_dense(self):
        r1 = _make_result(routing_mode="dense")
        r2 = _make_result(routing_mode="dense")
        merged = _merge_results([r1, r2])
        assert merged.routing_mode == "dense"

    def test_all_unique_chunks_preserved(self):
        r1 = _make_result(["A", "B"], ["id-A", "id-B"])
        r2 = _make_result(["C", "D"], ["id-C", "id-D"])
        merged = _merge_results([r1, r2])
        assert len(merged.chunks) == 4

    def test_all_unique_triples_preserved(self):
        t1 = _make_triple("ATP", "CAUSES", "Energy")
        t2 = _make_triple("Glucose", "PART_OF", "Metabolism")
        r1 = _make_result(triples=[t1])
        r2 = _make_result(triples=[t2])
        merged = _merge_results([r1, r2])
        assert len(merged.subgraph) == 2

    def test_single_result_returned_unchanged(self):
        r = _make_result(["Only chunk"], ["id-1"])
        merged = _merge_results([r])
        assert len(merged.chunks) == 1
        assert merged.chunks[0].text == "Only chunk"


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------

class TestReason:
    def _setup(self, mocker, sub_questions, route_modes, relevant_flags):
        """Patch decompose, route_query, retrieve_dense/graph, reflect_retrieval."""
        mocker.patch(
            "backend.self_rag.multi_hop_reasoner._decompose",
            return_value=sub_questions,
        )

        route_iter = iter(route_modes)
        mocker.patch(
            "backend.self_rag.multi_hop_reasoner.route_query",
            side_effect=lambda q, cfg: next(route_iter),
        )

        retrieve_counter = {"n": 0}

        def fake_dense(q, idx, store, cfg):
            retrieve_counter["n"] += 1
            return _make_result([f"dense-chunk-{retrieve_counter['n']}"],
                                [f"dense-id-{retrieve_counter['n']}"])

        def fake_graph(q, builder, store, embedder, cfg):
            retrieve_counter["n"] += 1
            return _make_result(
                [f"graph-chunk-{retrieve_counter['n']}"],
                [f"graph-id-{retrieve_counter['n']}"],
                routing_mode="graph",
            )

        mocker.patch("backend.self_rag.multi_hop_reasoner.retrieve_dense", side_effect=fake_dense)
        mocker.patch("backend.self_rag.multi_hop_reasoner.retrieve_graph", side_effect=fake_graph)

        verdict_iter = iter([_verdict(r) for r in relevant_flags])
        mocker.patch(
            "backend.self_rag.multi_hop_reasoner.reflect_retrieval",
            side_effect=lambda q, res, cfg: next(verdict_iter),
        )

        return retrieve_counter

    def test_single_relevant_sub_question(self, mocker):
        self._setup(mocker, ["What is ATP?"], ["dense"], [True])
        result = reason("What is ATP?", None, None, None, None, _ollama_cfg())
        assert len(result.chunks) == 1
        assert result.routing_mode == "dense"

    def test_irrelevant_sub_question_skipped(self, mocker):
        self._setup(mocker, ["Q1", "Q2"], ["dense", "dense"], [False, True])
        result = reason("Complex question.", None, None, None, None, _ollama_cfg())
        # Only Q2 passes reflection
        assert len(result.chunks) == 1

    def test_none_routed_sub_question_skipped(self, mocker):
        self._setup(mocker, ["Hi!", "What is entropy?"], ["none", "dense"], [True])
        result = reason("Hi! What is entropy?", None, None, None, None, _ollama_cfg())
        assert len(result.chunks) == 1

    def test_all_irrelevant_returns_empty(self, mocker):
        self._setup(mocker, ["Q1", "Q2"], ["dense", "dense"], [False, False])
        result = reason("Unrelated question.", None, None, None, None, _ollama_cfg())
        assert result.chunks == []
        assert result.subgraph == []

    def test_graph_routing_used(self, mocker):
        self._setup(mocker, ["How do concepts relate?"], ["graph"], [True])
        result = reason("How do concepts relate?", None, None, None, None, _ollama_cfg())
        assert result.routing_mode == "graph"

    def test_chunks_merged_across_sub_questions(self, mocker):
        self._setup(mocker, ["Q1", "Q2", "Q3"], ["dense", "dense", "dense"], [True, True, True])
        result = reason("Three-part question.", None, None, None, None, _ollama_cfg())
        assert len(result.chunks) == 3

    def test_retrieve_dense_called_for_dense_mode(self, mocker):
        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["Q1"])
        mocker.patch("backend.self_rag.multi_hop_reasoner.route_query", return_value="dense")
        mock_dense = mocker.patch(
            "backend.self_rag.multi_hop_reasoner.retrieve_dense",
            return_value=_make_result(["chunk"]),
        )
        mocker.patch(
            "backend.self_rag.multi_hop_reasoner.reflect_retrieval",
            return_value=_verdict(True),
        )
        reason("Q1", None, None, None, None, _ollama_cfg())
        mock_dense.assert_called_once()

    def test_retrieve_graph_called_for_graph_mode(self, mocker):
        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["Q1"])
        mocker.patch("backend.self_rag.multi_hop_reasoner.route_query", return_value="graph")
        mock_graph = mocker.patch(
            "backend.self_rag.multi_hop_reasoner.retrieve_graph",
            return_value=_make_result([], routing_mode="graph"),
        )
        mocker.patch(
            "backend.self_rag.multi_hop_reasoner.reflect_retrieval",
            return_value=_verdict(True),
        )
        reason("Q1", None, None, None, None, _ollama_cfg())
        mock_graph.assert_called_once()
