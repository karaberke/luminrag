"""
Tests for backend/self_rag/multi_hop_reasoner.py
"""

import json
import threading
import pytest

from backend.schemas import Chunk, GraphTriple, ReflectionVerdict, RetrievalResult
from backend.self_rag.multi_hop_reasoner import _decompose, _merge_results, reason, retrieve_for_mode


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
    def test_provenance_is_unioned_across_decomposed_subquestions(self):
        first = _make_result(["Shared evidence"], ["same-id"], routing_mode="dense")
        second = _make_result(["Shared evidence"], ["same-id"], routing_mode="graph")
        first.chunks[0].metadata["retrieval_source"] = "dense"
        second.chunks[0].metadata["retrieval_source"] = "graph"
        merged = _merge_results([first, second])
        assert len(merged.chunks) == 1
        assert merged.routing_mode == "hybrid"
        assert merged.chunks[0].metadata["retrieval_source"] == "both"
        assert first.chunks[0].metadata["retrieval_source"] == "dense"

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

    def test_routing_mode_hybrid_when_dense_and_graph_results_are_merged(self):
        r1 = _make_result(routing_mode="dense")
        r2 = _make_result(routing_mode="graph")
        merged = _merge_results([r1, r2])
        assert merged.routing_mode == "hybrid"

    def test_routing_mode_graph_if_all_graph(self):
        merged = _merge_results([
            _make_result(routing_mode="graph"),
            _make_result(routing_mode="graph"),
        ])
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
        assert result.routing_mode == "dense"

    @pytest.mark.parametrize("query_mode", ["fast", "thorough"])
    def test_auto_conversation_preserves_none_through_answer_generation(self, mocker, query_mode):
        from backend.generation import generator
        decompose = mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["Hello"])
        retrieval = mocker.Mock(side_effect=AssertionError("Conversation must not retrieve"))
        relevance = mocker.patch("backend.self_rag.multi_hop_reasoner.reflect_retrieval")
        provider = mocker.patch("backend.generation.generator._call_llm",
                                return_value=generator._AnswerOutput(answer="Hello! How can I help?"))
        mocker.patch("backend.generation.generator.reflect_answer", return_value=_verdict())
        config = {**_ollama_cfg(), "query_mode": query_mode}
        result = reason("Hello", None, None, None, None, config, retrieve_callback=retrieval)
        output = generator.generate("Hello", result, config)
        assert result.routing_mode == output.routing_mode == "none"
        assert result.chunks == output.evidence_chunk_ids == []
        assert output.answer == "Hello! How can I help?"
        retrieval.assert_not_called()
        relevance.assert_not_called()
        provider.assert_called_once()
        assert "general knowledge" in provider.call_args.args[0]
        assert decompose.call_count == (query_mode == "thorough")

    def test_all_irrelevant_returns_empty(self, mocker):
        self._setup(mocker, ["Hello", "Q1", "Q2"], ["none", "dense", "graph"], [False, False])
        result = reason("Unrelated question.", None, None, None, None, _ollama_cfg())
        assert result.chunks == []
        assert result.subgraph == []
        assert result.routing_mode == "dense"

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


class TestRetrievalBoundary:
    @pytest.mark.parametrize("mode", ["dense", "graph", "hybrid"])
    def test_callback_replaces_only_retrieval(self, mocker, mode):
        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["Sub-question"])
        mocker.patch("backend.self_rag.multi_hop_reasoner.route_query", return_value=mode)
        dense = mocker.patch("backend.self_rag.multi_hop_reasoner.retrieve_dense")
        graph = mocker.patch("backend.self_rag.multi_hop_reasoner.retrieve_graph")
        evidence = _make_result(["Detached evidence"], routing_mode=mode)
        retrieve = mocker.Mock(return_value=evidence)
        reflection = mocker.patch("backend.self_rag.multi_hop_reasoner.reflect_retrieval", return_value=_verdict())
        config = _ollama_cfg()

        result = reason("Complex question", None, None, None, None, config, retrieve_callback=retrieve)

        retrieve.assert_called_once_with("Sub-question", mode)
        reflection.assert_called_once_with("Sub-question", evidence, config)
        dense.assert_not_called()
        graph.assert_not_called()
        assert result.chunks == evidence.chunks
        assert result.routing_mode == mode

    def test_no_retrieval_callback_for_none_routing(self, mocker):
        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["Hello"])
        mocker.patch("backend.self_rag.multi_hop_reasoner.route_query", return_value="none")
        retrieve = mocker.Mock()
        reflection = mocker.patch("backend.self_rag.multi_hop_reasoner.reflect_retrieval")

        result = reason("Hello", None, None, None, None, _ollama_cfg(), retrieve_callback=retrieve)

        assert result.chunks == []
        retrieve.assert_not_called()
        reflection.assert_not_called()

    def test_revision_rejection_stops_before_reflection_or_later_hops(self, mocker):
        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["First", "Second"])
        mocker.patch("backend.self_rag.multi_hop_reasoner.route_query", return_value="dense")
        retrieve = mocker.Mock(side_effect=RuntimeError("Knowledge revision changed"))
        reflection = mocker.patch("backend.self_rag.multi_hop_reasoner.reflect_retrieval")

        with pytest.raises(RuntimeError, match="Knowledge revision changed"):
            reason("Complex question", None, None, None, None, _ollama_cfg(), retrieve_callback=retrieve)

        retrieve.assert_called_once_with("First", "dense")
        reflection.assert_not_called()

    def test_llm_stages_run_outside_the_callers_retrieval_lock(self, mocker):
        lock = threading.Lock()
        stages = []

        def decompose(_question, _config):
            assert not lock.locked()
            stages.append("decompose")
            return ["Sub-question"]

        def route(_question, _config):
            assert not lock.locked()
            stages.append("route")
            return "dense"

        def dense(_question, _index, _store, _config):
            assert lock.locked()
            stages.append("retrieve")
            return _make_result(["Detached evidence"])

        def reflect(_question, _result, _config):
            assert not lock.locked()
            stages.append("reflect")
            return _verdict()

        config = _ollama_cfg()

        def guarded_retrieve(question, mode):
            with lock:
                return retrieve_for_mode(question, mode, None, None, None, None, config)

        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", side_effect=decompose)
        mocker.patch("backend.self_rag.multi_hop_reasoner.route_query", side_effect=route)
        mocker.patch("backend.self_rag.multi_hop_reasoner.retrieve_dense", side_effect=dense)
        mocker.patch("backend.self_rag.multi_hop_reasoner.reflect_retrieval", side_effect=reflect)

        result = reason("Complex question", None, None, None, None, config, retrieve_callback=guarded_retrieve)

        assert len(result.chunks) == 1
        assert stages == ["decompose", "route", "retrieve", "reflect"]

    def test_forced_mode_callback_skips_router_and_reflection(self, mocker):
        mocker.patch("backend.self_rag.multi_hop_reasoner._decompose", return_value=["Sub-question"])
        router = mocker.patch("backend.self_rag.multi_hop_reasoner.route_query")
        reflection = mocker.patch("backend.self_rag.multi_hop_reasoner.reflect_retrieval")
        retrieve = mocker.Mock(return_value=_make_result(["Evidence"], routing_mode="hybrid"))

        result = reason("Question", None, None, None, None, _ollama_cfg(), "hybrid", retrieve_callback=retrieve)

        retrieve.assert_called_once_with("Sub-question", "hybrid")
        router.assert_not_called()
        reflection.assert_not_called()
        assert result.routing_mode == "hybrid"

    @pytest.mark.parametrize("mode,dispatch", [("dense", "retrieve_dense"), ("graph", "retrieve_graph"), ("hybrid", "_retrieve_hybrid")])
    def test_shared_dispatcher_keeps_existing_retrieval_paths(self, mocker, mode, dispatch):
        expected = _make_result(["Evidence"], routing_mode=mode)
        selected = mocker.patch(f"backend.self_rag.multi_hop_reasoner.{dispatch}", return_value=expected)
        references = [object() for _ in range(4)]
        config = _ollama_cfg()

        result = retrieve_for_mode("Question", mode, *references, config)

        assert result is expected
        if mode == "dense":
            selected.assert_called_once_with("Question", references[0], references[2], config)
        elif mode == "graph":
            selected.assert_called_once_with("Question", references[1], references[2], references[3], config)
        else:
            selected.assert_called_once_with("Question", *references, config)
