"""Bounded work and detached generation tested with synthetic API state."""

from __future__ import annotations

import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from backend.schemas import Chunk, GenerationResult, ReflectionVerdict, RetrievalResult
from backend.tests.test_ingest_safety import api_state, _staged_attempt
from backend.utils.state import current_epoch, run_locked


def _answer():
    return GenerationResult(
        answer="Synthetic answer", routing_mode="dense", evidence_chunk_ids=[],
        reflection=ReflectionVerdict(needs_retrieval=True, is_relevant=True,
                                     is_supported=True, is_useful=True, reasoning="fixture"),
    )


def _blocked_generator(api, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(api.main, "reason", lambda **_kwargs: RetrievalResult(
        chunks=[], subgraph=[], routing_mode="dense"
    ))

    def generate(**_kwargs):
        entered.set()
        assert release.wait(10), "test did not release the fake provider"
        return _answer()

    monkeypatch.setattr(api.main, "generate", generate)
    return entered, release


def test_graph_read_finishes_while_query_provider_is_waiting(api_state, monkeypatch):
    api = api_state
    entered, release = _blocked_generator(api, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as workers:
        query = workers.submit(api.main.query, api.main.QueryRequest(query="What is the concept?"))
        try:
            assert entered.wait(3)
            graph = workers.submit(api.main.get_graph).result(timeout=1)
            assert graph.nodes
        finally:
            release.set()
        assert query.result(timeout=3).content == "Synthetic answer"


@pytest.mark.parametrize("mutation", ["upload", "clear", "graph_edit"])
def test_query_rejects_stale_citations_after_knowledge_changes(api_state, monkeypatch, mutation):
    api = api_state
    entered, release = _blocked_generator(api, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as workers:
        query = workers.submit(api.main.query, api.main.QueryRequest(query="What is the concept?"))
        try:
            assert entered.wait(3)
            if mutation == "upload":
                job, staged, job_dir = _staged_attempt(api)
                workers.submit(run_locked, api.main._ingest_locked, job, staged, current_epoch(), job_dir).result(timeout=3)
                assert job.status == "done"
            elif mutation == "clear":
                workers.submit(run_locked, api.main._clear_data_locked).result(timeout=3)
            else:
                node_id = next(iter(api.graph.graph))
                workers.submit(api.main.patch_graph_node, node_id, api.main.PatchNodeRequest(summary="edited material")).result(timeout=3)
        finally:
            release.set()
        with pytest.raises(HTTPException) as exc:
            query.result(timeout=3)
        assert exc.value.status_code == 409


def test_illustration_only_update_does_not_invalidate_query(api_state, monkeypatch):
    api = api_state
    entered, release = _blocked_generator(api, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as workers:
        query = workers.submit(api.main.query, api.main.QueryRequest(query="What is the concept?"))
        try:
            assert entered.wait(3)

            def illustration_update():
                node = next(iter(api.graph.graph))
                api.graph.graph.nodes[node]["illustration"] = {"kind": "generated", "image_path": "fixture.png"}

            workers.submit(run_locked, illustration_update).result(timeout=1)
        finally:
            release.set()
        assert query.result(timeout=3).content == "Synthetic answer"




def test_document_preview_memory_is_bounded_by_preview_not_corpus(api_state):
    api = api_state
    chunks = [Chunk(id=str(index), source_id="fixture", modality="pdf", text="x" * 100_000)
              for index in range(100)]
    tracemalloc.start()
    try:
        preview = api.main._make_doc_preview(chunks, ["notes.pdf"])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert preview == ("Source files: notes.pdf\n\n" + "x" * 100_000)[:3000]
    assert peak < 100_000  # old implementation allocated ~20 MB for this fixture


@pytest.mark.parametrize("metadata", [
    {"start_time": "bad", "page_number": "bad"},
    {"start_time": float("nan"), "page_number": float("inf")},
    {"start_time": float("inf"), "page_number": {"page": 1}},
])
def test_all_evidence_routes_tolerate_invalid_persisted_metadata(api_state, metadata):
    api = api_state
    chunk = api.old.model_copy(update={"metadata": metadata})
    api.store.save_chunks([chunk])
    result = RetrievalResult(chunks=[chunk], subgraph=[], routing_mode="dense")
    direct = api.main._chunks_to_evidence(result)
    resolved = api.main._chunk_ids_to_evidence([chunk.id])
    detail = api.main.get_graph_node_detail(next(iter(api.graph.graph))).evidence
    for evidence in (direct, resolved, detail):
        assert len(evidence) == 1
        assert evidence[0].timestamp is None
        assert evidence[0].page is None


def test_ingest_capacity_rejection_removes_staging_without_scheduling(api_state, monkeypatch):
    from backend.utils.jobs import MAX_ACTIVE_JOBS

    api = api_state
    api.main._jobs.update({str(index): api.main._IngestJob(str(index)) for index in range(MAX_ACTIVE_JOBS)})
    monkeypatch.setattr(api.main, "_spawn", lambda *_args: pytest.fail("Rejected work must not be scheduled"))
    job, staged, job_dir = _staged_attempt(api)
    with pytest.raises(HTTPException) as exc:
        api.main._admit_ingest_job(job.job_id, staged, job_dir, current_epoch())
    assert exc.value.status_code == 429
    assert not job_dir.exists()
    assert len(api.main._jobs) == MAX_ACTIVE_JOBS




def test_ingest_reuses_existing_vectors_and_encodes_only_changed_source(api_state, monkeypatch):
    import numpy as np
    from backend.retrieval.vector_retriever import VectorIndex

    class CountingEmbedder:
        dimension = 2

        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            rows = np.array([[len(text), 1.0] for text in texts], dtype=np.float32)
            return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    api = api_state
    unchanged = Chunk(id="unchanged", source_id="another-source", modality="pdf", text="unchanged material")
    api.store.save_chunks([unchanged])
    embedder = CountingEmbedder()
    previous = VectorIndex(embedder)
    previous.build(api.store.get_all_chunks())
    embedder.calls.clear()
    monkeypatch.setattr(api.main, "VectorIndex", VectorIndex)
    monkeypatch.setattr(api.main.state, "embedder", embedder)
    monkeypatch.setattr(api.main.state, "vector_index", previous)
    job, staged, job_dir = _staged_attempt(api)
    api.main._ingest(job, staged, current_epoch(), job_dir)
    assert job.status == "done"
    assert embedder.calls == [["new material"]]
    assert api.store.get_chunk("unchanged").text == "unchanged material"


@pytest.mark.parametrize("phase", ["decompose", "route", "reflect"])
@pytest.mark.parametrize("clear_during_call", [False, True])
def test_reasoner_provider_waits_allow_reads_and_detect_cleared_data(
    api_state, monkeypatch, phase, clear_during_call,
):
    from backend.self_rag import multi_hop_reasoner as reasoner

    api = api_state
    entered, release = threading.Event(), threading.Event()

    def wait_then(value):
        entered.set()
        assert release.wait(10)
        return value

    monkeypatch.setattr(reasoner, "_decompose", lambda question, _config: wait_then([question]) if phase == "decompose" else [question])
    monkeypatch.setattr(reasoner, "route_query", lambda *_args: wait_then("dense") if phase == "route" else "dense")
    monkeypatch.setattr(reasoner, "reflect_retrieval", lambda *_args: wait_then(_answer().reflection) if phase == "reflect" else _answer().reflection)
    monkeypatch.setattr(reasoner, "retrieve_for_mode", lambda *_args: RetrievalResult(chunks=[], subgraph=[], routing_mode="dense"))
    monkeypatch.setattr(api.main, "generate", lambda **_kwargs: _answer())
    with ThreadPoolExecutor(max_workers=3) as workers:
        query = workers.submit(api.client.post, "/api/query", json={"query": "Explain this complex concept"})
        try:
            assert entered.wait(3)
            response = workers.submit(api.client.get, "/api/graph").result(timeout=1)
            assert response.status_code == 200
            if clear_during_call:
                cleared = workers.submit(api.client.delete, "/api/data").result(timeout=3)
                assert cleared.status_code == 200
        finally:
            release.set()
        response = query.result(timeout=3)
    assert response.status_code == (409 if clear_during_call else 200)
