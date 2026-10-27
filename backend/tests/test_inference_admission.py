"""Inference overload must fail before providers run and release every slot."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pytest
from fastapi import HTTPException

from backend.schemas import GenerationResult, ReflectionVerdict, RetrievalResult
from backend.tests.test_ingest_safety import api_state
from backend.utils import inference


def test_parallel_calls_reject_without_waiting_and_restore_capacity():
    admission = inference.InferenceAdmission(2)
    entered = Queue()
    release = threading.Event()

    def hold_slot():
        with admission.reserve():
            entered.put(True)
            assert release.wait(10), "test did not release admitted calls"

    with ThreadPoolExecutor(max_workers=3) as workers:
        active = [workers.submit(hold_slot) for _ in range(2)]
        try:
            for _ in active:
                entered.get(timeout=3)
            rejected = workers.submit(hold_slot)
            with pytest.raises(HTTPException) as exc:
                rejected.result(timeout=1)
            assert exc.value.status_code == 429
            assert exc.value.headers == {"Retry-After": "2"}
        finally:
            release.set()
        for call in active:
            call.result(timeout=3)

    # Both original slots, including the one following the rejection, remain
    # available. Rejecting a request must never release another call's slot.
    with admission.reserve(), admission.reserve():
        with pytest.raises(HTTPException):
            with admission.reserve():
                pytest.fail("A third call was admitted")


def test_error_releases_capacity():
    admission = inference.InferenceAdmission(1)
    with pytest.raises(RuntimeError, match="provider failed"):
        with admission.reserve():
            raise RuntimeError("provider failed")
    with admission.reserve():
        pass


@pytest.mark.parametrize("raw", ["0", "-1", "bad", "1.5", ""])
def test_invalid_config_fails_closed(monkeypatch, raw):
    monkeypatch.setenv(inference.INFERENCE_LIMIT_ENV, raw)
    with pytest.raises(ValueError, match="positive integer"):
        inference.configured_inference_limit()


def test_default_and_configured_capacity(monkeypatch):
    monkeypatch.delenv(inference.INFERENCE_LIMIT_ENV, raising=False)
    assert inference.configured_inference_limit() == 4
    monkeypatch.setenv(inference.INFERENCE_LIMIT_ENV, "2")
    assert inference.configured_inference_limit() == 2


def _configure_query_provider(api, monkeypatch, *, entered=None, release=None):
    """Exercise query response validation and admission without provider traffic."""
    calls = []
    monkeypatch.setattr(api.main, "reason", lambda **_kwargs: RetrievalResult(
        chunks=[], subgraph=[], routing_mode="dense",
    ))

    def generate(**kwargs):
        calls.append(kwargs)
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(10), "test did not release the fake provider"
        return GenerationResult(
            answer="Synthetic answer", routing_mode="dense", evidence_chunk_ids=[],
            reflection=ReflectionVerdict(
                needs_retrieval=True, is_relevant=True, is_supported=True,
                is_useful=True, reasoning="fixture",
            ),
        )

    monkeypatch.setattr(api.main, "generate", generate)
    return "/api/query", {"query": "Explain enzymes"}, calls


def test_http_overload_precedes_provider_calls_and_binding_survives(api_state, monkeypatch):
    api = api_state
    admission = inference.InferenceAdmission(1)
    monkeypatch.setattr(inference, "admission", admission)
    route, body, calls = _configure_query_provider(api, monkeypatch)

    with admission.reserve():
        rejected = api.client.post(route, json=body)
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "2"
    assert calls == []

    accepted = api.client.post(route, json=body)
    assert accepted.status_code == 200, accepted.text
    assert len(calls) == 1
    assert calls[0]["question"] == "Explain enzymes"


def test_parallel_queries_share_one_http_admission_budget(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(inference, "admission", inference.InferenceAdmission(1))
    entered, release = threading.Event(), threading.Event()
    route, body, calls = _configure_query_provider(api, monkeypatch, entered=entered, release=release)

    with ThreadPoolExecutor(max_workers=2) as workers:
        active = workers.submit(api.client.post, route, json=body)
        try:
            assert entered.wait(3)
            rejected = workers.submit(api.client.post, route, json=body).result(timeout=1)
            assert rejected.status_code == 429
            assert len(calls) == 1
            assert api.client.get("/api/graph").status_code == 200
        finally:
            release.set()
        assert active.result(timeout=3).status_code == 200
    assert len(calls) == 1
    assert api.client.post(route, json=body).status_code == 200
    assert len(calls) == 2


@pytest.mark.parametrize(
    "question", ["", " " * 4, "x" * (inference.MAX_QUERY_CHARACTERS + 1)],
    ids=["empty", "blank", "oversized"],
)
def test_invalid_query_text_is_rejected_before_provider_work(api_state, monkeypatch, question):
    api = api_state
    monkeypatch.setattr(api.main, "reason", lambda **_kwargs: pytest.fail("Invalid query reached inference"))
    response = api.client.post("/api/query", json={"query": question})
    assert response.status_code == 422


def test_maximum_query_length_is_accepted(api_state, monkeypatch):
    route, body, calls = _configure_query_provider(api_state, monkeypatch)
    body["query"] = "x" * inference.MAX_QUERY_CHARACTERS
    response = api_state.client.post(route, json=body)
    assert response.status_code == 200
    assert len(calls) == 1


def test_http_provider_error_releases_slot(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(inference, "admission", inference.InferenceAdmission(1))
    route, body, calls = _configure_query_provider(api, monkeypatch)
    original_generate = api.main.generate

    def fail(**_kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(api.main, "generate", fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        api.client.post(route, json=body)
    monkeypatch.setattr(api.main, "generate", original_generate)
    assert api.client.post(route, json=body).status_code == 200
    assert len(calls) == 1
