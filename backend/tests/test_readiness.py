"""Deployment probes exercise real routes without loading models or real data."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from backend.tests.test_ingest_safety import api_state


@pytest.mark.parametrize("started,storage_error,expected", [
    (False, None, 503),
    (True, None, 200),
    (True, "private storage path / customer data must not be exposed", 503),
])
def test_public_readiness_reports_availability_without_details(
    api_state, monkeypatch, started, storage_error, expected,
):
    api = api_state
    monkeypatch.setattr(api.main.state, "startup_complete", started)
    monkeypatch.setattr(api.main.state, "storage_error", storage_error)
    # Clear even the test's default bearer credential: monitors need no key.
    api.client.headers.pop("authorization", None)
    response = api.client.get("/api/ready")
    assert response.status_code == expected
    assert response.json() == {"status": "ready" if expected == 200 else "unavailable"}
    assert response.headers["cache-control"] == "no-store"
    health = api.client.get("/api/health").json()
    assert health["status"] == "ok"
    assert ".docx" in health["features"]["accepted_extensions"]
    assert "storage_error" not in health and "chunks" not in health
    assert api.client.get("/api/graph").status_code == 401


def test_empty_knowledge_base_is_ready_for_onboarding(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(api.main.state, "startup_complete", True)
    monkeypatch.setattr(api.main.state, "storage_error", None)
    monkeypatch.setattr(api.main.state, "index_loaded", False)
    monkeypatch.setattr(api.main.state, "graph_loaded", False)
    assert api.client.get("/api/ready").status_code == 200


def _stub_startup(api, monkeypatch):
    main = api.main
    store = Mock()
    store.count.return_value = 0
    graph = Mock()
    graph.node_count.return_value = 0
    scheduler = Mock()
    scheduler.scan_and_enqueue_pending = AsyncMock(return_value=0)
    scheduler.stop = AsyncMock()
    monkeypatch.setattr(main, "_load_config", lambda: main.state.config)
    monkeypatch.setattr(main, "DocumentStore", lambda _path: store)
    monkeypatch.setattr(main, "_recover_storage", Mock())
    monkeypatch.setattr(main, "Embedder", Mock())
    monkeypatch.setattr(main, "VectorIndex", Mock())
    monkeypatch.setattr(main, "GraphBuilder", lambda _path: graph)
    monkeypatch.setattr(main, "_warm_ollama", Mock())

    def start_scheduler():
        main.state.illustration_worker = scheduler
        return scheduler

    monkeypatch.setattr(main, "_start_illustration_scheduler", start_scheduler)
    monkeypatch.setattr(main.state, "startup_complete", False)
    monkeypatch.setattr(main.state, "storage_error", None)
    return store, scheduler


@pytest.mark.parametrize("stop_fails", [False, True])
def test_lifespan_revokes_readiness_and_closes_storage_even_on_error(
    api_state, monkeypatch, stop_fails,
):
    api = api_state
    store, scheduler = _stub_startup(api, monkeypatch)
    if stop_fails:
        scheduler.stop.side_effect = RuntimeError("worker stop failed")

    async def serve():
        assert not api.main.state.startup_complete
        async with api.main.lifespan(api.main.app):
            assert api.main.state.startup_complete
            raise ValueError("server interrupted")

    with pytest.raises(RuntimeError if stop_fails else ValueError):
        asyncio.run(serve())
    assert not api.main.state.startup_complete
    store.close.assert_called_once()
    scheduler.stop.assert_awaited_once()


def test_failed_startup_never_reports_ready(api_state, monkeypatch):
    main = api_state.main
    monkeypatch.setattr(main.state, "startup_complete", True)
    monkeypatch.setattr(main, "_load_config", Mock(side_effect=RuntimeError("invalid config")))

    async def start():
        async with main.lifespan(main.app):
            pytest.fail("startup should have failed")

    with pytest.raises(RuntimeError, match="invalid config"):
        asyncio.run(start())
    assert not main.state.startup_complete
