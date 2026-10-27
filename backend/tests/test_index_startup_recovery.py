"""Rebuild only unusable derived vectors from recovered authoritative chunks."""

import asyncio
import json
import sqlite3
from unittest.mock import Mock

import numpy as np
import pytest

from backend.retrieval.vector_retriever import VectorIndex
from backend.tests.test_ingest_safety import api_state  # noqa: F401


class _Embedder:
    dimension = 3
    model_name = "current-model"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    def embed_one(self, text):
        return self.embed([text])[0]


class _Scheduler:
    async def scan_and_enqueue_pending(self):
        return 0

    async def stop(self):
        pass


def _startup(api, monkeypatch):
    embedder = _Embedder()
    monkeypatch.setattr(api.main, "_load_config", lambda: api.main.state.config)
    monkeypatch.setattr(api.main, "DocumentStore", lambda _path: api.store)
    monkeypatch.setattr(api.main, "Embedder", lambda _name: embedder)
    monkeypatch.setattr(api.main, "VectorIndex", VectorIndex)
    monkeypatch.setattr(VectorIndex, "save", Mock(side_effect=AssertionError("Startup cannot bypass journal publication")))
    monkeypatch.setattr(api.main, "_warm_ollama", Mock())
    monkeypatch.setattr(api.main, "_start_illustration_scheduler", lambda: _Scheduler())
    return embedder


@pytest.mark.parametrize("damage", [
    "missing_index", "corrupt_index", "missing_sidecar", "invalid_json", "sidecar_count",
    "sidecar_version", "legacy", "different_model", "different_dimension", "unsupported_index",
])
def test_startup_rebuilds_damaged_derived_index_in_memory_only(api_state, monkeypatch, damage):
    api = api_state
    index_path = api.processed / "faiss.index"
    sidecar_path = index_path.with_suffix(".json")
    dimension = 4 if damage == "different_dimension" else 3
    with index_path.open("wb") as stream:
        matrix = np.ones((1, dimension), dtype=np.float64 if damage == "unsupported_index" else np.float32)
        np.save(stream, matrix, allow_pickle=False)
    metadata = {"version": 1, "model": "current-model", "chunk_ids": [api.old.id]}
    if damage == "legacy":
        metadata = [api.old.id]
    elif damage == "different_model":
        metadata["model"] = "different-same-dimension-model"
    elif damage == "sidecar_count":
        metadata["chunk_ids"] = []
    elif damage == "sidecar_version":
        metadata["version"] = 99
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    if damage == "missing_index":
        index_path.unlink()
    elif damage == "missing_sidecar":
        sidecar_path.unlink()
    elif damage == "corrupt_index":
        index_path.write_bytes(b"corrupted NumPy bytes")
    elif damage == "invalid_json":
        sidecar_path.write_text("not JSON", encoding="utf-8")
    before = {path: path.read_bytes() if path.exists() else None for path in (index_path, sidecar_path)}
    raw_bytes = api.source.read_bytes()
    embedder = _startup(api, monkeypatch)

    async def start():
        async with api.main.lifespan(api.main.app):
            assert api.main.state.index_loaded
            assert embedder.calls == [["previous material"]]
            assert api.main.state.vector_index.search("query", 1)[0][0] == api.old.id
            assert api.store.get_chunk(api.old.id).text == "previous material"
            assert api.source.read_bytes() == raw_bytes
            for path, content in before.items():
                assert (path.read_bytes() if path.exists() else None) == content

    asyncio.run(start())


def test_missing_index_with_empty_store_stays_unavailable_without_embedding(api_state, monkeypatch):
    api = api_state
    api.store.clear_all()
    (api.processed / "faiss.index").unlink()
    embedder = _startup(api, monkeypatch)

    async def start():
        async with api.main.lifespan(api.main.app):
            assert not api.main.state.index_loaded
            assert embedder.calls == []
            assert not (api.processed / "faiss.index").exists()

    asyncio.run(start())


@pytest.mark.parametrize("failure", ["embedder", "rebuild"])
def test_startup_model_failure_closes_store_and_propagates(api_state, monkeypatch, failure):
    api = api_state
    _startup(api, monkeypatch)
    close = Mock(wraps=api.store.close)
    monkeypatch.setattr(api.store, "close", close)
    if failure == "embedder":
        monkeypatch.setattr(api.main, "Embedder", Mock(side_effect=RuntimeError("model unavailable")))
    else:
        monkeypatch.setattr(VectorIndex, "build", Mock(side_effect=RuntimeError("model unavailable")))

    async def start():
        async with api.main.lifespan(api.main.app):
            pytest.fail("A failed rebuild must not start with a partial index")

    with pytest.raises(RuntimeError, match="model unavailable"):
        asyncio.run(start())
    close.assert_called_once()
    assert not api.main.state.startup_complete
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        api.store.count()
