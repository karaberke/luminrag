"""Background model/cache work must never convoy the shared application lock."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest

from backend.retrieval import graph_retriever
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_retriever import IndexModelMismatchError, VectorIndex
from backend.tests.test_ingest_safety import api_state  # noqa: F401


def _embedder(entered, release):
    class Model:
        def embed(self, texts, **_kwargs):
            if "background" in texts:
                entered.set()
                assert release.wait(10), "Test must release background inference"
            return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

        def query_embed(self, text):
            return self.embed([text])

        def get_embedding_dimension(self):
            return 2

    embedder = Embedder.__new__(Embedder)
    embedder._model = Model()
    embedder._dimension = 2
    embedder._model_name = "synthetic-resource-test"
    embedder._inference_lock = threading.RLock()
    return embedder


@contextmanager
def _busy_resource(api, monkeypatch, resource):
    from backend.self_rag import multi_hop_reasoner

    entered, release = threading.Event(), threading.Event()
    embedder = _embedder(entered, release)
    index = VectorIndex(embedder)
    index.build(api.store.get_all_chunks())
    monkeypatch.setattr(api.main.state, "embedder", embedder)
    monkeypatch.setattr(api.main.state, "vector_index", index)
    monkeypatch.setattr(multi_hop_reasoner, "_decompose", lambda question, _config: [question])
    monkeypatch.setattr(api.main, "generate", Mock(side_effect=AssertionError("Busy query must not call provider")))
    monkeypatch.setattr(api.main, "warm_graph_retriever", graph_retriever.warm_up)
    graph_retriever.clear_node_embeddings()

    with ThreadPoolExecutor(max_workers=4) as pool:
        if resource == "embedding":
            background = pool.submit(embedder.embed, ["background"])
        else:
            graph = nx.MultiDiGraph()
            graph.add_node("background", name="background", node_type="content")

            class SlowCacheEmbedder:
                def embed(self, texts):
                    entered.set()
                    assert release.wait(10), "Test must release background cache warmup"
                    return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))

            background = pool.submit(graph_retriever.warm_up, graph, SlowCacheEmbedder())
        try:
            assert entered.wait(3)
            yield pool
        finally:
            release.set()
            background.result(timeout=3)
    graph_retriever.clear_node_embeddings()


@pytest.mark.parametrize("mode,resource", [("dense", "embedding"), ("hybrid", "embedding"), ("graph", "cache"), ("hybrid", "cache")])
def test_busy_query_returns_prompt_503_and_graph_reads_remain_available(api_state, monkeypatch, mode, resource):
    api = api_state
    with _busy_resource(api, monkeypatch, resource) as pool:
        response = pool.submit(api.client.post, "/api/query", json={
            "query": "Explain the material", "routing_mode": mode,
        }).result(timeout=1)
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert "retry" in response.json()["detail"].lower()
        assert pool.submit(api.client.get, "/api/graph").result(timeout=1).status_code == 200




@pytest.mark.parametrize("resource", ["embedding", "cache"])
def test_manual_graph_edit_persists_and_defers_busy_cache_warmup(api_state, monkeypatch, resource):
    api = api_state
    with _busy_resource(api, monkeypatch, resource) as pool:
        response = pool.submit(api.client.post, "/api/graph/node", json={"name": "New manual topic"}).result(timeout=1)
        assert response.status_code == 201
        graph = pool.submit(api.client.get, "/api/graph").result(timeout=1)
        assert graph.status_code == 200
        assert any(node["name"] == "New manual topic" for node in graph.json()["nodes"])


def test_clear_busy_cache_fails_before_files_or_sql_change(api_state, monkeypatch):
    api = api_state
    with _busy_resource(api, monkeypatch, "cache") as pool:
        response = pool.submit(api.client.delete, "/api/data").result(timeout=1)
        assert response.status_code == 503
        assert api.store.count() == 1
        assert api.source.read_bytes() == b"old raw"
        assert (api.processed / "graph.json").exists()
        assert pool.submit(api.client.get, "/api/graph").result(timeout=1).status_code == 200
    assert api.client.delete("/api/data").status_code == 200
    assert api.store.count() == 0




def test_startup_rebuilds_unknown_model_in_memory_without_overwriting_disk(api_state, monkeypatch):
    api = api_state
    before_index = (api.processed / "faiss.index").read_bytes()
    built = []

    class LegacyIndex:
        def __init__(self, _embedder):
            pass

        def load(self, _path):
            raise IndexModelMismatchError("legacy metadata")

        def build(self, chunks):
            built.extend(chunks)

        def save(self, _path):
            pytest.fail("Startup migration must not bypass durable publication")

    class Scheduler:
        async def scan_and_enqueue_pending(self):
            return 0

        async def stop(self):
            pass

    monkeypatch.setattr(api.main, "_load_config", lambda: api.main.state.config)
    monkeypatch.setattr(api.main, "DocumentStore", lambda _path: api.store)
    monkeypatch.setattr(api.main, "Embedder", lambda _name: SimpleNamespace(model_name="new-model"))
    monkeypatch.setattr(api.main, "VectorIndex", LegacyIndex)
    monkeypatch.setattr(api.main, "_warm_ollama", Mock())
    monkeypatch.setattr(api.main, "_start_illustration_scheduler", lambda: Scheduler())

    async def start():
        async with api.main.lifespan(api.main.app):
            assert api.main.state.index_loaded
            assert [chunk.id for chunk in built] == [api.old.id]
            assert (api.processed / "faiss.index").read_bytes() == before_index

    asyncio.run(start())
