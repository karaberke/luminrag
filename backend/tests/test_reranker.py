"""
Tests for backend/retrieval/reranker.py

The cross-encoder loader is patched with unittest.mock, so no model is
downloaded and fastembed is never imported.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.retrieval.reranker import rerank
from backend.schemas import Chunk

_LOADER = "backend.retrieval.reranker._get_cross_encoder"


@pytest.fixture
def chunks() -> list[Chunk]:
    return [Chunk(id=f"c{i}", text=f"text {i}", source_id="s", modality="pdf") for i in range(5)]


def _encoder(logits: list[float]) -> MagicMock:
    ce = MagicMock()
    ce.rerank.return_value = np.array(logits)
    return ce


class TestModelFailure:
    def test_failure_still_honours_max_sources(self, chunks):
        with patch(_LOADER, side_effect=RuntimeError("no model")):
            out = rerank("q", chunks, {}, min_relevancy=0.5, max_sources=2)
        assert [c.id for c in out] == ["c0", "c1"]
        assert all("relevancy_score" not in c.metadata for c in out)

    def test_predict_failure_still_honours_max_sources(self, chunks):
        ce = MagicMock()
        ce.rerank.side_effect = RuntimeError("OOM")
        with patch(_LOADER, return_value=ce):
            out = rerank("q", chunks, {}, max_sources=3)
        assert [c.id for c in out] == ["c0", "c1", "c2"]

    def test_failure_without_cap_returns_all(self, chunks):
        with patch(_LOADER, side_effect=RuntimeError("no model")):
            out = rerank("q", chunks, {}, max_sources=None)
        assert out == chunks

    def test_failure_logs_cutoff_not_applied(self, chunks, caplog):
        with caplog.at_level(logging.ERROR, logger="backend.retrieval.reranker"), \
             patch(_LOADER, side_effect=RuntimeError("no model")):
            rerank("q", chunks, {}, min_relevancy=0.4, max_sources=2)
        assert "degraded" in caplog.text
        assert "min_relevancy=0.40 cutoff NOT applied" in caplog.text


class TestScoring:
    def test_sorts_filters_and_caps(self, chunks):
        with patch(_LOADER, return_value=_encoder([-5.0, 3.0, 1.0, 2.0, -4.0])):
            out = rerank("q", chunks, {}, min_relevancy=0.5, max_sources=2)
        assert [c.id for c in out] == ["c1", "c3"]
        assert out[0].metadata["relevancy_score"] > out[1].metadata["relevancy_score"]

    def test_all_below_cutoff_returns_empty(self, chunks):
        with patch(_LOADER, return_value=_encoder([-5.0] * 5)):
            assert rerank("q", chunks, {}, min_relevancy=0.5) == []

    def test_empty_input_skips_model(self):
        with patch(_LOADER) as loader:
            assert rerank("q", [], {}, max_sources=1) == []
        loader.assert_not_called()


def test_concurrent_queries_load_once_and_keep_scored_results(monkeypatch, chunks):
    from backend.retrieval import reranker

    loads = []
    ready = threading.Barrier(8)

    class NonReentrantEncoder:
        def __init__(self, model_name, **_kwargs):
            loads.append(model_name)
            self.busy = threading.Lock()
            time.sleep(0.02)  # Exposes duplicate first-use loading without coordination.

        def rerank(self, query, documents):
            assert self.busy.acquire(blocking=False), "Concurrent model/tokenizer access"
            try:
                time.sleep(0.002)
                return np.arange(len(documents), dtype=np.float32)
            finally:
                self.busy.release()

    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", SimpleNamespace(TextCrossEncoder=NonReentrantEncoder))
    monkeypatch.setattr(reranker, "_cross_encoder", None)
    monkeypatch.setattr(reranker, "_loaded_model_name", None)

    def query(_):
        ready.wait(timeout=5)
        return rerank("question", chunks, {}, max_sources=2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(query, range(8)))
    assert loads == [reranker._DEFAULT_MODEL]
    for result in results:
        assert [chunk.id for chunk in result] == ["c4", "c3"]
        assert all("relevancy_score" in chunk.metadata for chunk in result)
