"""Derivative recovery must never bypass an unresolved storage transaction."""

import asyncio
from unittest.mock import Mock

import pytest

from backend.db.document_store import DocumentStore
from backend.tests.test_ingest_safety import api_state  # noqa: F401
from backend.utils.recovery import RecoveryError


def test_corrupt_pending_journal_prevents_all_index_rebuild_and_model_work(api_state, monkeypatch):
    api = api_state
    recovery_dir = api.processed / "recovery"
    recovery_dir.mkdir()
    journal = recovery_dir / f"{'f' * 32}.json"
    journal.write_text("{", encoding="utf-8")
    index_path = api.processed / "faiss.index"
    index_bytes = index_path.read_bytes()
    source_bytes = api.source.read_bytes()
    monkeypatch.setattr(api.main, "_load_config", lambda: api.main.state.config)
    monkeypatch.setattr(api.main, "DocumentStore", lambda _path: api.store)
    monkeypatch.setattr(api.main.state, "storage_error", None)
    embedder = Mock(side_effect=AssertionError("Model work must follow successful journal recovery"))
    index = Mock(side_effect=AssertionError("Index rebuilding must follow successful journal recovery"))
    monkeypatch.setattr(api.main, "Embedder", embedder)
    monkeypatch.setattr(api.main, "VectorIndex", index)

    async def start():
        async with api.main.lifespan(api.main.app):
            pytest.fail("Startup must not serve potentially mixed storage")

    with pytest.raises(RecoveryError):
        asyncio.run(start())
    embedder.assert_not_called()
    index.assert_not_called()
    assert api.main.state.storage_error is not None
    assert journal.read_text(encoding="utf-8") == "{"
    assert index_path.read_bytes() == index_bytes
    assert api.source.read_bytes() == source_bytes
    with DocumentStore(api.store.path) as reopened:
        assert reopened.get_chunk(api.old.id).text == "previous material"
