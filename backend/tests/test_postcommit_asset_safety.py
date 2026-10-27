"""Real committed SQLite rows must never reference finally-deleted image files."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.db.document_store import DocumentStore
from backend.graph.illustration_worker import IllustrationScheduler
from backend.tests.test_ingest_safety import _staged_attempt, api_state
from backend.utils.recovery import RecoveryError
from backend.utils.state import current_epoch, run_locked


def _run(api, job, staged, job_dir):
    asyncio.run(api.main._run_ingest_job(job, staged, job_dir, current_epoch()))


def _assert_committed_assets(api):
    # A separate real SQLite connection observes durable rows, independent of
    # the original connection's marker-read fault and in-memory app objects.
    with DocumentStore(api.store.path) as reopened:
        chunk = reopened.get_chunk(api.old.id)
    assert chunk.text == "new material"
    assert api.source.read_bytes() == b"new raw"
    assert Path(chunk.metadata["image_paths"][0]).read_bytes() == b"new derived image"
    return chunk


def _assert_coherent_or_unavailable(api):
    response = api.client.get("/api/graph")
    if response.status_code == 200:
        names = {node["name"] for node in response.json()["nodes"]}
        assert "New concept" in names
        assert "Old concept" not in names
        assert api.main.state.vector_index is not api.index
    else:
        assert response.status_code == 503
        assert api.main.state.storage_error
        assert api.client.get(f"/api/source/{api.sid}").status_code == 503


@pytest.mark.parametrize("callback_raises", [False, True])
def test_recovered_commit_after_marker_read_errors_keeps_new_assets_and_adopts_state(api_state, monkeypatch, callback_raises):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    job, staged, job_dir = _staged_attempt(api)
    commit, read_marker = api.store.replace_sources, api.store.transaction_committed
    failures = 0

    def commit_then_fault(*args, **kwargs):
        nonlocal failures
        result = commit(*args, **kwargs)
        failures = 1 if callback_raises else 2
        if callback_raises:
            raise RuntimeError("callback failed after a real SQLite commit")
        return result

    def unreadable_twice(transaction_id):
        nonlocal failures
        if failures:
            failures -= 1
            raise sqlite3.OperationalError("temporary marker read failure")
        return read_marker(transaction_id)

    monkeypatch.setattr(api.store, "replace_sources", commit_then_fault)
    monkeypatch.setattr(api.store, "transaction_committed", unreadable_twice)
    _run(api, job, staged, job_dir)
    assert job.status == "done"
    assert job.publication.status == "committed"
    _assert_committed_assets(api)
    _assert_coherent_or_unavailable(api)
    assert not list((api.processed / "recovery").glob("*.json"))
    assert not job_dir.exists()


@pytest.mark.parametrize("actually_commit", [False, True])
def test_uncertain_marker_failure_preserves_assets_and_fails_closed(api_state, monkeypatch, actually_commit):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    job, staged, job_dir = _staged_attempt(api)
    candidate_image = api.processed / "pdf_images" / staged[0].path.stem / "image.png"
    commit = api.store.replace_sources
    read_marker = api.store.transaction_committed
    publication_attempted = False

    def fail_marker_after_publication(transaction_id):
        if publication_attempted:
            raise sqlite3.OperationalError("SQLite marker remains unavailable")
        return read_marker(transaction_id)

    def publish_or_fail(*args, **kwargs):
        nonlocal publication_attempted
        if actually_commit:
            result = commit(*args, **kwargs)
            publication_attempted = True
            return result
        publication_attempted = True
        raise RuntimeError("SQL failed before committing")

    monkeypatch.setattr(api.store, "replace_sources", publish_or_fail)
    monkeypatch.setattr(api.store, "transaction_committed", fail_marker_after_publication)
    _run(api, job, staged, job_dir)
    assert job.status == "failed"
    assert job.publication.status == "pending"
    assert candidate_image.read_bytes() == b"new derived image"
    assert job_dir.exists(), "Uncertain publication retains its recovery candidates"
    assert list((api.processed / "recovery").glob("*.json"))
    assert api.client.get("/api/graph").status_code == 503
    assert api.client.get(f"/api/source/{api.sid}").status_code == 503
    if actually_commit:
        _assert_committed_assets(api)
    else:
        with DocumentStore(api.store.path) as reopened:
            assert reopened.get_chunk(api.old.id).text == "previous material"


@pytest.mark.parametrize("fault", ["after_publish", "bump_revision"])
def test_postcommit_bookkeeping_failure_cannot_delete_live_assets(api_state, monkeypatch, fault):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    job, staged, job_dir = _staged_attempt(api)
    if fault == "after_publish":
        publish = api.main._publish_storage

        def after_publish(*args, **kwargs):
            publish(*args, **kwargs)
            _assert_committed_assets(api)
            raise RuntimeError("fault immediately after durable publication")

        monkeypatch.setattr(api.main, "_publish_storage", after_publish)
    else:
        def after_commit_revision():
            _assert_committed_assets(api)
            raise RuntimeError("revision bookkeeping failed")

        monkeypatch.setattr(api.main, "bump_revision", after_commit_revision)
    worker = Mock()
    monkeypatch.setattr(api.main.state, "illustration_worker", worker)
    _run(api, job, staged, job_dir)
    assert job.status == "failed"
    assert job.publication.status == "committed"
    _assert_committed_assets(api)
    assert api.client.get("/api/graph").status_code == 503
    worker.suspend.assert_called()
    assert job_dir.exists(), "A committed failed adoption retains its diagnostic/recovery candidates"


def test_response_validation_failure_happens_before_commit_and_cleans_only_new_assets(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    job, staged, job_dir = _staged_attempt(api)
    candidate_image = api.processed / "pdf_images" / staged[0].path.stem / "image.png"
    monkeypatch.setattr(api.main, "IngestResponse", Mock(side_effect=ValueError("invalid response fields")))
    _run(api, job, staged, job_dir)
    assert job.status == "failed"
    assert job.publication.status == "not_started"
    assert api.source.read_bytes() == b"old raw"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    assert Path(api.old.metadata["image_paths"][0]).read_bytes() == b"previous derived image"
    assert not candidate_image.exists()
    assert api.client.get("/api/graph").status_code == 200


def test_known_rollback_still_cleans_candidate_assets_and_serves_old_state(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    job, staged, job_dir = _staged_attempt(api)
    candidate_image = api.processed / "pdf_images" / staged[0].path.stem / "image.png"
    monkeypatch.setattr(api.store, "replace_sources", Mock(side_effect=sqlite3.OperationalError("database full")))
    _run(api, job, staged, job_dir)
    assert job.status == "failed"
    assert job.publication.status == "rolled_back"
    assert api.source.read_bytes() == b"old raw"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    assert Path(api.old.metadata["image_paths"][0]).read_bytes() == b"previous derived image"
    assert not candidate_image.exists()
    assert api.client.get("/api/graph").status_code == 200


def test_clear_postcommit_failure_fences_reads_and_real_old_illustration_writer(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    illustrations = api.processed / "illustrations"
    illustrations.mkdir()
    worker = IllustrationScheduler(api.graph, api.store, illustrations, {})
    key = next(iter(api.graph.graph))
    old_generation = worker._generation
    monkeypatch.setattr(api.main.state, "illustration_worker", worker)

    def fault_after_sql():
        with DocumentStore(api.store.path) as reopened:
            assert reopened.count() == 0
        raise RuntimeError("clear bookkeeping failed after real commit")

    monkeypatch.setattr(api.main, "bump_revision", fault_after_sql)
    with pytest.raises(RecoveryError, match="cleared"):
        run_locked(api.main._clear_data_locked)
    assert api.client.get("/api/graph").status_code == 503
    assert api.client.get(f"/api/source/{api.sid}").status_code == 503
    assert not api.source.exists()
    graph_path = api.processed / "graph.json"
    assert not graph_path.exists()

    # A provider thread that started before clear completes afterwards. The
    # actual scheduler must reject its old builder even though adoption failed
    # before reset() could bind it to an empty graph.
    late_output = illustrations / "late.png"
    late_output.write_bytes(b"late old illustration")
    worker._commit(old_generation, api.graph, key, late_output)
    assert not late_output.exists()
    assert not graph_path.exists(), "Old illustration work resurrected the cleared graph"


def test_clear_recovers_transient_marker_failure_and_adopts_empty_state(api_state, monkeypatch):
    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    commit, read_marker = api.store.clear_all, api.store.transaction_committed
    failures = 0

    def committed_but_callback_failed(*args, **kwargs):
        nonlocal failures
        commit(*args, **kwargs)
        failures = 1
        raise RuntimeError("clear callback failed after commit")

    def transient_marker(transaction_id):
        nonlocal failures
        if failures:
            failures -= 1
            raise sqlite3.OperationalError("temporary marker read error")
        return read_marker(transaction_id)

    monkeypatch.setattr(api.store, "clear_all", committed_but_callback_failed)
    monkeypatch.setattr(api.store, "transaction_committed", transient_marker)
    response = api.client.delete("/api/data")
    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 1
    assert api.store.count() == 0
    assert not api.source.exists()
    assert api.main.state.storage_error is None
    assert not api.main.state.graph_loaded
    assert not api.main.state.index_loaded
    assert api.client.get("/api/graph").status_code == 200
