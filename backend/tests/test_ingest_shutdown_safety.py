"""Forced shutdown cannot race asset cleanup against an ingestion thread."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from uvicorn import Config, Server

from backend.db.document_store import DocumentStore
from backend.graph.illustration_worker import IllustrationScheduler
from backend.tests.test_ingest_safety import api_state, _staged_attempt  # noqa: F401
from backend.utils.state import current_epoch


def test_force_exit_runner_cancellation_preserves_committed_image(api_state, monkeypatch):
    api = api_state
    ready, publisher_waiting, cleanup_requested = (threading.Event() for _ in range(3))
    allow_publish, committed = threading.Event(), threading.Event()
    original_publish, original_replace = api.main._publish_storage, api.store.replace_sources
    original_locked = api.main.run_locked
    fake_index = api.main.VectorIndex
    lifecycle_tasks, jobs, controller_errors = [], [], []

    class StartupIndex(fake_index):
        def load(self, _path):
            self.ids = [chunk.id for chunk in api.store.get_all_chunks()]

    def scheduler():
        worker = IllustrationScheduler(api.main.state.graph_builder, api.store, api.processed / "illustrations", {})
        worker.start()
        api.main.state.illustration_worker = worker
        return worker

    monkeypatch.setattr(api.main, "_load_config", lambda: api.main.state.config)
    monkeypatch.setattr(api.main, "DocumentStore", lambda _path: api.store)
    monkeypatch.setattr(api.main, "Embedder", lambda _name: object())
    monkeypatch.setattr(api.main, "VectorIndex", StartupIndex)
    monkeypatch.setattr(api.main, "_warm_ollama", Mock())
    monkeypatch.setattr(api.main, "_start_illustration_scheduler", scheduler)

    def delayed_publication(*args, **kwargs):
        publisher_waiting.set()
        assert allow_publish.wait(5)
        return original_publish(*args, **kwargs)

    def sql_commit(*args, **kwargs):
        value = original_replace(*args, **kwargs)
        committed.set()
        return value

    def observed_lock(function, *args, **kwargs):
        if function is api.main._cleanup_ingest_job:
            cleanup_requested.set()
        return original_locked(function, *args, **kwargs)

    monkeypatch.setattr(api.main, "_publish_storage", delayed_publication)
    monkeypatch.setattr(api.store, "replace_sources", sql_commit)
    monkeypatch.setattr(api.main, "run_locked", observed_lock)

    def release_publisher_after_cancel():
        try:
            assert cleanup_requested.wait(5), "Runner must cancel the ingest task and request cleanup"
        except BaseException as exc:
            controller_errors.append(exc)
        finally:
            allow_publish.set()

    controller = threading.Thread(target=release_publisher_after_cancel)
    controller.start()

    async def app_lifecycle():
        async with api.main.lifespan(api.main.app):
            ready.set()
            await asyncio.Event().wait()

    async def server_main():
        lifecycle_tasks.append(asyncio.create_task(app_lifecycle()))
        assert await asyncio.to_thread(ready.wait, 3)
        job, staged, job_dir = _staged_attempt(api)
        jobs.append(job)
        api.main._spawn(api.main._run_ingest_job(job, staged, job_dir, current_epoch()))
        assert await asyncio.to_thread(publisher_waiting.wait, 3)
        server = Server(Config(api.main.app, log_config=None))
        server.servers = []
        server.force_exit = True  # Actual Uvicorn second-Ctrl+C branch.
        server.lifespan = Mock()
        await server.shutdown()
        server.lifespan.shutdown.assert_not_called()
        # Returning runs real asyncio.Runner.close/_cancel_all_tasks. The
        # active publisher still owns MUTATION_LOCK when cleanup is submitted.

    try:
        asyncio.run(server_main())
    finally:
        controller.join(timeout=6)
    assert not controller.is_alive()
    assert not controller_errors
    assert committed.is_set()
    assert jobs[0].publication.status == "committed"
    with DocumentStore(api.store.path) as reopened:
        saved = reopened.get_chunk(api.old.id)
        assert saved.text == "new material"
        assert Path(saved.metadata["image_paths"][0]).read_bytes() == b"new derived image"
    assert api.source.read_bytes() == b"new raw"


@pytest.mark.parametrize("cancel_cleanup_waiter", [False, True])
def test_cleanup_first_fences_publication_even_if_raw_staging_removal_fails(
    api_state, monkeypatch, cancel_cleanup_waiter,
):
    api = api_state
    publisher_waiting, allow_publish, publisher_finished = (threading.Event() for _ in range(3))
    cleanup_entered, allow_cleanup = threading.Event(), threading.Event()
    original_locked, original_cleanup = api.main.run_locked, api.main.cleanup_attempt_assets
    original_remove = api.main.remove_staging
    job, staged, job_dir = _staged_attempt(api)
    image = api.processed / "pdf_images" / staged[0].path.stem / "image.png"

    def observed_lock(function, *args, **kwargs):
        if function is api.main._publish_ingest:
            publisher_waiting.set()
            assert allow_publish.wait(5)
            try:
                return original_locked(function, *args, **kwargs)
            finally:
                publisher_finished.set()
        return original_locked(function, *args, **kwargs)

    def delayed_cleanup(*args, **kwargs):
        cleanup_entered.set()
        assert allow_cleanup.wait(5)
        return original_cleanup(*args, **kwargs)

    def cannot_remove_staging(path, root):
        # Windows may retain an open raw file despite ignore_errors=True.
        if path != job_dir:
            original_remove(path, root)

    monkeypatch.setattr(api.main, "run_locked", observed_lock)
    monkeypatch.setattr(api.main, "cleanup_attempt_assets", delayed_cleanup)
    monkeypatch.setattr(api.main, "remove_staging", cannot_remove_staging)

    async def cancel_during_build():
        task = asyncio.create_task(api.main._run_ingest_job(job, staged, job_dir, current_epoch()))
        try:
            assert await asyncio.to_thread(publisher_waiting.wait, 3)
            task.cancel()
            assert await asyncio.to_thread(cleanup_entered.wait, 3)
            assert job.status == "cancelled"
            if cancel_cleanup_waiter:
                task.cancel()  # The submitted cleanup thread must retain its lock.
            allow_publish.set()
            assert not await asyncio.to_thread(publisher_finished.wait, 0.05)
        finally:
            allow_cleanup.set()
            allow_publish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(publisher_finished.wait, 3)

    asyncio.run(cancel_during_build())
    assert staged[0].path.exists(), "Regression must exercise a surviving raw candidate"
    assert not image.exists()
    assert job.publication.status == "not_started"
    assert job.status == "cancelled"
    assert api.source.read_bytes() == b"old raw"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    assert api.main.state.graph_builder is api.graph
    assert api.client.get("/api/graph").status_code == 200


def test_cancelled_job_cannot_restart_at_snapshot(api_state, monkeypatch):
    api = api_state
    job, staged, job_dir = _staged_attempt(api)
    job.status = "cancelled"
    parser = Mock(side_effect=AssertionError("A cancelled job must not restart parsing"))
    monkeypatch.setattr(api.main, "_ingest_single_file", parser)
    api.main._ingest(job, staged, current_epoch(), job_dir)
    parser.assert_not_called()
    assert job.status == "cancelled"
    assert job.publication.status == "not_started"
    assert api.store.get_chunk(api.old.id).text == "previous material"


@pytest.mark.parametrize("parser_raises", [False, True])
def test_cancelled_producer_removes_images_written_after_async_cleanup(api_state, monkeypatch, parser_raises):
    api = api_state
    parser_waiting, allow_parser = threading.Event(), threading.Event()
    wrote_after_cleanup, worker_finished = threading.Event(), threading.Event()
    job, staged, job_dir = _staged_attempt(api)
    image_dir = api.processed / "pdf_images" / staged[0].path.stem
    early_image, late_image = image_dir / "early.png", image_dir / "image.png"
    original_parse, original_ingest = api.main._ingest_single_file, api.main._ingest

    def paused_parser(path, config, slides):
        image_dir.mkdir(parents=True)
        early_image.write_bytes(b"before cancellation")
        parser_waiting.set()
        assert allow_parser.wait(5)
        assert not image_dir.exists(), "Async cleanup must finish before the late write"
        image_dir.mkdir(parents=True)
        late_image.write_bytes(b"after cancellation cleanup")
        wrote_after_cleanup.set()
        if parser_raises:
            raise RuntimeError("parser failed after late image output")
        return original_parse(path, config, slides)

    def tracked_worker(*args, **kwargs):
        try:
            return original_ingest(*args, **kwargs)
        finally:
            worker_finished.set()

    monkeypatch.setattr(api.main, "_ingest_single_file", paused_parser)
    monkeypatch.setattr(api.main, "_ingest", tracked_worker)

    async def cancel_then_finish_parser():
        task = asyncio.create_task(api.main._run_ingest_job(job, staged, job_dir, current_epoch()))
        try:
            assert await asyncio.to_thread(parser_waiting.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert job.status == "cancelled"
            assert not image_dir.exists()
            assert not job_dir.exists()
        finally:
            allow_parser.set()
        assert await asyncio.to_thread(worker_finished.wait, 3)

    asyncio.run(cancel_then_finish_parser())
    assert wrote_after_cleanup.is_set()
    assert not early_image.exists()
    assert not late_image.exists()
    assert not image_dir.exists()
    assert not job_dir.exists()
    assert job.publication.status == "not_started"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    assert api.source.read_bytes() == b"old raw"
    assert api.main.state.graph_builder is api.graph
