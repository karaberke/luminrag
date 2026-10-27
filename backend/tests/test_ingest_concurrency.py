"""Deterministic ingest barriers keep reads/edits responsive without stale publication."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from backend.graph.graph_builder import GraphBuilder
from backend.graph.illustration_worker import IllustrationScheduler
from backend.graph.schema import ContentAttrs, Illustration
from backend.tests.test_ingest_safety import api_state, _staged_attempt  # noqa: F401
from backend.utils.state import current_epoch


@contextmanager
def _paused_ingest(api, monkeypatch, stage="parser"):
    entered, release = Event(), Event()
    function_name = "_ingest_single_file" if stage == "parser" else "_run_graph_pipeline"
    process = getattr(api.main, function_name)

    def blocked_parser(*args, **kwargs):
        entered.set()
        assert release.wait(10), "Test must release the paused parser"
        return process(*args, **kwargs)

    monkeypatch.setattr(api.main, function_name, blocked_parser)
    job, staged, job_dir = _staged_attempt(api)
    epoch = current_epoch()
    with ThreadPoolExecutor(max_workers=3) as pool:
        future = pool.submit(lambda: asyncio.run(api.main._run_ingest_job(job, staged, job_dir, epoch)))
        try:
            assert entered.wait(5), "Ingest must reach its parser"
            yield job, pool
        finally:
            release.set()
            future.result(timeout=10)


def test_graph_read_completes_while_ingest_parser_is_paused(api_state, monkeypatch):
    with _paused_ingest(api_state, monkeypatch) as (job, pool):
        response = pool.submit(api_state.client.get, "/api/graph").result(timeout=3)
        assert response.status_code == 200
        assert any(node["name"] == "Old concept" for node in response.json()["nodes"])
    assert job.status == "done"


@pytest.mark.parametrize("stage", ["parser", "graph"])
def test_manual_edit_during_ingest_is_preserved_and_stale_ingest_fails(api_state, monkeypatch, stage):
    api = api_state
    with _paused_ingest(api, monkeypatch, stage) as (job, pool):
        response = pool.submit(api.client.post, "/api/graph/node", json={"name": "Manual preserved", "node_type": "topic"}).result(timeout=3)
        assert response.status_code == 201
    assert job.status == "failed"
    assert "changed" in job.error.lower()
    assert api.source.read_bytes() == b"old raw"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    names = {attrs["name"] for _, attrs in api.main.state.graph_builder.graph.nodes(data=True)}
    assert "Manual preserved" in names
    assert "New concept" not in names
    persisted = GraphBuilder(api.processed / "graph.json")
    assert "Manual preserved" in {attrs["name"] for _, attrs in persisted.graph.nodes(data=True)}


@pytest.mark.parametrize("stage", ["parser", "graph"])
def test_clear_during_ingest_prevents_source_or_graph_resurrection(api_state, monkeypatch, stage):
    api = api_state
    with _paused_ingest(api, monkeypatch, stage) as (job, pool):
        response = pool.submit(api.client.delete, "/api/data").result(timeout=3)
        assert response.status_code == 200
        assert api.store.count() == 0
    assert job.status == "cancelled"
    assert api.store.count() == 0
    assert not api.source.exists()
    assert api.main.state.graph_builder.node_count() == 0
    assert not (api.processed / "faiss.index").exists()
    assert not (api.processed / "graph.json").exists()


@pytest.mark.parametrize("change_hint", [False, True])
def test_background_illustration_is_merged_only_for_unchanged_content(api_state, monkeypatch, change_hint):
    api = api_state
    key = api.graph.add_node(ContentAttrs(
        name="Manual figure", origin="manual", summary="Stable content",
        illustration=Illustration(kind="diagram", hint="old hint"),
    ))
    api.graph.save()
    if change_hint:
        pipeline = api.main._run_graph_pipeline

        def changed_hint_pipeline(*args, **kwargs):
            result = pipeline(*args, **kwargs)
            builder = args[4]
            builder.graph.nodes[key]["illustration"] = {"kind": "diagram", "hint": "new hint"}
            builder.save()
            return result

        monkeypatch.setattr(api.main, "_run_graph_pipeline", changed_hint_pipeline)

    image = api.processed / "illustrations" / "completed.svg"
    image.parent.mkdir(parents=True)
    image.write_text("<svg/>", encoding="utf-8")
    scheduler = IllustrationScheduler(api.graph, api.store, image.parent, {})
    with _paused_ingest(api, monkeypatch) as (job, pool):
        pool.submit(scheduler._commit, 0, api.graph, key, image).result(timeout=3)
        assert api.graph.graph.nodes[key]["illustration_path"] == "/static/illustrations/completed.svg"
    assert job.status == "done"
    attrs = api.main.state.graph_builder.graph.nodes[key]
    assert attrs["illustration_path"] == (None if change_hint else "/static/illustrations/completed.svg")
    assert attrs["illustration"]["hint"] == ("new hint" if change_hint else "old hint")
    persisted = GraphBuilder(api.processed / "graph.json")
    assert persisted.graph.nodes[key]["illustration_path"] == attrs["illustration_path"]
