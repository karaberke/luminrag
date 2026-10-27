"""Portable storage, capability isolation and untrusted bundle validation."""
from __future__ import annotations

import io
import json
import tempfile
import uuid
import zipfile

import networkx as nx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.graph.schema import TopicAttrs
from backend.schemas import Chunk
from backend.workspace import Workspace, WorkspaceManager, export_bundle, import_bundle, read_bundle
from backend.tests.test_ingest_safety import api_state


class TinyEmbedder:
    model_name = "test-embedding"
    dimension = 3

    def embed(self, texts):
        vectors = np.array([[len(text) + 1, sum(map(ord, text)) % 17 + 1, 1] for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def embed_one(self, text): return self.embed([text])[0]
    embed_query = embed_one


@pytest.fixture
def workspaces(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    manager = WorkspaceManager({}, TinyEmbedder())
    yield manager
    manager.close()


def _filled(manager):
    ws = manager.create(str(uuid.uuid4()))
    chunks = [Chunk(id="one", text="Photosynthesis uses light", source_id="course", modality="markdown"), Chunk(id="two", text="Respiration uses stored chemical energy", source_id="course", modality="markdown")]
    ws.store.save_chunks(chunks)
    ws.vector_index.build(chunks)
    ws.index_loaded = True
    ws.graph_builder.add_node(TopicAttrs(name="Biology", source_ids=["one"]))
    ws.graph_loaded = True
    return ws


def test_export_import_preserves_exact_top_k_and_positions(workspaces):
    original = _filled(workspaces)
    key = next(iter(original.graph_builder.graph))
    original.graph_builder.graph.nodes[key].update(x=12.5, y=-5.0)
    restored = workspaces.create(str(uuid.uuid4()))
    import_bundle(restored, export_bundle(original))
    assert restored.vector_index.search("light", 2) == original.vector_index.search("light", 2)
    assert restored.graph_builder.graph.nodes[key]["x"] == 12.5
    assert restored.store.count() == 2
    assert restored.root != original.root


def test_import_reembeds_a_different_model(workspaces, monkeypatch):
    original = _filled(workspaces)
    target = workspaces.create(str(uuid.uuid4()))
    class Other(TinyEmbedder): model_name = "other-model"
    target.embedder = Other()
    import_bundle(target, export_bundle(original))
    assert target.vector_index.model_name == "other-model"
    assert len(target.vector_index) == 2


def test_legacy_bundle_ignores_retired_quiz_payload(workspaces):
    ws = workspaces.create(str(uuid.uuid4()))
    import_bundle(ws, _archive({"quizzes.json": b"retired payload is not parsed"}))
    assert ws.store.count() == 0
    assert not hasattr(ws, "quizzes")
    assert not hasattr(ws, "quiz_jobs")
    with zipfile.ZipFile(io.BytesIO(export_bundle(ws))) as archive:
        assert "quizzes.json" not in archive.namelist()


def _archive(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in {"manifest.json": '{"format":"lumin","version":1}', "chunks.jsonl": "", "graph.json": '{"nodes":[],"edges":[]}', **files}.items():
            info = zipfile.ZipInfo(name)
            info.filename = name  # Preserve attacker-supplied Windows slashes.
            archive.writestr(info, data)
    return out.getvalue()


@pytest.mark.parametrize("path", ["../escape.txt", "C:/escape.txt", "assets/../../escape.png", "assets\\escape.png", "/absolute.txt"])
def test_import_rejects_archive_path_traversal(path):
    with pytest.raises(ValueError):
        read_bundle(_archive({path: b"bad"}))


def test_import_cannot_read_arbitrary_host_image(workspaces, tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"secret-image")
    chunk = Chunk(id="a", text="text", source_id="source", modality="image", metadata={"source_file": str(secret), "image_paths": [str(secret)], "source_path": "../../secret.png"})
    ws = workspaces.create(str(uuid.uuid4()))
    import_bundle(ws, _archive({"chunks.jsonl": chunk.model_dump_json()}))
    metadata = ws.store.get_chunk("a").metadata
    assert "source_file" not in metadata
    assert not metadata.get("image_paths")
    assert "source_path" not in metadata


def test_bundle_assets_are_remapped_to_new_workspace(workspaces):
    original = _filled(workspaces)
    image = original.processed_dir / "pdf_images" / "course" / "p0001_img0.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    chunk = original.store.get_chunk("one")
    chunk.metadata["image_paths"] = [str(image)]
    original.store.save_chunks([chunk])
    data = export_bundle(original)
    target = workspaces.create(str(uuid.uuid4()))
    import_bundle(target, data)
    paths = target.store.get_chunk("one").metadata["image_paths"]
    assert paths == [str(target.processed_dir / "pdf_images/course/p0001_img0.png")]
    assert original.root.as_posix().encode() not in data


def test_lru_never_evicts_busy_workspace(workspaces):
    workspaces.max_workspaces = 1
    ws = _filled(workspaces)
    ws.active = 1
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        workspaces.create(str(uuid.uuid4()))
    assert error.value.status_code == 503
    ws.active = 0


def test_idle_eviction_waits_for_active_requests(workspaces):
    ws = _filled(workspaces)
    workspaces.idle_seconds = 1
    ws.last_used -= 10
    ws.active = 1
    workspaces.evict()
    assert ws.id in workspaces._workspaces
    ws.active = 0
    workspaces.evict()
    assert ws.id not in workspaces._workspaces
    assert not ws.root.exists()


def test_source_illustrations_only_attach_workspace_owned_assets(workspaces, tmp_path):
    from backend.utils.assets import attach_source_illustrations
    ws = _filled(workspaces)
    key = next(iter(ws.graph_builder.graph))
    ws.graph_builder.graph.nodes[key]["illustration"] = {"kind": "diagram", "hint": "A leaf."}
    chunk = ws.store.get_chunk("one")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    chunk.metadata["image_paths"] = [str(outside)]
    assert not attach_source_illustrations(ws.graph_builder.graph, [chunk], ws.processed_dir)
    image = ws.processed_dir / "pdf_images" / "leaf.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    chunk.metadata["image_paths"] = [str(image)]
    assert attach_source_illustrations(ws.graph_builder.graph, [chunk], ws.processed_dir)
    assert ws.graph_builder.graph.nodes[key]["illustration_path"] == "assets/pdf_images/leaf.png"


def test_workspace_disk_budget_bounds_staging_and_publication(workspaces, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from backend import main
    from backend.utils.recovery import PublicationOutcome
    from backend.utils.uploads import UploadRejected
    from backend.workspace import current_workspace, copy_workspace_upload
    ws = workspaces.create(str(uuid.uuid4()))
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    monkeypatch.setenv("LUMIN_MAX_WORKSPACE_BYTES", str(ws.disk_bytes() + 32))
    token = current_workspace.set(ws)
    try:
        first = ws.staging_dir / "first.txt"
        copy_workspace_upload(io.BytesIO(b"x" * 16), first, 50)
        with pytest.raises(UploadRejected):
            copy_workspace_upload(io.BytesIO(b"y" * 32), ws.staging_dir / "second.txt", 50)
        assert not (ws.staging_dir / "second.txt").exists()
        publication = SimpleNamespace(publish=Mock())
        outcome = PublicationOutcome()
        with pytest.raises(UploadRejected):
            main._publish_storage(publication, [(first, ws.processed_dir / "candidate.txt")], lambda _tx: None, outcome=outcome, expected_sql_bytes=1)
        publication.publish.assert_not_called()
        assert outcome.status == "not_started"
    finally:
        current_workspace.reset(token)


def test_hosted_ingest_keeps_original_only_in_browser(workspaces, monkeypatch):
    from backend import main
    from backend.graph.graph_builder import ApplyResult
    from backend.utils.uploads import StagedFile, new_job_staging_dir, source_identity, staged_path
    from backend.workspace import current_workspace
    ws = workspaces.create(str(uuid.uuid4()))
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    def pipeline(chunks, filenames, graph_path, config, builder, embedder, **kwargs):
        builder.add_node(TopicAttrs(name="Light", source_ids=[chunks[0].id]))
        builder.save()
        return ApplyResult(topics_added=1)
    monkeypatch.setattr(main, "_run_graph_pipeline", pipeline)
    identifier = str(uuid.uuid4())
    sid = source_identity("lesson.txt")
    directory = new_job_staging_dir(ws.staging_dir, identifier)
    path = staged_path(directory, 0, sid + "_" + identifier + ".txt")
    path.write_text("Light powers photosynthesis.", encoding="utf-8")
    staged = StagedFile(path, "lesson.txt", "lesson.txt", sid)
    job = main._IngestJob(identifier)
    token = current_workspace.set(ws)
    try:
        main._ingest(job, [staged], ws.epoch, directory)
        assert job.status == "done"
        assert ws.store.count() == 1
        assert not list(ws.raw_dir.rglob("*"))
        assert ws.vector_index.search("light", 1)
    finally:
        current_workspace.reset(token)


@pytest.fixture
def hosted_client(workspaces, monkeypatch):
    from backend import main
    from backend.utils.auth import AuthMiddleware
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    monkeypatch.setattr(main, "_workspace_manager", workspaces)
    app = FastAPI()
    app.router.routes.extend(main.app.router.routes)
    app.add_middleware(AuthMiddleware, allowed_origins=["http://localhost:5173"])
    app.add_middleware(main._WorkspaceMiddleware)
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def _headers(identifier):
    return {"X-Workspace-Id": identifier, "Origin": "http://localhost:5173"}


def test_hosted_status_requires_no_global_login_even_behind_proxy(hosted_client, monkeypatch):
    monkeypatch.setenv("LUMINRAG_API_TOKEN", "self-host-token-must-not-gate-public-service")
    response = hosted_client.get("/api/auth/status")
    assert response.status_code == 200
    assert response.json() == {"authenticated": True, "token_required": False}
    # An application-login-free status must not expose material without a
    # capability on the actual data endpoint.
    assert hosted_client.get("/api/graph").status_code == 400


def test_api_hydration_isolation_delete_and_missing_key(hosted_client, workspaces):
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    a, b = _headers(first), _headers(second)
    assert hosted_client.get("/api/workspace/status", headers=a).json() == {"loaded": False, "version": 0}
    response = hosted_client.get("/api/graph", headers=a)
    assert response.status_code == 409 and response.json()["code"] == "workspace_not_loaded"
    assert hosted_client.post("/api/workspace/import", headers=a, data={"empty": "true"}).status_code == 200
    assert hosted_client.post("/api/workspace/import", headers=b, data={"empty": "true"}).status_code == 200
    response = hosted_client.post("/api/graph/ops", headers=a, json={"ops": [{"op": "add_node", "node": {"id": "note-one", "name": "Remember", "node_type": "note", "body": "Light powers photosynthesis."}}]})
    assert response.status_code == 200, response.text
    assert len(hosted_client.get("/api/graph", headers=a).json()["nodes"]) == 1
    assert hosted_client.get("/api/graph", headers=b).json()["nodes"] == []
    assert hosted_client.post("/api/query", headers=a, json={"query": "Why?"}).status_code == 401
    assert hosted_client.delete("/api/data", headers=b).status_code == 200
    assert workspaces.get(first).store.count() == 1
    assert workspaces.get(second).store.count() == 0
    inverse = response.json()["inverses"]
    undo = hosted_client.post("/api/graph/ops", headers=a, json={"ops": inverse})
    assert undo.status_code == 200, undo.text
    assert workspaces.get(first).store.count() == 0
    assert hosted_client.get("/api/graph", headers=a).json()["nodes"] == []


@pytest.mark.parametrize("path", ["/static/illustrations/file.png", "/api/source/source", "/api/upload/presign", "/api/ingest/from-s3"])
def test_hosted_disables_legacy_global_paths(hosted_client, path):
    assert hosted_client.get(path, headers=_headers(str(uuid.uuid4()))).status_code == 404


def test_hosted_rejects_non_uuid_workspace(hosted_client):
    assert hosted_client.get("/api/graph", headers={"X-Workspace-Id": "local"}).status_code == 400


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/quiz"),
    ("POST", "/api/quiz/generate"),
    ("GET", "/api/quiz/jobs/old"),
    ("POST", "/api/quiz/old/grade"),
    ("GET", "/api/quiz/old/hint/question"),
    ("POST", "/api/quiz/old/deep-dive/question"),
])
def test_retired_quiz_routes_are_not_available_hosted(hosted_client, method, path):
    # Retired endpoints need neither a loaded workspace nor provider headers.
    response = hosted_client.request(method, path)
    assert response.status_code == 404


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/quiz"),
    ("POST", "/api/quiz/generate"),
    ("GET", "/api/quiz/jobs/old"),
    ("POST", "/api/quiz/old/grade"),
])
def test_retired_quiz_routes_are_not_available_selfhost(api_state, method, path):
    assert api_state.client.request(method, path).status_code == 404


def test_invalid_embedding_import_preserves_loaded_workspace(hosted_client, workspaces):
    ws = _filled(workspaces)
    before = ws.vector_index.search("light", 2)
    bad = _archive({"manifest.json": json.dumps({"format": "lumin", "version": 1, "embedding_model": TinyEmbedder.model_name}), "chunks.jsonl": ws.store.get_chunk("one").model_dump_json(), "embeddings.npy": b"not-a-numpy-file"})
    response = hosted_client.post("/api/workspace/import", headers=_headers(ws.id), files={"file": ("broken.lumin", bad, "application/zip")})
    assert response.status_code == 422
    assert workspaces.get(ws.id) is ws
    assert ws.vector_index.search("light", 2) == before
    assert ws.store.count() == 2


def test_graph_batch_failure_does_not_publish_partial_notes(hosted_client, workspaces):
    ws = workspaces.create(str(uuid.uuid4()))
    response = hosted_client.post("/api/graph/ops", headers=_headers(ws.id), json={"ops": [{"op": "add_node", "node": {"id": "new", "name": "A note", "node_type": "note", "body": "Body"}}, {"op": "add_edge", "edge": {"source": "new", "target": "missing", "relation": "RELATED_TO"}}]})
    assert response.status_code == 422
    assert ws.store.count() == 0
    assert ws.graph_builder.node_count() == 0


def test_forged_inverse_payload_is_validated_before_publication(hosted_client, workspaces):
    ws = _filled(workspaces)
    key = next(iter(ws.graph_builder.graph))
    original = dict(ws.graph_builder.graph.nodes[key])
    operation = {"op": "restore_nodes", "ids": [key], "nodes": [{"id": key, "attrs": {**original, "illustration_path": {"unsafe": "value"}}}], "edges": []}
    response = hosted_client.post("/api/graph/ops", headers=_headers(ws.id), json={"ops": [operation]})
    assert response.status_code == 422
    assert dict(ws.graph_builder.graph.nodes[key]) == original


def test_concurrent_request_keys_stay_out_of_workspace_config(hosted_client, workspaces, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from backend import main
    from backend.schemas import GenerationResult, ReflectionVerdict, RetrievalResult
    first, second = _filled(workspaces), _filled(workspaces)
    barrier = threading.Barrier(2)
    def reason(**kwargs):
        barrier.wait(timeout=10)
        return RetrievalResult(chunks=[], subgraph=[], routing_mode="dense")
    def generate(**kwargs):
        return GenerationResult(answer=kwargs["config"]["generator"]["api_key"], routing_mode="dense", reflection=ReflectionVerdict(needs_retrieval=True, is_relevant=True, is_supported=True, is_useful=True, reasoning="test"), evidence_chunk_ids=[])
    monkeypatch.setattr(main, "reason", reason)
    monkeypatch.setattr(main, "generate", generate)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "forbidden-fallback")
    def ask(ws, key):
        headers = {**_headers(ws.id), "X-LLM-Provider": "anthropic", "X-LLM-Key": key, "X-LLM-Model-Fast": "test-fast", "X-LLM-Model-Smart": "test-smart"}
        return hosted_client.post("/api/query", headers=headers, json={"query": "Explain light", "query_mode": "fast"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(ask, first, "fake-provider-one")
        two = executor.submit(ask, second, "fake-provider-two")
        responses = [one.result(), two.result()]
    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["content"] for response in responses] == ["fake-provider-one", "fake-provider-two"]
    assert "api_key" not in first.config.get("generator", {})
    assert "api_key" not in second.config.get("generator", {})


def test_selfhost_empty_import_publishes_from_allowed_staging_root(api_state):
    response = api_state.client.post("/api/workspace/import", data={"empty": "true"})
    assert response.status_code == 200, response.text
    assert response.json()["loaded"] is True
    assert api_state.store.count() == 0
    assert api_state.main.state.graph_builder.node_count() == 0
    assert not list(api_state.staging.iterdir())
    assert not api_state.main.state.storage_error


def test_selfhost_import_preserves_assets_after_candidate_cleanup(api_state, workspaces, monkeypatch):
    original = _filled(workspaces)
    asset = original.processed_dir / "pdf_images" / "course" / "figure.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"safe-figure")
    chunk = original.store.get_chunk("one")
    chunk.metadata["image_paths"] = [str(asset)]
    original.store.save_chunks([chunk])
    monkeypatch.setattr(api_state.main.state, "embedder", TinyEmbedder())
    response = api_state.client.post("/api/workspace/import", files={"file": ("course.lumin", export_bundle(original), "application/zip")})
    assert response.status_code == 200, response.text
    assert api_state.store.count() == 2
    restored = api_state.store.get_chunk("one")
    final_asset = api_state.processed / "pdf_images" / "course" / "figure.png"
    assert restored.metadata["image_paths"] == [str(final_asset)]
    assert final_asset.read_bytes() == b"safe-figure"
    assert api_state.main.state.vector_index.search("light", 1)
    assert not list(api_state.staging.iterdir())
