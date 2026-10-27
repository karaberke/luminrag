"""
Tests for backend/utils/uploads.py — upload validation, identity, bounded
copies, raw-source resolution, S3 bucket policy and issued-key registry.

Filesystem work happens only under tmp_path; S3 is a hand-written fake.
"""

from __future__ import annotations

import io
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.schemas import Chunk
from backend.utils import uploads
from backend.utils.uploads import (
    IssuedUploadRegistry,
    UploadBodyLimitMiddleware,
    UploadRejected,
    commit_raw_source,
    copy_bounded,
    download_issued_object,
    find_legacy_raw_source,
    remap_chunks,
    remove_staging,
    resolve_raw_source,
    s3_upload_bucket,
    sanitize_filename,
    source_identity,
    validate_batch_size,
    validate_declared_size,
    validate_file,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateFile:
    @pytest.mark.parametrize(
        "name, ctype",
        [
            ("notes.pdf", "application/pdf"),
            ("photo.JPG", "image/jpeg"),
            ("talk.m4a", "audio/x-m4a"),
            ("lecture.mp4", "video/mp4"),
            ("deck.pdf", "application/octet-stream"),
            ("deck.pdf", None),
        ],
    )
    def test_accepts_allowlisted(self, name, ctype):
        assert validate_file(name, ctype) == name

    @pytest.mark.parametrize("name", ["x.svg", "x.js", "x.exe", "noext"])
    def test_rejects_unsupported_extensions(self, name):
        with pytest.raises(UploadRejected) as exc:
            validate_file(name, "application/octet-stream")
        assert exc.value.status == 415

    @pytest.mark.parametrize("ctype", ["text/html", "image/svg+xml", "application/xhtml+xml"])
    def test_rejects_active_content_types(self, ctype):
        with pytest.raises(UploadRejected):
            validate_file("innocent.pdf", ctype)

    def test_rejects_mismatched_type(self):
        with pytest.raises(UploadRejected):
            validate_file("clip.mp4", "application/pdf")


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("../../etc/passwd.pdf", "passwd.pdf"),
            ("C:\\Users\\x\\notes.pdf", "notes.pdf"),
            ("a#b?c%d&e.pdf", "a_b_c_d_e.pdf"),
            ("ctrl\x00\x1fname.pdf", "ctrlname.pdf"),
        ],
    )
    def test_strips_paths_and_specials(self, raw, expected):
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "..", "/", "   "])
    def test_rejects_empty(self, raw):
        with pytest.raises(UploadRejected):
            sanitize_filename(raw)

    def test_long_names_keep_extension(self):
        name = sanitize_filename("a" * 500 + ".pdf")
        assert name.endswith(".pdf") and len(name) <= 180


class TestLimits:
    def test_batch_size(self):
        validate_batch_size(uploads.MAX_BATCH_FILES)
        with pytest.raises(UploadRejected) as exc:
            validate_batch_size(uploads.MAX_BATCH_FILES + 1)
        assert exc.value.status == 413
        with pytest.raises(UploadRejected):
            validate_batch_size(0)

    def test_max_upload_bytes_env(self, monkeypatch):
        monkeypatch.delenv(uploads.MAX_UPLOAD_BYTES_ENV, raising=False)
        assert uploads.max_upload_bytes() == 64 * 1024 * 1024
        monkeypatch.setenv(uploads.MAX_UPLOAD_BYTES_ENV, "1000")
        assert uploads.max_upload_bytes() == 1000
        monkeypatch.setenv(uploads.MAX_UPLOAD_BYTES_ENV, "garbage")
        assert uploads.max_upload_bytes() == 64 * 1024 * 1024

    def test_declared_size(self):
        validate_declared_size(10, limit=10)
        for bad in (0, -1):
            with pytest.raises(UploadRejected):
                validate_declared_size(bad, limit=10)
        with pytest.raises(UploadRejected) as exc:
            validate_declared_size(11, limit=10)
        assert exc.value.status == 413


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_same_stem_different_extension_do_not_collide(self):
        assert source_identity("lecture.pdf") != source_identity("lecture.mp4")

    def test_deterministic(self):
        assert source_identity("lecture.pdf") == source_identity("lecture.pdf")

    def test_identity_is_safe_path_component(self):
        sid = source_identity("../we ird#name.pdf")
        assert uploads.is_safe_source_id(sid)
        assert "/" not in sid and "\\" not in sid

    def test_remap_chunks(self):
        chunks = [
            Chunk(id="lecture_chunk_0", text="a", source_id="lecture", modality="pdf"),
            Chunk(id="other", text="b", source_id="lecture", modality="pdf"),
        ]
        sid = source_identity("lecture.pdf")
        out = remap_chunks(chunks, sid, "lecture.pdf", f"{sid}/lecture.pdf")
        assert [c.id for c in out] == [f"{sid}_chunk_0", f"{sid}_other"]
        assert all(c.source_id == sid for c in out)
        assert out[0].metadata["original_filename"] == "lecture.pdf"
        assert out[0].metadata["source_path"] == f"{sid}/lecture.pdf"
        assert chunks[0].metadata == {}  # inputs untouched


# ---------------------------------------------------------------------------
# Bounded copies + staging + raw sources
# ---------------------------------------------------------------------------

class TestCopyBounded:
    def test_copies_within_limit(self, tmp_path):
        dest = tmp_path / "f.pdf"
        assert copy_bounded(io.BytesIO(b"x" * 10), dest, limit=10) == 10
        assert dest.read_bytes() == b"x" * 10

    def test_over_limit_removes_partial(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uploads, "COPY_CHUNK_BYTES", 4)
        dest = tmp_path / "f.pdf"
        with pytest.raises(UploadRejected) as exc:
            copy_bounded(io.BytesIO(b"x" * 11), dest, limit=10)
        assert exc.value.status == 413
        assert not dest.exists()

    def test_reads_in_bounded_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uploads, "COPY_CHUNK_BYTES", 4)
        sizes: list[int] = []

        class Spy(io.BytesIO):
            def read(self, n=-1):
                sizes.append(n)
                return super().read(n)

        copy_bounded(Spy(b"x" * 10), tmp_path / "f.pdf", limit=100)
        assert sizes and all(n == 4 for n in sizes)

    def test_expected_size_mismatch(self, tmp_path):
        with pytest.raises(UploadRejected):
            copy_bounded(io.BytesIO(b"x" * 5), tmp_path / "f.pdf", limit=100, expected_size=6)
        assert not (tmp_path / "f.pdf").exists()

    def test_never_clobbers_existing(self, tmp_path):
        dest = tmp_path / "f.pdf"
        dest.write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            copy_bounded(io.BytesIO(b"new"), dest, limit=100)
        assert dest.read_bytes() == b"keep"


class TestStagingAndRaw:
    def test_remove_staging_refuses_outside_root(self, tmp_path):
        root = tmp_path / "staging"
        outside = tmp_path / "keep"
        outside.mkdir()
        (outside / "f").write_text("x")
        root.mkdir()
        remove_staging(outside, root)
        remove_staging(root, root)
        assert (outside / "f").exists() and root.exists()

        job = root / "job1"
        (job / "00").mkdir(parents=True)
        remove_staging(job, root)
        assert not job.exists()

    def test_commit_and_resolve(self, tmp_path):
        raw = tmp_path / "raw"
        staged = tmp_path / "s.pdf"
        staged.write_bytes(b"%PDF")
        sid = source_identity("notes.pdf")
        dest = commit_raw_source(staged, raw, sid, "notes.pdf")
        assert dest == (raw / sid / "notes.pdf").resolve()
        assert resolve_raw_source(raw, f"{sid}/notes.pdf") == dest
        # re-commit replaces in place
        staged.write_bytes(b"%PDF-2")
        commit_raw_source(staged, raw, sid, "notes.pdf")
        assert dest.read_bytes() == b"%PDF-2"
        assert [p.name for p in (raw / sid).iterdir()] == ["notes.pdf"]

    def test_commit_rejects_unsafe_identity(self, tmp_path):
        staged = tmp_path / "s.pdf"
        staged.write_bytes(b"x")
        with pytest.raises(UploadRejected):
            commit_raw_source(staged, tmp_path / "raw", "../escape", "s.pdf")
        with pytest.raises(UploadRejected):
            commit_raw_source(staged, tmp_path / "raw", "ok", "../s.pdf")

    def test_resolve_rejects_traversal_and_active_types(self, tmp_path):
        raw = tmp_path / "raw"
        (raw / "src").mkdir(parents=True)
        (raw / "src" / "page.html").write_text("<script>")
        (tmp_path / "secret.pdf").write_bytes(b"x")
        assert resolve_raw_source(raw, "src/page.html") is None
        assert resolve_raw_source(raw, "../secret.pdf") is None
        assert resolve_raw_source(raw, str(tmp_path / "secret.pdf")) is None
        assert resolve_raw_source(raw, None) is None

    def test_legacy_lookup(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "lecture_01.pdf").write_bytes(b"%PDF")
        (raw / "evil.html").write_text("<script>")
        assert find_legacy_raw_source(raw, "lecture_01") == (raw / "lecture_01.pdf").resolve()
        assert find_legacy_raw_source(raw, "lecture_01.pdf") == (raw / "lecture_01.pdf").resolve()
        assert find_legacy_raw_source(raw, "evil") is None
        assert find_legacy_raw_source(raw, "../raw/lecture_01") is None


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

class TestBucketPolicy:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.delenv(uploads.S3_UPLOAD_BUCKET_ENV, raising=False)
        monkeypatch.delenv(uploads.S3_FRONTEND_BUCKET_ENV, raising=False)
        monkeypatch.setenv("S3_BUCKET", "luminrag-frontend-bucket")  # legacy var is ignored

    def test_required(self):
        with pytest.raises(UploadRejected) as exc:
            s3_upload_bucket()
        assert exc.value.status == 503

    def test_refuses_legacy_frontend_bucket(self, monkeypatch):
        monkeypatch.setenv(uploads.S3_UPLOAD_BUCKET_ENV, "luminrag-frontend-bucket")
        with pytest.raises(UploadRejected):
            s3_upload_bucket()

    def test_refuses_configured_frontend_bucket(self, monkeypatch):
        monkeypatch.setenv(uploads.S3_FRONTEND_BUCKET_ENV, "site-bucket")
        monkeypatch.setenv(uploads.S3_UPLOAD_BUCKET_ENV, "site-bucket")
        with pytest.raises(UploadRejected):
            s3_upload_bucket()

    def test_dedicated_bucket(self, monkeypatch):
        monkeypatch.setenv(uploads.S3_UPLOAD_BUCKET_ENV, "uploads-bucket")
        assert s3_upload_bucket() == "uploads-bucket"


class TestRegistry:
    def test_issue_and_single_use(self):
        reg = IssuedUploadRegistry(ttl_seconds=60)
        rec = reg.issue("a.pdf", 10, now=0)
        assert rec.key.startswith("uploads/") and rec.key.endswith("/a.pdf")
        assert reg.claim([rec.key], now=1) == [rec]
        with pytest.raises(UploadRejected) as exc:
            reg.claim([rec.key], now=2)
        assert exc.value.status == 404

    def test_unknown_key_rejects_whole_batch(self):
        reg = IssuedUploadRegistry()
        rec = reg.issue("a.pdf", 10)
        with pytest.raises(UploadRejected):
            reg.claim([rec.key, "uploads/forged/a.pdf"])
        assert reg.claim([rec.key]) == [rec]  # untouched by the failed claim

    def test_duplicate_keys(self):
        reg = IssuedUploadRegistry()
        rec = reg.issue("a.pdf", 10)
        with pytest.raises(UploadRejected):
            reg.claim([rec.key, rec.key])

    def test_expired(self):
        reg = IssuedUploadRegistry(ttl_seconds=10)
        rec = reg.issue("a.pdf", 10, now=0)
        with pytest.raises(UploadRejected):
            reg.claim([rec.key], now=11)

    def test_release_restores(self):
        reg = IssuedUploadRegistry(ttl_seconds=10)
        rec = reg.issue("a.pdf", 10, now=0)
        reg.claim([rec.key], now=1)
        reg.release([rec], now=2)
        assert reg.claim([rec.key], now=3) == [rec]

    def test_capacity(self):
        reg = IssuedUploadRegistry(max_entries=1)
        reg.issue("a.pdf", 1)
        with pytest.raises(UploadRejected) as exc:
            reg.issue("b.pdf", 1)
        assert exc.value.status == 429


class _FakeBody(io.BytesIO):
    pass


class _FakeS3:
    def __init__(self, length: int, data: bytes, ctype: str = "application/pdf") -> None:
        self.length = length
        self.data = data
        self.ctype = ctype
        self.get_calls: list[dict] = []

    def head_object(self, Bucket, Key):
        return {"ContentLength": self.length, "ContentType": self.ctype, "ETag": '"abc"'}

    def get_object(self, **params):
        self.get_calls.append(params)
        return {"Body": _FakeBody(self.data)}


class TestDownloadIssuedObject:
    def _record(self, size: int):
        return IssuedUploadRegistry().issue("a.pdf", size)

    def test_happy_path_pins_etag(self, tmp_path):
        s3 = _FakeS3(4, b"%PDF")
        dest = tmp_path / "a.pdf"
        assert download_issued_object(s3, "b", self._record(4), dest, limit=100) == 4
        assert s3.get_calls[0]["IfMatch"] == '"abc"'

    def test_size_mismatch_rejected_before_download(self, tmp_path):
        s3 = _FakeS3(999, b"x" * 999)
        with pytest.raises(UploadRejected):
            download_issued_object(s3, "b", self._record(4), tmp_path / "a.pdf", limit=100)
        assert s3.get_calls == []

    def test_over_limit_rejected_before_download(self, tmp_path):
        s3 = _FakeS3(200, b"x" * 200)
        with pytest.raises(UploadRejected):
            download_issued_object(s3, "b", self._record(200), tmp_path / "a.pdf", limit=100)
        assert s3.get_calls == []

    def test_active_content_type_rejected(self, tmp_path):
        s3 = _FakeS3(4, b"<svg", ctype="image/svg+xml")
        with pytest.raises(UploadRejected):
            download_issued_object(s3, "b", self._record(4), tmp_path / "a.pdf", limit=100)
        assert s3.get_calls == []

    def test_body_longer_than_head_is_bounded(self, tmp_path):
        s3 = _FakeS3(4, b"x" * 50)
        dest = tmp_path / "a.pdf"
        with pytest.raises(UploadRejected):
            download_issued_object(s3, "b", self._record(4), dest, limit=100)
        assert not dest.exists()


class TestBodyLimitMiddleware:
    def _client(self):
        app = FastAPI()

        @app.post("/api/ingest")
        def ingest():
            return {"ok": True}

        app.add_middleware(UploadBodyLimitMiddleware, paths=("/api/ingest",))
        return TestClient(app)

    def test_oversized_rejected(self, monkeypatch):
        monkeypatch.setenv(uploads.MAX_UPLOAD_BYTES_ENV, "10")
        big = b"x" * (uploads.MAX_BATCH_FILES * 10 + uploads._MULTIPART_OVERHEAD_BYTES + 1)
        assert self._client().post("/api/ingest", content=big).status_code == 413

    def test_small_allowed(self):
        assert self._client().post("/api/ingest", content=b"x").status_code == 200


@pytest.fixture
def api_state(monkeypatch, tmp_path):
    """Load actual routes with dotenv, startup, model work and AWS disabled."""
    from backend.db.document_store import DocumentStore
    from backend.graph.graph_builder import ApplyResult, GraphBuilder
    from backend.graph.graph_export import export_graph
    from backend.graph.schema import ContentAttrs

    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path.name == ".env" else original_exists(path))
    import backend.main as main

    monkeypatch.setenv("LUMINRAG_API_TOKEN", "fixture-token")
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    staging = tmp_path / "staging"
    for path in (raw, processed, staging):
        path.mkdir()
    monkeypatch.setattr(main, "_RAW_DIR", raw)
    monkeypatch.setattr(main, "_PROCESSED_DIR", processed)
    monkeypatch.setattr(main, "_STAGING_DIR", staging)
    monkeypatch.setattr(main, "_ILLUS_DIR", processed / "illustrations")
    monkeypatch.setattr(main, "_jobs", {})
    monkeypatch.setattr(main, "_background_tasks", set())
    monkeypatch.setattr(main, "_ingest_queue", asyncio.Lock())
    monkeypatch.setattr(main, "_issued_uploads", IssuedUploadRegistry())

    class FakeIndex:
        def __init__(self, embedder):
            self.ids = []

        def build(self, chunks, **kwargs):
            self.ids = [chunk.id for chunk in chunks]

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-index")
            path.with_suffix(".json").write_text(json.dumps(self.ids), encoding="utf-8")

    monkeypatch.setattr(main, "VectorIndex", FakeIndex)
    monkeypatch.setattr(main, "warm_graph_retriever", Mock())
    monkeypatch.setattr(main, "_get_s3", Mock(side_effect=AssertionError("Real S3 access forbidden in tests")))
    store = DocumentStore(tmp_path / "chunks.db")
    config = {
        "vector_db": {"path": str(processed / "faiss.index")},
        "graph_db": {"path": str(processed / "graph.json")},
    }
    sid = source_identity("notes.pdf")
    old_asset = processed / "pdf_images" / "previous_attempt" / "image.png"
    old_asset.parent.mkdir(parents=True)
    old_asset.write_bytes(b"previous derived image")
    old = Chunk(id=f"{sid}_chunk_0", source_id=sid, modality="pdf", text="previous material", metadata={"source_path": f"{sid}/notes.pdf", "image_paths": [str(old_asset)]})
    store.save_chunks([old])
    source = raw / sid / "notes.pdf"
    source.parent.mkdir()
    source.write_bytes(b"old raw")
    graph = GraphBuilder(processed / "graph.json")
    graph.add_node(ContentAttrs(name="Old concept", source_ids=[old.id]))
    graph.save()
    index = FakeIndex(None)
    index.build([old])
    index.save(processed / "faiss.index")
    for name, value in {
        "store": store, "config": config, "embedder": object(), "vector_index": index,
        "index_loaded": True, "graph_loaded": True, "graph_builder": graph, "illustration_worker": None,
        "storage_error": None,
    }.items():
        monkeypatch.setattr(main.state, name, value)

    def parse(path, _config, _is_slides):
        # Processor-created assets have a different path for each attempt.
        asset = processed / "pdf_images" / path.stem
        asset.mkdir(parents=True, exist_ok=True)
        (asset / "image.png").write_bytes(b"new derived image")
        return [Chunk(id=f"{path.stem}_chunk_0", source_id=path.stem, modality="pdf", text="new material", metadata={"image_paths": [str(asset / "image.png")]})]

    def graph_pipeline(chunks, _names, path, _config, builder, _embedder, **kwargs):
        for chunk in chunks:
            builder.add_node(ContentAttrs(name="New concept", source_ids=[chunk.id]))
        builder.save()
        export_graph(builder.graph, path)
        return ApplyResult(contents_added=len(chunks))

    monkeypatch.setattr(main, "_ingest_single_file", parse)
    monkeypatch.setattr(main, "_run_graph_pipeline", graph_pipeline)

    @asynccontextmanager
    async def no_startup(_app):
        yield

    monkeypatch.setattr(main.app.router, "lifespan_context", no_startup)
    fixture = SimpleNamespace(main=main, store=store, sid=sid, source=source, old=old, index=index, graph=graph, staging=staging, processed=processed, raw=raw)
    with TestClient(main.app, base_url="http://localhost", headers={"Authorization": "Bearer fixture-token"}) as client:
        fixture.client = client
        yield fixture
    store.close()


def _staged_attempt(api):
    job_id = "attempt"
    job_dir = api.staging / job_id
    path = job_dir / "00" / f"{api.sid}_{job_id}.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new raw")
    staged = uploads.StagedFile(path, "notes.pdf", "notes.pdf", api.sid)
    return api.main._IngestJob(job_id), [staged], job_dir


def _assert_old_state(api, old_graph_bytes, old_index_bytes):
    assert api.source.read_bytes() == b"old raw"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    assert api.main.state.graph_builder is api.graph
    assert api.main.state.vector_index is api.index
    assert (api.processed / "graph.json").read_bytes() == old_graph_bytes
    assert (api.processed / "faiss.index").read_bytes() == old_index_bytes
    assert Path(api.old.metadata["image_paths"][0]).read_bytes() == b"previous derived image"


@pytest.mark.parametrize("fault", ["embedding", "graph", "store", "file_publish"])
def test_failed_reingest_preserves_all_committed_data(api_state, monkeypatch, fault):
    from backend.utils.state import current_epoch, run_locked
    import backend.utils.state as state_helpers

    api = api_state
    job, staged, job_dir = _staged_attempt(api)
    graph_bytes = (api.processed / "graph.json").read_bytes()
    index_bytes = (api.processed / "faiss.index").read_bytes()
    if fault == "embedding":
        monkeypatch.setattr(api.main.VectorIndex, "build", Mock(side_effect=RuntimeError("embedding failed")))
    elif fault == "graph":
        monkeypatch.setattr(api.main, "_run_graph_pipeline", Mock(side_effect=RuntimeError("graph failed")))
    elif fault == "store":
        monkeypatch.setattr(api.store, "replace_sources", Mock(side_effect=RuntimeError("database full")))
    else:
        replace = state_helpers.os.replace
        failed = False

        def fail_index_once(source, destination):
            nonlocal failed
            if not failed and Path(destination) == api.processed / "faiss.index":
                failed = True
                raise OSError("index publish failed")
            return replace(source, destination)

        monkeypatch.setattr(state_helpers.os, "replace", fail_index_once)
    with pytest.raises((RuntimeError, OSError)):
        run_locked(api.main._ingest_locked, job, staged, current_epoch(), job_dir)
    _assert_old_state(api, graph_bytes, index_bytes)


def test_successful_reingest_publishes_coherent_state(api_state):
    from backend.utils.state import current_epoch, run_locked

    api = api_state
    job, staged, job_dir = _staged_attempt(api)
    run_locked(api.main._ingest_locked, job, staged, current_epoch(), job_dir)
    assert job.status == "done"
    assert job.result.ingested_source_ids == [api.sid]
    assert api.source.read_bytes() == b"new raw"
    assert api.store.get_chunk(api.old.id).text == "new material"
    assert api.main.state.vector_index.ids == [api.old.id]
    assert [attrs["name"] for _, attrs in api.main.state.graph_builder.graph.nodes(data=True)] == ["New concept"]
    assert not list(api.source.parent.glob("*.bak"))
    image = Path(api.store.get_chunk(api.old.id).metadata["image_paths"][0])
    assert image.read_bytes() == b"new derived image"
    assert image != Path(api.old.metadata["image_paths"][0])
    assert Path(api.old.metadata["image_paths"][0]).read_bytes() == b"previous derived image"


def test_actual_api_rejects_unauthenticated_clear_without_changes(api_state):
    api_state.client.headers.pop("authorization")
    assert api_state.client.delete("/api/data").status_code == 401
    assert api_state.store.count() == 1
    assert api_state.source.read_bytes() == b"old raw"


def test_actual_session_cookie_opens_source_and_protects_mutations(api_state):
    client = api_state.client
    client.headers.pop("authorization")
    response = client.post("/api/auth/session", json={"token": "fixture-token"}, headers={"Origin": "http://localhost"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    source = client.get(f"/api/source/{api_state.sid}")
    assert source.status_code == 200
    assert source.content == b"old raw"
    assert source.headers["x-content-type-options"] == "nosniff"
    assert client.delete("/api/data", headers={"Origin": "https://evil.example"}).status_code == 403
    assert api_state.store.count() == 1


def test_actual_api_rejects_executable_svg_before_staging(api_state):
    response = api_state.client.post("/api/ingest", files={"files": ("payload.svg", b"<svg><script>1</script></svg>", "image/svg+xml")})
    assert response.status_code == 415
    assert list(api_state.staging.iterdir()) == []
    assert api_state.store.count() == 1


def test_actual_s3_rejects_unissued_key_before_network(api_state, monkeypatch):
    monkeypatch.setenv("S3_UPLOAD_BUCKET", "private-fixture-uploads")
    response = api_state.client.post("/api/ingest/from-s3", json={"keys": ["private/admin.pdf"]})
    assert response.status_code == 404
    api_state.main._get_s3.assert_not_called()


def test_missing_optional_s3_sdk_has_actionable_error(monkeypatch):
    import sys
    from backend import main
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setattr(main, "_s3_client", None)
    with pytest.raises(UploadRejected) as error:
        main._get_s3()
    assert error.value.status == 503
    assert "optional boto3" in error.value.detail


@pytest.mark.parametrize("operation", ["presign", "ingest"])
def test_s3_optional_sdk_failure_returns_503_and_preserves_retry(api_state, monkeypatch, operation):
    monkeypatch.setenv("S3_UPLOAD_BUCKET", "private-fixture-uploads")
    monkeypatch.setattr(api_state.main, "_get_s3", Mock(side_effect=UploadRejected("Optional S3 SDK is missing.", status=503)))
    if operation == "presign":
        response = api_state.client.post("/api/upload/presign", json=[
            {"filename": "notes.pdf", "content_type": "application/pdf", "size": 4},
        ])
    else:
        record = api_state.main._issued_uploads.issue("notes.pdf", 4)
        response = api_state.client.post("/api/ingest/from-s3", json={"keys": [record.key]})
        assert api_state.main._issued_uploads.claim([record.key]) == [record]
        assert list(api_state.staging.iterdir()) == []
    assert response.status_code == 503
    assert "SDK" in response.json()["detail"]


def test_actual_clear_removes_sidecar_and_rebinds_worker(api_state):
    api = api_state
    worker = Mock()
    api.main.state.illustration_worker = worker
    assert api.client.delete("/api/data").status_code == 200
    assert api.store.count() == 0
    assert not (api.processed / "faiss.index").exists()
    assert not (api.processed / "faiss.json").exists()
    assert not api.source.exists()
    worker.reset.assert_called_once_with(api.main.state.graph_builder, api.store)


def test_queued_ingest_after_clear_cannot_restore_data(api_state):
    from backend.utils.state import current_epoch, run_locked

    api = api_state
    job, staged, job_dir = _staged_attempt(api)
    admitted = current_epoch()
    api.client.delete("/api/data")
    run_locked(api.main._ingest_locked, job, staged, admitted, job_dir)
    assert job.status == "cancelled"
    assert api.store.count() == 0
    assert not api.source.exists()


def test_slide_selection_and_identity_survive_sanitizer_collisions(api_state):
    items = api_state.main._validate_batch([("a+b.pdf", "application/pdf"), ("a_b.pdf", "application/pdf")], {"a_b.pdf"})
    assert items[0].source_id != items[1].source_id
    assert [item.is_slides for item in items] == [False, True]


def test_multipart_json_slide_names_preserve_commas(api_state, monkeypatch):
    admit = Mock(return_value=api_state.main.IngestJobResponse(job_id="fixture", status="queued"))
    monkeypatch.setattr(api_state.main, "_admit_ingest_job", admit)
    response = api_state.client.post(
        "/api/ingest", files={"files": ("lecture,1.pdf", b"%PDF", "application/pdf")},
        data={"slides": json.dumps(["lecture,1.pdf"])},
    )
    assert response.status_code == 202
    staged = admit.call_args.args[1][0]
    assert staged.is_slides is True
    assert staged.original_filename == "lecture,1.pdf"
    assert staged.path.stem != staged.source_id  # each attempt owns its processor assets


def test_s3_registry_preserves_original_identity(api_state, monkeypatch):
    monkeypatch.setenv("S3_UPLOAD_BUCKET", "private-fixture-uploads")
    s3 = Mock()
    s3.generate_presigned_url.return_value = "https://fixture.invalid/upload"
    monkeypatch.setattr(api_state.main, "_get_s3", Mock(return_value=s3))
    originals = ["a+b.pdf", "a?b.pdf"]
    response = api_state.client.post("/api/upload/presign", json=[
        {"filename": name, "content_type": "application/pdf", "size": 4} for name in originals
    ])
    assert response.status_code == 200
    records = api_state.main._issued_uploads.claim([item["key"] for item in response.json()])
    assert [record.original_filename for record in records] == originals
    checked = api_state.main._validate_batch([(record.original_filename, None) for record in records], set())
    assert checked[0].source_id != checked[1].source_id


def test_reingest_migrates_matching_legacy_source(api_state):
    from backend.utils.state import current_epoch, run_locked

    api = api_state
    (api.raw / "notes.pdf").write_bytes(b"legacy raw")
    api.store.save_chunks([Chunk(id="notes_chunk_0", source_id="notes", modality="pdf", text="legacy obsolete")])
    job, staged, job_dir = _staged_attempt(api)
    run_locked(api.main._ingest_locked, job, staged, current_epoch(), job_dir)
    assert api.store.count() == 1
    assert api.store.get_chunks_by_source("notes") == []
    assert api.client.get("/api/source/notes").status_code == 404


def test_reingest_preserves_legacy_source_of_another_modality(api_state):
    from backend.utils.state import current_epoch, run_locked

    api = api_state
    (api.raw / "notes.pdf").write_bytes(b"legacy pdf")
    api.store.save_chunks([Chunk(id="notes_chunk_0", source_id="notes", modality="audio", text="audio material")])
    job, staged, job_dir = _staged_attempt(api)
    run_locked(api.main._ingest_locked, job, staged, current_epoch(), job_dir)
    assert api.store.get_chunk("notes_chunk_0").text == "audio material"
    assert api.store.count() == 2


def test_concurrent_ingests_publish_an_index_containing_both_sources(api_state, monkeypatch):
    from backend.utils.state import current_epoch

    api = api_state
    first_started, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original_parse = api.main._ingest_single_file
    first_id = source_identity("first.pdf")
    second_id = source_identity("second.pdf")

    def parse(path, config, slides):
        if path.stem.startswith(first_id):
            first_started.set()
            assert release.wait(5), "test did not release first parse"
        else:
            second_started.set()
        return original_parse(path, config, slides)

    monkeypatch.setattr(api.main, "_ingest_single_file", parse)

    def prepare(name, sid):
        directory = api.staging / name
        path = directory / "00" / f"{sid}_{name}.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture pdf")
        return api.main._IngestJob(name), [uploads.StagedFile(path, name + ".pdf", name + ".pdf", sid)], directory

    first, second = prepare("first", first_id), prepare("second", second_id)

    async def run():
        epoch = current_epoch()
        a = asyncio.create_task(api.main._run_ingest_job(*first, epoch))
        assert await asyncio.to_thread(first_started.wait, 5)
        b = asyncio.create_task(api.main._run_ingest_job(*second, epoch))
        try:
            await asyncio.sleep(0.02)
            assert not second_started.is_set()
        finally:
            release.set()
        await asyncio.gather(a, b)

    asyncio.run(run())
    assert first[0].status == second[0].status == "done"
    assert set(api.main.state.vector_index.ids) == {api.old.id, f"{first_id}_chunk_0", f"{second_id}_chunk_0"}


def test_https_frontend_receives_secure_cookie_behind_http_proxy(api_state):
    from backend.utils.auth import AuthMiddleware

    app = FastAPI()
    app.post("/api/auth/session")(api_state.main.create_session)
    app.add_middleware(AuthMiddleware, allowed_origins=["https://app.example"])
    with TestClient(app, base_url="http://backend") as client:
        response = client.post("/api/auth/session", json={"token": "fixture-token"}, headers={"Origin": "https://app.example"})
    assert response.status_code == 200
    assert "; secure" in response.headers["set-cookie"].lower()


def test_unresolved_recovery_closes_data_routes(api_state, monkeypatch):
    from backend.utils.recovery import RecoveryError

    api = api_state
    monkeypatch.setattr(api.main.state, "storage_error", None)
    manager = Mock()
    manager.publish.side_effect = RecoveryError("rollback could not finish")
    manager.recover.side_effect = RecoveryError("required backup missing")
    with pytest.raises(RecoveryError):
        api.main._publish_storage(manager, [], Mock())
    assert api.main.state.storage_error is not None
    for route in ("/api/graph", f"/api/source/{api.sid}", "/static/illustrations/a.png"):
        response = api.client.get(route)
        assert response.status_code == 503
        assert "Storage recovery" in response.json()["detail"]
    assert api.client.get("/api/health").status_code == 200
    assert api.client.get("/api/auth/status").status_code == 200


def test_startup_recovers_before_derived_loads_and_staging_cleanup(api_state, monkeypatch):
    api = api_state
    events = []
    orphan = api.staging / "orphan"
    orphan.mkdir()
    (orphan / "candidate").write_bytes(b"recovery input")

    def recover():
        assert (orphan / "candidate").exists()
        events.append("recover")
        return 0

    manager = Mock()
    manager.recover.side_effect = recover
    monkeypatch.setattr(api.main, "_recovery_manager", lambda: manager)
    monkeypatch.setattr(api.main, "_load_config", lambda: api.main.state.config)
    monkeypatch.setattr(api.main, "DocumentStore", lambda _path: api.store)

    def embedder(_name):
        assert events == ["recover"]
        events.append("embedder")
        return object()

    class StartupIndex:
        def __init__(self, _embedder):
            pass

        def load(self, _path):
            assert "recover" in events
            events.append("index")

    graph_type = api.main.GraphBuilder

    def graph(path):
        assert "recover" in events
        events.append("graph")
        return graph_type(path)

    remove = api.main.remove_staging

    def remove_after_recovery(path, root):
        assert "recover" in events
        events.append("staging_cleanup")
        return remove(path, root)

    class Scheduler:
        async def scan_and_enqueue_pending(self):
            return 0

        async def stop(self):
            pass

    monkeypatch.setattr(api.main, "Embedder", embedder)
    monkeypatch.setattr(api.main, "VectorIndex", StartupIndex)
    monkeypatch.setattr(api.main, "GraphBuilder", graph)
    monkeypatch.setattr(api.main, "_warm_ollama", Mock())
    monkeypatch.setattr(api.main, "_start_illustration_scheduler", lambda: Scheduler())
    monkeypatch.setattr(api.main, "remove_staging", remove_after_recovery)

    async def start():
        async with api.main.lifespan(api.main.app):
            assert events[0] == "recover"
            assert events.index("recover") < events.index("index") < events.index("staging_cleanup")
            assert not orphan.exists()
            assert Path(api.old.metadata["image_paths"][0]).read_bytes() == b"previous derived image"

    asyncio.run(start())


def test_committed_asset_paths_survive_staging_cleanup_and_store_reopen(api_state):
    from backend.db.document_store import DocumentStore
    from backend.utils.state import current_epoch

    api = api_state
    job, staged, job_dir = _staged_attempt(api)
    asyncio.run(api.main._run_ingest_job(job, staged, job_dir, current_epoch()))
    assert job.status == "done"
    assert not job_dir.exists()
    with DocumentStore(api.store.path) as reopened:
        image = Path(reopened.get_chunk(api.old.id).metadata["image_paths"][0])
        assert api.staging not in image.parents
        assert image.read_bytes() == b"new derived image"
