"""The batch client must preserve data and never retry uncertain uploads."""

import json

import httpx
import pytest

from scripts import ingest


def test_dry_run_leaves_existing_data_and_makes_no_requests(tmp_path):
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"%PDF existing")
    database = tmp_path / "chunks.db"
    database.write_bytes(b"existing database")

    def unexpected(request):
        pytest.fail("dry run must not contact the server")

    assert ingest.main([str(tmp_path), "--dry-run"], transport=httpx.MockTransport(unexpected)) == 0
    assert source.read_bytes() == b"%PDF existing"
    assert database.read_bytes() == b"existing database"


def test_upload_keeps_original_identity_and_json_slide_names(tmp_path, monkeypatch):
    name = "lecture,+notes.pdf"
    (tmp_path / name).write_bytes(b"%PDF content")
    monkeypatch.setenv(ingest.TOKEN_ENV, "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        if request.method == "POST":
            body = request.read().decode()
            assert f'filename="{name}"' in body
            assert json.dumps([name]) in body
            assert request.url.path == "/api/ingest"
            return httpx.Response(202, json={"job_id": "test-job"})
        assert request.url.path == "/api/ingest/jobs/test-job"
        return httpx.Response(200, json={"status": "done", "result": {"files_processed": 1}})

    assert ingest.main([str(tmp_path), "--slides", name], transport=httpx.MockTransport(handler)) == 0
    assert [r.method for r in requests] == ["POST", "GET"]


def test_distinct_names_that_sanitize_identically_remain_distinct(tmp_path):
    for name in ("a+b.pdf", "a_b.pdf"):
        (tmp_path / name).write_bytes(b"%PDF content")
    plan = ingest.plan_directory(tmp_path, set())
    assert not plan.problems
    assert len({item.source_id for item in plan.files}) == 2


def test_duplicate_basenames_rejected_before_any_upload(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "lecture.pdf").write_bytes(b"%PDF content")

    def unexpected(request):
        pytest.fail("duplicates must be rejected before an upload")

    assert ingest.main([str(tmp_path)], transport=httpx.MockTransport(unexpected)) == 2


def test_junction_directory_is_not_scanned(tmp_path, monkeypatch):
    from pathlib import Path

    linked = tmp_path / "junction"
    linked.mkdir()
    (linked / "private.pdf").write_bytes(b"must not upload")
    (tmp_path / "lecture.pdf").write_bytes(b"selected")
    monkeypatch.setattr(Path, "is_junction", lambda path: path == linked, raising=False)
    assert [item.filename for item in ingest.plan_directory(tmp_path, set()).files] == ["lecture.pdf"]


def test_slide_choice_is_not_applied_to_sanitizer_collision(tmp_path):
    for name in ("a+b.pdf", "a_b.pdf"):
        (tmp_path / name).write_bytes(b"%PDF content")
    plan = ingest.plan_directory(tmp_path, {"a_b.pdf"})
    assert {item.filename for item in plan.files if item.is_slides} == {"a_b.pdf"}


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_job_keeps_local_source(tmp_path, status):
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"original")

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "job"})
        return httpx.Response(200, json={"status": status, "error": "processing failed"})

    assert ingest.main([str(tmp_path)], transport=httpx.MockTransport(handler)) == 1
    assert source.read_bytes() == b"original"


def test_uncertain_post_is_not_retried_and_stream_is_closed(tmp_path, monkeypatch):
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"original")
    streams = []
    requests = []

    def open_file(path):
        stream = path.open("rb")
        streams.append(stream)
        return stream

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("server stopped responding", request=request)

    monkeypatch.setattr(ingest, "_open_for_upload", open_file)
    assert ingest.main([str(tmp_path)], transport=httpx.MockTransport(handler)) == 1
    assert len(requests) == 1
    assert streams and all(stream.closed for stream in streams)


def test_redirect_does_not_forward_credentials(tmp_path, monkeypatch):
    (tmp_path / "lecture.pdf").write_bytes(b"original")
    monkeypatch.setenv(ingest.TOKEN_ENV, "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://untrusted.invalid/upload"})

    assert ingest.main([str(tmp_path)], transport=httpx.MockTransport(handler)) == 1
    assert len(requests) == 1
    assert requests[0].url.host == "127.0.0.1"


def test_polling_stops_at_deadline():
    elapsed = [0.0]
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "running"})

    def sleep(seconds):
        elapsed[0] += seconds

    with ingest.build_client(ingest.DEFAULT_API_URL, None, httpx.MockTransport(handler)) as client:
        with pytest.raises(ingest.CLIError, match="Gave up waiting"):
            ingest.poll_job(client, "job", interval=1, timeout=2, has_token=False,
                            sleep=sleep, clock=lambda: elapsed[0])
    assert len(requests) == 3


def test_missing_job_does_not_poll_forever():
    with ingest.build_client(ingest.DEFAULT_API_URL, None,
                             httpx.MockTransport(lambda request: httpx.Response(404))) as client:
        with pytest.raises(ingest.CLIError, match="unknown to the server"):
            ingest.poll_job(client, "job", interval=1, timeout=100, has_token=False)
