"""Public-service boundaries without model downloads or external requests."""

import asyncio
import time
import uuid
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk
from backend.utils.auth import AuthMiddleware, load_allowed_origins
from backend.utils.hosted import HostedSecurityMiddleware, JSON_BODY_LIMIT


def client(monkeypatch, hosted=True):
    monkeypatch.setenv("LUMIN_HOSTED", "1" if hosted else "0")
    app = FastAPI()

    @app.post("/api/query")
    async def query(request: Request):
        return {"received": len(await request.body())}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    app.add_middleware(HostedSecurityMiddleware)
    return TestClient(app)


def test_provider_requests_rate_limited_without_trusting_forwarded_ip(monkeypatch):
    browser = client(monkeypatch)
    for index in range(20):
        assert browser.post("/api/query", content=b"{}", headers={"X-Forwarded-For": f"10.0.0.{index}"}).status_code == 200
    response = browser.post("/api/query", content=b"{}", headers={"X-Forwarded-For": "192.0.2.99"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert browser.get("/api/health").status_code == 200


async def consume_request(scope, receive, send):
    await Request(scope, receive).body()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def request_scope(path, identifier=None):
    return {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"", "server": ("local", 80),
            "client": ("192.0.2.10", 1234), "headers": [(b"content-length", b"100"),
            (b"x-workspace-id", (identifier or str(uuid.uuid4())).encode())]}


@pytest.mark.parametrize("path,count", [("/api/query", 20), ("/api/ingest", 5)])
def test_rotating_workspace_ids_does_not_reset_peer_category_limits(monkeypatch, path, count):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        app = HostedSecurityMiddleware(consume_request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as browser:
            for _ in range(count):
                assert (await browser.post(path, content=b"{}", headers={"X-Workspace-Id": str(uuid.uuid4())})).status_code == 200
            response = await browser.post(path, content=b"{}", headers={"X-Workspace-Id": str(uuid.uuid4())})
            assert response.status_code == 429
            assert response.headers["retry-after"] == "60"
    asyncio.run(scenario())


def test_new_workspace_limit_preserves_rehydration_of_an_existing_capability(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        app = HostedSecurityMiddleware(consume_request)
        identifiers = [str(uuid.uuid4()) for _ in range(4)]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as browser:
            for identifier in identifiers[:3]:
                assert (await browser.post("/api/workspace/import", content=b"{}", headers={"X-Workspace-Id": identifier})).status_code == 200
            assert (await browser.post("/api/workspace/import", content=b"{}", headers={"X-Workspace-Id": identifiers[3]})).status_code == 429
            # The failed fresh-ID attempt does not consume the creation bucket for an already imported ID.
            assert (await browser.post("/api/workspace/import", content=b"{}", headers={"X-Workspace-Id": identifiers[0]})).status_code == 200
    asyncio.run(scenario())


def test_rate_limits_run_before_a_busy_admission_gate(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        app = HostedSecurityMiddleware(consume_request)
        assert app.import_slots.acquire(blocking=False)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as browser:
                for _ in range(3):
                    response = await browser.post("/api/workspace/import", content=b"{}", headers={"X-Workspace-Id": str(uuid.uuid4())})
                    assert response.status_code == 429
                    assert response.headers["retry-after"] == "2"
                response = await browser.post("/api/workspace/import", content=b"{}", headers={"X-Workspace-Id": str(uuid.uuid4())})
                assert response.status_code == 429
                assert response.headers["retry-after"] == "60"
        finally:
            app.import_slots.release()
    asyncio.run(scenario())


@pytest.mark.parametrize("path,gate_name,capacity", [("/api/workspace/import", "import_slots", 1), ("/api/ingest", "upload_slots", 2)])
def test_idle_body_timeout_returns_408_and_releases_upload_admission(monkeypatch, path, gate_name, capacity):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        app = HostedSecurityMiddleware(consume_request, body_idle_seconds=0.02, body_total_seconds=0.2)
        calls = 0
        messages = []

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b"x", "more_body": True}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await asyncio.wait_for(app(request_scope(path), receive, send), 0.5)
        assert next(message["status"] for message in messages if message["type"] == "http.response.start") == 408
        gate = getattr(app, gate_name)
        for _ in range(capacity):
            assert gate.acquire(blocking=False)
        assert not gate.acquire(blocking=False)
        for _ in range(capacity):
            gate.release()
        # The next actual body is accepted after the stalled request is gone.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as browser:
            assert (await browser.post(path, content=b"{}", headers={"X-Workspace-Id": str(uuid.uuid4())})).status_code == 200
    asyncio.run(scenario())


def test_total_body_deadline_stops_a_continually_progressing_trickle(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        app = HostedSecurityMiddleware(consume_request, body_idle_seconds=0.2, body_total_seconds=0.04)
        messages, chunks = [], []

        async def receive():
            await asyncio.sleep(0.005)
            chunks.append(b"x")
            return {"type": "http.request", "body": b"x", "more_body": True}

        async def send(message):
            messages.append(message)

        started = time.monotonic()
        await asyncio.wait_for(app(request_scope("/api/workspace/import"), receive, send), 0.5)
        assert time.monotonic() - started < 0.4
        assert len(chunks) >= 2
        assert next(message["status"] for message in messages if message["type"] == "http.response.start") == 408
        assert app.import_slots.acquire(blocking=False)
        app.import_slots.release()
    asyncio.run(scenario())


def test_body_deadline_does_not_cancel_processing_after_the_body_is_complete(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        async def processing(scope, receive, send):
            await Request(scope, receive).body()
            await asyncio.sleep(0.04)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        app = HostedSecurityMiddleware(processing, body_idle_seconds=0.01, body_total_seconds=0.02)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as browser:
            assert (await browser.post("/api/workspace/import", content=b"{}", headers={"X-Workspace-Id": str(uuid.uuid4())})).status_code == 200
    asyncio.run(scenario())


def test_cancelled_body_read_always_releases_admission(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        entered = asyncio.Event()
        app = HostedSecurityMiddleware(consume_request)

        async def receive():
            entered.set()
            await asyncio.Event().wait()

        async def send(_message):
            pass

        pending = asyncio.create_task(app(request_scope("/api/workspace/import"), receive, send))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert app.import_slots.acquire(blocking=False)
        app.import_slots.release()
    asyncio.run(scenario())


def test_large_json_rejected_before_endpoint(monkeypatch):
    browser = client(monkeypatch)
    assert browser.post("/api/query", content=b"x" * (JSON_BODY_LIMIT + 1)).status_code == 413


def test_streamed_json_size_limit_without_content_length(monkeypatch):
    browser = client(monkeypatch)
    response = browser.post("/api/query", content=iter([b"x" * JSON_BODY_LIMIT, b"x"]))
    assert response.status_code == 413


def test_responses_cannot_be_cached_or_mime_sniffed(monkeypatch):
    response = client(monkeypatch).post("/api/query", content=b"{}")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_hosted_environment_alone_never_disables_authentication(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    monkeypatch.setenv("LUMINRAG_API_TOKEN", "local-administrator-token")
    app = FastAPI()

    @app.get("/api/private")
    def private():
        return {"secret": "protected"}

    app.add_middleware(AuthMiddleware, allowed_origins=["https://lumin.example"])
    # A client cannot supply the ASGI-only workspace authorization marker.
    response = TestClient(app).get("/api/private", headers={"X-Workspace-Id": "a" * 36, "hosted_workspace_authorized": "true"})
    assert response.status_code == 401


def test_hosted_origins_accept_explicit_list_and_discard_wildcard(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://lumin.example,*,https://lumin.example/")
    assert load_allowed_origins() == ["https://lumin.example"]


def test_import_admission_rejects_before_reading_another_body(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")

    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def upload(scope, receive, send):
            entered.set()
            await finish.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        app = HostedSecurityMiddleware(upload)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as browser:
            first = asyncio.create_task(browser.post("/api/workspace/import", content=b"first"))
            await entered.wait()
            second = await browser.post("/api/workspace/import", content=b"second")
            assert second.status_code == 429
            finish.set()
            assert (await first).status_code == 200
            # Completion releases admission for the next import.
            assert (await browser.post("/api/workspace/import", content=b"third")).status_code == 200

    asyncio.run(scenario())


def test_sql_metacharacters_remain_data_in_reads_and_deletes(tmp_path):
    store = DocumentStore(tmp_path / "chunks.db")
    attack = "'; DROP TABLE chunks; --"
    try:
        store.save_chunks([
            Chunk(id=attack, source_id=attack, modality="markdown", text="Untrusted text"),
            Chunk(id="safe", source_id="safe", modality="markdown", text="Keep me"),
        ])
        assert store.get_chunk(attack).text == "Untrusted text"
        assert [item.id for item in store.get_chunks_by_ids([attack])] == [attack]
        assert store.get_chunks_by_source("' OR 1=1 --") == []
        assert store.delete_source("' OR 1=1 --") == 0
        assert store.delete_source(attack) == 1
        assert store.count() == 1
        assert store.get_chunk("safe").text == "Keep me"
    finally:
        store.close()
