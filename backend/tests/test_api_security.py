"""
Tests for backend/utils/auth.py — authentication + CSRF origin middleware.

Uses a minimal FastAPI app (no models, no data dirs) wrapped so the ASGI
client address can be set explicitly; backend.main is not imported.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from backend.utils import auth
from backend.utils.auth import (
    SESSION_COOKIE,
    AuthMiddleware,
    load_allowed_origins,
    make_session_value,
    verify_session_value,
)

ALLOWED = "http://localhost:5173"
TOKEN = "s3cret-token-value"


class _ClientAddress:
    """ASGI wrapper that pins scope['client'] to a chosen peer address."""

    def __init__(self, app, host: str) -> None:
        self.app = app
        self.host = host

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope, client=(self.host, 50000))
        await self.app(scope, receive, send)


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/auth/status")
    def status(request: Request):
        return {
            "authenticated": auth.evaluate_request(request).authenticated,
            "token_required": auth.get_api_token() is not None,
        }

    @app.post("/api/auth/session")
    def login(response: Response):
        response.set_cookie(SESSION_COOKIE, "x", httponly=True, samesite="strict")
        return {"ok": True}

    @app.get("/api/data")
    def read():
        return {"ok": True}

    @app.post("/api/data")
    def write():
        return {"ok": True}

    @app.get("/static/illustrations/a.png")
    def illus():
        return {"ok": True}

    @app.get("/public")
    def public():
        return {"ok": True}

    app.add_middleware(AuthMiddleware, allowed_origins=[ALLOWED])
    return app


def _client(host: str = "127.0.0.1") -> TestClient:
    return TestClient(_ClientAddress(_build_app(), host), base_url="http://localhost")


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv(auth.TOKEN_ENV, raising=False)
    monkeypatch.delenv(auth.CORS_ORIGINS_ENV, raising=False)


# ---------------------------------------------------------------------------
# Local (no token) mode
# ---------------------------------------------------------------------------

class TestLocalMode:
    def test_loopback_allowed(self):
        assert _client("127.0.0.1").get("/api/data").status_code == 200
        assert _client("::1").get("/api/data").status_code == 200

    def test_remote_rejected(self):
        assert _client("10.0.0.5").get("/api/data").status_code == 403

    def test_forwarded_headers_ignored(self):
        r = _client("10.0.0.5").get(
            "/api/data",
            headers={"X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"},
        )
        assert r.status_code == 403

    def test_dns_rebinding_host_rejected(self):
        response = _client().get("/api/data", headers={"Host": "evil.example"})
        assert response.status_code == 403

    def test_matching_host_origin_does_not_bypass_local_restriction(self):
        response = _client().post(
            "/api/data", headers={"Host": "evil.example", "Origin": "http://evil.example"}
        )
        assert response.status_code == 403

    def test_non_ip_client_fails_closed(self):
        assert _client("testclient").get("/api/data").status_code == 403

    def test_static_illustrations_protected(self):
        assert _client("10.0.0.5").get("/static/illustrations/a.png").status_code == 403

    def test_unprotected_path_untouched(self):
        assert _client("10.0.0.5").get("/public").status_code == 200

    def test_health_public_and_minimal(self):
        r = _client("10.0.0.5").get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_auth_status_public(self):
        r = _client("10.0.0.5").get("/api/auth/status")
        assert r.json() == {"authenticated": False, "token_required": False}
        r = _client("127.0.0.1").get("/api/auth/status")
        assert r.json() == {"authenticated": True, "token_required": False}


class TestCsrf:
    def test_unknown_origin_rejected_on_post(self):
        r = _client().post("/api/data", headers={"Origin": "https://evil.example"})
        assert r.status_code == 403

    def test_null_origin_rejected(self):
        assert _client().post("/api/data", headers={"Origin": "null"}).status_code == 403

    def test_allowed_origin_accepted(self):
        assert _client().post("/api/data", headers={"Origin": ALLOWED}).status_code == 200

    def test_host_does_not_expand_origin_allowlist(self):
        r = _client().post("/api/data", headers={"Origin": "http://localhost"})
        assert r.status_code == 403

    def test_no_origin_cli_accepted(self):
        assert _client().post("/api/data").status_code == 200

    def test_cross_site_fetch_without_origin_rejected(self):
        r = _client().post("/api/data", headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403

    def test_unknown_origin_get_allowed(self):
        r = _client().get("/api/data", headers={"Origin": "https://evil.example"})
        assert r.status_code == 200

    def test_session_login_checks_origin(self):
        r = _client().post("/api/auth/session", headers={"Origin": "https://evil.example"})
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Token mode
# ---------------------------------------------------------------------------

class TestTokenMode:
    @pytest.fixture(autouse=True)
    def _token(self, monkeypatch):
        monkeypatch.setenv(auth.TOKEN_ENV, TOKEN)

    def test_missing_credentials_401_even_on_loopback(self):
        assert _client("127.0.0.1").get("/api/data").status_code == 401

    def test_bearer_accepted_from_remote(self):
        r = _client("10.0.0.5").get("/api/data", headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 200

    def test_wrong_bearer_rejected(self):
        r = _client().get("/api/data", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_bearer_skips_origin_check(self):
        r = _client().post(
            "/api/data",
            headers={"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example"},
        )
        assert r.status_code == 200

    def test_session_cookie_accepted(self):
        cookie = make_session_value(TOKEN)
        r = _client("10.0.0.5").get("/api/data", headers={"Cookie": f"{SESSION_COOKIE}={cookie}"})
        assert r.status_code == 200

    def test_cookie_post_requires_known_origin(self):
        cookie = make_session_value(TOKEN)
        headers = {"Cookie": f"{SESSION_COOKIE}={cookie}", "Origin": "https://evil.example"}
        assert _client().post("/api/data", headers=headers).status_code == 403
        headers["Origin"] = ALLOWED
        assert _client().post("/api/data", headers=headers).status_code == 200

    def test_cookie_cannot_trust_attacker_supplied_host(self):
        cookie = make_session_value(TOKEN)
        headers = {
            "Cookie": f"{SESSION_COOKIE}={cookie}",
            "Host": "evil.example",
            "Origin": "http://evil.example",
        }
        assert _client().post("/api/data", headers=headers).status_code == 403

    def test_tampered_cookie_rejected(self):
        cookie = make_session_value(TOKEN)
        expires, nonce, sig = cookie.split(".")
        forged = f"{int(expires) + 9999}.{nonce}.{sig}"
        r = _client().get("/api/data", headers={"Cookie": f"{SESSION_COOKIE}={forged}"})
        assert r.status_code == 401

    def test_status_reports_token_required(self):
        r = _client().get("/api/auth/status")
        assert r.json() == {"authenticated": False, "token_required": True}
        r = _client().get("/api/auth/status", headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.json() == {"authenticated": True, "token_required": True}


class TestSessionValues:
    def test_expiry(self):
        value = make_session_value(TOKEN, now=1_000)
        assert verify_session_value(value, TOKEN, now=1_001)
        assert not verify_session_value(value, TOKEN, now=1_000 + auth.SESSION_TTL_SECONDS + 1)

    def test_token_rotation_invalidates(self):
        value = make_session_value(TOKEN)
        assert not verify_session_value(value, "another-token")

    @pytest.mark.parametrize("bad", ["", "a.b", "x.y.z", "1.2.3.4", "é.é.é"])
    def test_malformed(self, bad):
        assert not verify_session_value(bad, TOKEN)


class TestOrigins:
    def test_env_origins_normalised_and_wildcard_dropped(self, monkeypatch):
        monkeypatch.setenv(auth.CORS_ORIGINS_ENV, "https://App.example/ , *, https://app.example")
        assert load_allowed_origins() == ["https://app.example"]

    def test_defaults(self):
        origins = load_allowed_origins()
        assert "http://localhost:5173" in origins
        assert all(not o.endswith("/") for o in origins)
