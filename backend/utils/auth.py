"""
API authentication + CSRF origin checks.

Modes
-----
  Token mode   (LUMINRAG_API_TOKEN set): a request is authenticated by
               ``Authorization: Bearer <token>`` (constant-time compare) or by a
               signed HttpOnly session cookie issued by POST /api/auth/session.
  Local mode   (no token): only requests whose actual socket peer
               (``scope["client"]``) is a loopback address are allowed.
               Forwarded headers are ignored; anything else fails closed.

CSRF
----
Cookie- and local-mode requests with a state-changing method must not carry
an unknown ``Origin``: it has to be an explicitly allowed CORS origin.
Requests with no Origin (curl / CLI) are accepted. Bearer-auth
requests skip this check because browsers never attach the header implicitly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.utils.environment import hosted_mode

TOKEN_ENV = "LUMINRAG_API_TOKEN"
CORS_ORIGINS_ENV = "LUMINRAG_ALLOWED_ORIGINS"
SESSION_COOKIE = "luminrag_session"
SESSION_TTL_SECONDS = 8 * 60 * 60

DEFAULT_CORS_ORIGINS = (
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

PROTECTED_PREFIXES = ("/api", "/static/illustrations")
# (method, path) pairs reachable without authentication
PUBLIC_ROUTES = frozenset({
    ("GET", "/api/health"),
    ("GET", "/api/ready"),
    ("GET", "/api/auth/status"),
    ("POST", "/api/auth/session"),
    ("DELETE", "/api/auth/session"),
})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_api_token() -> str | None:
    """Configured API token, read at call time so rotation/tests take effect."""
    token = os.getenv(TOKEN_ENV, "").strip()
    return token or None


def _normalise_origin(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def load_allowed_origins() -> list[str]:
    """CORS origins from LUMINRAG_ALLOWED_ORIGINS (comma-separated) or defaults."""
    raw = os.getenv("ALLOWED_ORIGINS", os.getenv(CORS_ORIGINS_ENV, ""))
    origins = [o for o in (_normalise_origin(x) for x in raw.split(",")) if o]
    if not origins:
        origins = [_normalise_origin(o) for o in DEFAULT_CORS_ORIGINS]
    # "*" would combine with allow_credentials into a CSRF hole — never allow it.
    return [o for o in dict.fromkeys(origins) if o != "*"]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def tokens_match(supplied: str | None, expected: str | None) -> bool:
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _session_key(token: str) -> bytes:
    # Derived from the token so rotating the token invalidates every session.
    return hmac.new(token.encode("utf-8"), b"luminrag-session-v1", hashlib.sha256).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_session_value(token: str, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + SESSION_TTL_SECONDS)
    payload = f"{expires}.{secrets.token_urlsafe(16)}"
    sig = hmac.new(_session_key(token), payload.encode("ascii"), hashlib.sha256).digest()
    return f"{payload}.{_b64(sig)}"


def verify_session_value(value: str | None, token: str | None, now: float | None = None) -> bool:
    if not value or not token:
        return False
    parts = value.split(".")
    if len(parts) != 3:
        return False
    expires_raw, nonce, sig = parts
    payload = f"{expires_raw}.{nonce}"
    try:
        expected = _b64(
            hmac.new(_session_key(token), payload.encode("ascii"), hashlib.sha256).digest()
        )
    except UnicodeEncodeError:
        return False
    if not hmac.compare_digest(sig.encode("ascii", "replace"), expected.encode("ascii")):
        return False
    try:
        expires = int(expires_raw)
    except ValueError:
        return False
    return expires > (now if now is not None else time.time())


def is_loopback_client(request: Request) -> bool:
    """True only when the real socket peer is loopback. Forwarded headers are ignored."""
    client = request.client
    if client is None or not client.host:
        return False
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        return False


def is_loopback_host(request: Request) -> bool:
    """Reject DNS rebinding even when the connection reaches a loopback socket."""
    try:
        parsed = urlsplit("//" + request.headers.get("host", ""))
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            return False
        host = parsed.hostname
        if host == "localhost":
            return True
        return bool(host) and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def origin_allowed(request: Request, allowed_origins: Iterable[str]) -> bool:
    """
    False when the request carries an Origin outside the explicit allowlist.
    Missing Origin (CLI) is accepted
    unless the browser explicitly flags the request as cross-site.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return request.headers.get("sec-fetch-site", "").lower() != "cross-site"
    norm = _normalise_origin(origin)
    if not norm or norm == "null":
        return False
    return norm in set(allowed_origins)


# ---------------------------------------------------------------------------
# Request evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthResult:
    authenticated: bool
    method: str | None = None  # "bearer" | "cookie" | "local"
    status: int = 401
    detail: str = "Authentication required."


def evaluate_request(request: Request) -> AuthResult:
    # Only the outer workspace middleware can set this ASGI state marker.
    # It validates the capability and denies legacy file/S3/session routes
    # before handing a hosted request to this authentication layer.
    if hosted_mode() and getattr(request.state, "hosted_workspace_authorized", False):
        return AuthResult(True, "workspace")
    token = get_api_token()
    if token:
        bearer = _bearer_token(request)
        if bearer is not None:
            if tokens_match(bearer, token):
                return AuthResult(True, "bearer")
            return AuthResult(False, status=401, detail="Invalid API token.")
        if verify_session_value(request.cookies.get(SESSION_COOKIE), token):
            return AuthResult(True, "cookie")
        return AuthResult(False, status=401, detail="Authentication required.")

    if is_loopback_client(request) and is_loopback_host(request):
        return AuthResult(True, "local")
    return AuthResult(
        False,
        status=403,
        detail="No API token configured; only loopback clients are allowed.",
    )


def _canonical_path(path: str) -> str:
    while "//" in path:
        path = path.replace("//", "/")
    return path


def _is_protected(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES)


class AuthMiddleware:
    """
    Pure-ASGI middleware enforcing authentication + CSRF origin checks on
    /api and static illustration assets. Install it *inside* CORSMiddleware
    (add it first) so error responses still carry CORS headers.
    """

    def __init__(self, app, allowed_origins: Iterable[str]) -> None:
        self.app = app
        self.allowed_origins = [_normalise_origin(o) for o in allowed_origins]

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = _canonical_path(scope.get("path", ""))
        if not _is_protected(path):
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        method = request.method.upper()

        if (method, path) in PUBLIC_ROUTES:
            # Session login/logout set cookies, so they still need the origin check.
            if path == "/api/auth/session" and not origin_allowed(request, self.allowed_origins):
                await _deny(scope, receive, send, 403, "Origin not allowed.")
                return
            await self.app(scope, receive, send)
            return

        result = evaluate_request(request)
        if not result.authenticated:
            await _deny(scope, receive, send, result.status, result.detail)
            return

        if (
            result.method != "bearer"
            and method not in SAFE_METHODS
            and not origin_allowed(request, self.allowed_origins)
        ):
            await _deny(scope, receive, send, 403, "Origin not allowed.")
            return

        await self.app(scope, receive, send)


async def _deny(scope, receive, send, status: int, detail: str) -> None:
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    response = JSONResponse({"detail": detail}, status_code=status, headers=headers)
    await response(scope, receive, send)
