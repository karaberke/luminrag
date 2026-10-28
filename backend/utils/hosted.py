"""Shared HTTP limits for the public, single-process compute service."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import OrderedDict

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from backend.utils.environment import hosted_mode

JSON_BODY_LIMIT = 2 * 1024 * 1024
# Match the portable bundle bound plus multipart framing. A single admission
# slot bounds expanded import memory independently of concurrent visitors.
IMPORT_BODY_LIMIT = 257 * 1024 * 1024
PUBLIC_HEALTH_PATHS = frozenset({"/api/health", "/api/ready"})
BODY_IDLE_SECONDS = 15.0
BODY_TOTAL_SECONDS = 300.0
REMEMBERED_WORKSPACES = 4096
WORKSPACE_MEMORY_SECONDS = 3600


class HostedSecurityMiddleware:
    """Use SlowAPI's ``limits`` engine and bound bodies before parsing.

    Limits use the actual socket peer. Do not accept client-controlled
    forwarding headers unless the deployment explicitly trusts its proxy.
    """

    def __init__(self, app, *, body_idle_seconds=BODY_IDLE_SECONDS, body_total_seconds=BODY_TOTAL_SECONDS):
        self.app = app
        self.hosted = hosted_mode()
        self.body_idle_seconds = max(0.001, float(body_idle_seconds))
        self.body_total_seconds = max(0.001, float(body_total_seconds))
        self.import_slots = threading.BoundedSemaphore(1)
        self.upload_slots = threading.BoundedSemaphore(2)
        self.imported_workspaces = OrderedDict()
        self.workspace_lock = threading.Lock()
        self.limiter = None
        if self.hosted:
            from limits import parse
            from limits.storage import MemoryStorage
            from limits.strategies import MovingWindowRateLimiter

            self.limiter = MovingWindowRateLimiter(MemoryStorage())
            self.rates = {
                "all": parse("180/minute"),
                "compute": parse("20/minute"),
                "upload": parse("5/minute"),
                "new_workspace": parse("3/minute"),
            }

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        api = path == "/api" or path.startswith("/api/")
        response_status = None

        async def secure_send(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                if api:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        if not self.hosted or not api or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, secure_send)
            return
        if path not in PUBLIC_HEALTH_PATHS:
            peer = (scope.get("client") or ("unknown", 0))[0]
            workspace = dict(scope.get("headers", ())).get(b"x-workspace-id", b"").decode("ascii", "ignore")[:36]
            category = "upload" if path in {"/api/ingest", "/api/workspace/import"} else (
                "compute" if path in {"/api/query", "/api/graph/duplicates", "/api/graph/ops"} else "all"
            )
            # Rotating a capability cannot reset the socket peer's category budget.
            allowed = self.limiter.hit(self.rates["all"], peer, "all")
            if allowed and category != "all":
                allowed = self.limiter.hit(self.rates[category], peer, "peer", category)
                if allowed:
                    allowed = self.limiter.hit(self.rates[category], peer, workspace, "workspace", category)
            if allowed and path == "/api/workspace/import" and scope.get("method") == "POST" and not self._known_workspace(workspace):
                allowed = self.limiter.hit(self.rates["new_workspace"], peer, "new_workspace")
            if not allowed:
                await JSONResponse(
                    {"detail": "Too many requests. Please retry in a minute."},
                    status_code=429, headers={"Retry-After": "60"},
                )(scope, receive, secure_send)
                return

        if scope.get("method") in {"GET", "HEAD"}:
            await self.app(scope, receive, secure_send)
            return
        limit = IMPORT_BODY_LIMIT if path == "/api/workspace/import" else JSON_BODY_LIMIT
        if path == "/api/ingest":
            from backend.utils.uploads import MAX_BATCH_FILES, _MULTIPART_OVERHEAD_BYTES, max_upload_bytes

            limit = min(MAX_BATCH_FILES * max_upload_bytes(), 100 * 1024 * 1024) + _MULTIPART_OVERHEAD_BYTES
        headers = dict(scope.get("headers", ()))
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, secure_send)
            return
        if declared > limit:
            await JSONResponse({"detail": "Request body is too large."}, status_code=413)(scope, receive, secure_send)
            return

        gate = None
        if scope.get("method") == "POST":
            gate = self.import_slots if path == "/api/workspace/import" else (self.upload_slots if path == "/api/ingest" else None)
        if gate is not None and not gate.acquire(blocking=False):
            await JSONResponse(
                {"detail": "The server is preparing another upload. Please retry shortly."},
                status_code=429, headers={"Retry-After": "2"},
            )(scope, receive, secure_send)
            return
        try:
            await self._receive_body(scope, receive, secure_send, limit)
            if path == "/api/workspace/import" and response_status is not None and 200 <= response_status < 300:
                self._remember_workspace(workspace)
        finally:
            if gate is not None:
                gate.release()

    def _known_workspace(self, identifier):
        now = time.monotonic()
        with self.workspace_lock:
            while self.imported_workspaces and next(iter(self.imported_workspaces.values())) <= now - WORKSPACE_MEMORY_SECONDS:
                self.imported_workspaces.popitem(last=False)
            return identifier in self.imported_workspaces

    def _remember_workspace(self, identifier):
        try:
            parsed = uuid.UUID(identifier)
            if parsed.version != 4 or str(parsed) != identifier:
                return
        except (ValueError, AttributeError):
            return
        with self.workspace_lock:
            self.imported_workspaces.pop(identifier, None)
            self.imported_workspaces[identifier] = time.monotonic()
            while len(self.imported_workspaces) > REMEMBERED_WORKSPACES:
                self.imported_workspaces.popitem(last=False)

    async def _receive_body(self, scope, receive, send, limit):
        received = 0
        rejection = None
        started = False
        complete = False
        deadline = time.monotonic() + self.body_total_seconds

        async def bounded_receive():
            nonlocal received, rejection, complete
            if complete:
                return await receive()
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError
                message = await asyncio.wait_for(receive(), timeout=min(self.body_idle_seconds, remaining))
            except TimeoutError:
                rejection = _BodyRejected(408, "Request body timed out. Please retry the upload.")
                raise rejection from None
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejection = _BodyRejected(413, "Request body is too large.")
                    raise rejection
                complete = not message.get("more_body", False)
            return message

        async def bounded_send(message):
            nonlocal started
            if rejection is not None:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, bounded_send)
        except _BodyRejected as exc:
            rejection = exc
        if rejection is not None and not started:
            await JSONResponse({"detail": rejection.detail}, status_code=rejection.status)(scope, receive, send)


class _BodyRejected(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail
