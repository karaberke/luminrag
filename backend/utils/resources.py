"""Bound interactive resource waits while letting durable background work wait."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_allow_waiting: ContextVar[bool] = ContextVar("allow_model_resource_waiting", default=True)
INTERACTIVE_RESOURCE_TIMEOUT = 0.25


class ModelResourcesBusy(RuntimeError):
    """Retryable contention with background inference or graph cache work."""

    def __init__(self) -> None:
        super().__init__("Materials are still being prepared. Please retry shortly.")


@contextmanager
def nonblocking_model_resources():
    """Use bounded interactive admission so a busy resource cannot wait forever."""
    token = _allow_waiting.set(False)
    try:
        yield
    finally:
        _allow_waiting.reset(token)


@contextmanager
def background_model_resources():
    """Detach durable background work from a request's fail-fast policy."""
    token = _allow_waiting.set(True)
    try:
        yield
    finally:
        _allow_waiting.reset(token)


def model_resource_timeout() -> float | None:
    """One deadline policy for state locks and shared native inference."""
    return None if _allow_waiting.get() else INTERACTIVE_RESOURCE_TIMEOUT


@contextmanager
def reserve_model_resource(lock: Any):
    """Wait normally in background; bound each interactive reservation."""
    timeout = model_resource_timeout()
    acquired = lock.acquire(blocking=True) if timeout is None else lock.acquire(blocking=True, timeout=timeout)
    if not acquired:
        raise ModelResourcesBusy()
    try:
        yield
    finally:
        lock.release()
