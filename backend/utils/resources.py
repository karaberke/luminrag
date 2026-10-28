"""Bound interactive resource waits while letting durable background work wait."""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final


_allow_waiting: ContextVar[bool] = ContextVar("allow_model_resource_waiting", default=True)
INTERACTIVE_RESOURCE_TIMEOUT = 0.25
# ONNX Runtime's CPU arena keeps peak batch memory reserved; disable it for
# every pinned model so embedding and reranking peaks do not stack.
NATIVE_CPU_MEM_ARENA: Final[bool] = False

# One native batch at a time across all models, admitted in arrival order.
_native_condition = threading.Condition()
_native_waiters: deque[object] = deque()
_native_active = False


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


@contextmanager
def reserve_native_inference():
    """Admit one process-wide native batch in FIFO order; not reentrant.

    Take this inside any per-model reservation, never the other way around.
    """
    global _native_active
    timeout = model_resource_timeout()
    token = object()
    with _native_condition:
        _native_waiters.append(token)
        try:
            # Only the queue head may enter, so fresh arrivals cannot overtake.
            admitted = _native_condition.wait_for(
                lambda: not _native_active and _native_waiters[0] is token, timeout=timeout,
            )
            if admitted:
                _native_active = True
        finally:
            _native_waiters.remove(token)
            _native_condition.notify_all()
        if not admitted:
            raise ModelResourcesBusy()
    try:
        yield
    finally:
        with _native_condition:
            _native_active = False
            _native_condition.notify_all()
