"""Bound expensive synchronous requests without queuing behind model calls.

The admission limit is shared by expensive inference requests in one server
process. Background jobs have their own existing admission limits.
"""

from __future__ import annotations

import inspect
import os
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Callable, TypeVar

from fastapi import HTTPException

INFERENCE_LIMIT_ENV = "LUMINRAG_MAX_CONCURRENT_INFERENCE"
DEFAULT_MAX_CONCURRENT_INFERENCE = 4
MAX_QUERY_CHARACTERS = 8_000
RETRY_AFTER_SECONDS = 2
T = TypeVar("T")


def configured_inference_limit() -> int:
    raw = os.getenv(INFERENCE_LIMIT_ENV, str(DEFAULT_MAX_CONCURRENT_INFERENCE))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{INFERENCE_LIMIT_ENV} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{INFERENCE_LIMIT_ENV} must be a positive integer.")
    return value


class InferenceAdmission:
    """A fixed per-process capacity shared by all guarded endpoints."""

    def __init__(self, max_concurrent: int) -> None:
        if not isinstance(max_concurrent, int) or isinstance(max_concurrent, bool) or max_concurrent < 1:
            raise ValueError("Inference concurrency must be a positive integer.")
        self.max_concurrent = max_concurrent
        self._slots = threading.BoundedSemaphore(max_concurrent)

    @contextmanager
    def reserve(self):
        if not self._slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Too many requests are generating answers. Please retry shortly.",
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        try:
            yield
        finally:
            self._slots.release()


admission = InferenceAdmission(configured_inference_limit())


def inference_limited(fn: Callable[..., T]) -> Callable[..., T]:
    """Reject overload before endpoint preparation or provider work begins."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with admission.reserve():
            return fn(*args, **kwargs)

    # FastAPI resolves annotations against the wrapper's module. Resolve the
    # original function's annotations now so its request models still bind.
    wrapper.__signature__ = inspect.signature(fn, eval_str=True)
    return wrapper
