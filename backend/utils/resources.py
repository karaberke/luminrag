"""Never wait for inference resources while holding shared application state."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_allow_waiting: ContextVar[bool] = ContextVar("allow_model_resource_waiting", default=True)


class ModelResourcesBusy(RuntimeError):
    """Retryable contention with background inference or graph cache work."""

    def __init__(self) -> None:
        super().__init__("Materials are still being prepared. Please retry shortly.")


@contextmanager
def nonblocking_model_resources():
    """Use inside the mutation lock so a busy model cannot create a convoy."""
    token = _allow_waiting.set(False)
    try:
        yield
    finally:
        _allow_waiting.reset(token)


@contextmanager
def reserve_model_resource(lock: Any):
    """Background work waits normally; state-locked callers fail promptly."""
    if not lock.acquire(blocking=_allow_waiting.get()):
        raise ModelResourcesBusy()
    try:
        yield
    finally:
        lock.release()
