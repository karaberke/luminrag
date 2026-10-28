"""Bound process-local job history while preserving work that is still active."""

from __future__ import annotations

import threading
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import ContextVar, copy_context
from collections.abc import Collection, MutableMapping
from typing import Callable, TypeVar

MAX_ACTIVE_JOBS = 8
MAX_COMPLETED_JOBS = 128
MAX_RETAINED_ITEMS = 128
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
REGISTRY_LOCK = threading.RLock()
T = TypeVar("T")
R = TypeVar("R")


def ingestion_workers(config: dict) -> int:
    """Bound provider I/O fan-out independently of native model CPU threads."""
    maximum = min(8, max(1, int(os.getenv("LUMINRAG_INGEST_LLM_WORKERS", "8"))))
    return min(maximum, max(1, int(config.get("max_workers", maximum))))


class MapCancelledError(RuntimeError):
    """Work was skipped because another task in the same map tree failed."""


class _FailureRecord:
    """Thread-safe first-failure holder; the event is set after recording."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.failure: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    def record(self, error: BaseException) -> None:
        with self._lock:
            if self.failure is None:
                self.failure = error
            self.event.set()


# One scope per top-level map_bounded call, inherited by nested calls through
# the copied worker contexts so a failure anywhere stops new work everywhere.
_SCOPE: ContextVar[_FailureRecord | None] = ContextVar("map_bounded_scope", default=None)


def map_bounded(
    function: Callable[[T], R], items: list[T], *, max_workers: int,
    on_completed: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Run bounded I/O tasks, retaining input order and request context.

    Only one task per worker is submitted at a time. Graph mutations and
    publication remain on the caller after all independent results finish.
    After any task in the same invocation tree fails, no new task starts;
    started tasks drain and the original exception is raised.
    """
    if not items:
        return []
    scope = _SCOPE.get()
    token = _SCOPE.set(_FailureRecord()) if scope is None else None
    try:
        return _map_in_scope(function, items, _SCOPE.get(), max_workers, on_completed, owner=token is not None)
    finally:
        if token is not None:
            _SCOPE.reset(token)


def _map_in_scope(
    function: Callable[[T], R], items: list[T], scope: _FailureRecord, max_workers: int,
    on_completed: Callable[[int, int], None] | None, *, owner: bool,
) -> list[R]:
    call = _FailureRecord()

    def run(item: T) -> R:
        if scope.cancelled:
            raise MapCancelledError("Skipped after another task failed.")
        try:
            return function(item)
        except MapCancelledError:
            raise
        except BaseException as error:
            call.record(error)
            scope.record(error)
            raise

    workers = min(len(items), max(1, max_workers))
    results: dict[int, R] = {}
    pending = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest") as executor:
        try:
            while pending or (next_index < len(items) and not scope.cancelled):
                while not scope.cancelled and next_index < len(items) and len(pending) < workers:
                    context = copy_context()
                    future = executor.submit(context.run, run, items[next_index])
                    pending[future] = next_index
                    next_index += 1
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    try:
                        value = future.result()
                    except BaseException:
                        continue  # Recorded by the worker, or a cancellation skip.
                    results[index] = value
                    if on_completed and not scope.cancelled:
                        try:
                            on_completed(len(results), len(items))
                        except BaseException as error:
                            call.record(error)
                            scope.record(error)
        except BaseException as error:
            # Interrupts in the caller thread still stop the tree before the
            # executor drains in-flight work; the error is never swallowed.
            call.record(error)
            scope.record(error)
            raise
    if owner and scope.failure is not None:
        raise scope.failure
    if call.failure is not None:
        raise call.failure
    if scope.cancelled:
        raise MapCancelledError("Skipped after another task failed.")
    if len(results) != len(items):
        raise MapCancelledError("Bounded map finished without every result.")
    return [results[index] for index in range(len(items))]


class JobCapacityError(RuntimeError):
    """The allowed number of active jobs has already been admitted."""


def _validate_limit(limit: int) -> None:
    if limit < 0:
        raise ValueError("Registry limits must not be negative.")


def retain_recent(values: MutableMapping[str, T], *, max_items: int = MAX_RETAINED_ITEMS) -> list[str]:
    """Keep the newest inserted completed values, returning evicted IDs."""
    _validate_limit(max_items)
    with REGISTRY_LOCK:
        removed = list(values)[:max(0, len(values) - max_items)]
        for key in removed:
            values.pop(key, None)
        return removed


def prune_terminal_jobs(
    jobs: MutableMapping[str, T], *, max_completed: int = MAX_COMPLETED_JOBS,
    terminal_statuses: Collection[str] = TERMINAL_STATUSES,
) -> list[str]:
    """Prune oldest terminal entries; queued and running jobs always survive."""
    _validate_limit(max_completed)
    with REGISTRY_LOCK:
        completed = [key for key, job in list(jobs.items()) if getattr(job, "status", None) in terminal_statuses]
        removed = completed[:max(0, len(completed) - max_completed)]
        for key in removed:
            jobs.pop(key, None)
        return removed


def register_job(
    jobs: MutableMapping[str, T], job_id: str, job: T, *,
    max_active: int = MAX_ACTIVE_JOBS, max_completed: int = MAX_COMPLETED_JOBS,
    terminal_statuses: Collection[str] = TERMINAL_STATUSES,
) -> list[str]:
    """Atomically enforce admission/history limits and insert a queued job.

    Call before scheduling work. Callers retain responsibility for releasing
    staged uploads if admission fails. Worker status changes can only free
    active capacity; dictionary insertions and removals share REGISTRY_LOCK.
    """
    _validate_limit(max_active)
    _validate_limit(max_completed)
    with REGISTRY_LOCK:
        if job_id in jobs:
            raise ValueError("A job with this ID is already registered.")
        removed = prune_terminal_jobs(jobs, max_completed=max_completed, terminal_statuses=terminal_statuses)
        active_count = sum(getattr(current, "status", None) not in terminal_statuses for current in jobs.values())
        if active_count >= max_active:
            raise JobCapacityError("Too many jobs are active. Wait for one to finish and try again.")
        jobs[job_id] = job
        return removed
