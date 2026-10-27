"""Bound process-local job history while preserving work that is still active."""

from __future__ import annotations

import threading
from collections.abc import Collection, MutableMapping
from typing import TypeVar

MAX_ACTIVE_JOBS = 8
MAX_COMPLETED_JOBS = 128
MAX_RETAINED_ITEMS = 128
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
REGISTRY_LOCK = threading.RLock()
T = TypeVar("T")


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
