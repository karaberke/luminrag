"""Bounded histories must never evict active work or over-admit concurrent jobs."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from backend.utils.jobs import (
    JobCapacityError, prune_terminal_jobs, register_job, retain_recent,
)


def _job(status="queued"):
    return SimpleNamespace(status=status)


def test_retention_preserves_active_jobs_and_newest_terminal_history():
    jobs = {"running": _job("running"), "old-done": _job("done"), "queued": _job(),
            "old-failed": _job("failed"), "new-cancelled": _job("cancelled")}
    removed = prune_terminal_jobs(jobs, max_completed=1)
    assert removed == ["old-done", "old-failed"]
    assert list(jobs) == ["running", "queued", "new-cancelled"]


def test_active_capacity_rejection_preserves_admitted_work():
    first = _job("running")
    jobs = {"first": first}
    with pytest.raises(JobCapacityError):
        register_job(jobs, "second", _job(), max_active=1)
    assert jobs == {"first": first}
    first.status = "done"
    register_job(jobs, "second", _job(), max_active=1)
    assert jobs["second"].status == "queued"


def test_duplicate_registration_cannot_replace_running_job():
    first = _job("running")
    jobs = {"first": first}
    with pytest.raises(ValueError):
        register_job(jobs, "first", _job())
    assert jobs["first"] is first


def test_concurrent_admission_respects_shared_capacity():
    jobs = {}

    def admit(identifier):
        try:
            register_job(jobs, str(identifier), _job(), max_active=3)
            return True
        except JobCapacityError:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        admitted = list(pool.map(admit, range(64)))
    assert sum(admitted) == 3
    assert len(jobs) == 3


def test_repeated_completions_keep_history_bounded():
    jobs = {}
    for index in range(1000):
        register_job(jobs, str(index), _job(), max_active=2, max_completed=7)
        jobs[str(index)].status = "done"
        prune_terminal_jobs(jobs, max_completed=7)
    assert list(jobs) == [str(index) for index in range(993, 1000)]


def test_completed_item_retention_reports_evicted_ids_for_dependent_jobs():
    items = {str(index): object() for index in range(5)}
    assert retain_recent(items, max_items=2) == ["0", "1", "2"]
    assert list(items) == ["3", "4"]


def test_zero_history_limit_preserves_only_active_work():
    jobs = {"done": _job("done"), "active": _job("running")}
    assert prune_terminal_jobs(jobs, max_completed=0) == ["done"]
    assert list(jobs) == ["active"]
    items = {"item": object()}
    assert retain_recent(items, max_items=0) == ["item"]
    assert items == {}


def test_zero_active_limit_rejects_without_registering():
    jobs = {}
    with pytest.raises(JobCapacityError):
        register_job(jobs, "job", _job(), max_active=0)
    assert jobs == {}


@pytest.mark.parametrize("operation", [
    lambda: retain_recent({}, max_items=-1),
    lambda: prune_terminal_jobs({}, max_completed=-1),
    lambda: register_job({}, "job", _job(), max_active=-1),
])
def test_invalid_limits_do_not_silently_disable_retention(operation):
    with pytest.raises(ValueError):
        operation()
