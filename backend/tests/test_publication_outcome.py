"""Commit proof survives recovery cleanup and transient SQLite read failures."""

from __future__ import annotations

import sqlite3
import pytest

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk
from backend.utils.recovery import PublicationOutcome, RecoveryManager


@pytest.fixture
def publication(tmp_path):
    with DocumentStore(tmp_path / "chunks.db") as store:
        store.save_chunks([Chunk(id="chunk", source_id="source", modality="pdf", text="old")])
        original, candidate = tmp_path / "live", tmp_path / "candidate"
        original.write_bytes(b"old")
        candidate.write_bytes(b"new")
        manager = RecoveryManager(tmp_path / "recovery", [tmp_path], store)
        yield store, manager, original, candidate


def _commit(store, transaction_id):
    return store.replace_sources(
        [Chunk(id="chunk", source_id="source", modality="pdf", text="new")],
        transaction_id=transaction_id,
    )


def test_successful_outcome_retains_commit_proof_after_marker_and_journal_removed(publication):
    store, manager, original, candidate = publication
    outcome = PublicationOutcome()
    manager.publish([(candidate, original)], lambda tx: _commit(store, tx), outcome=outcome)
    assert outcome.status == "committed"
    assert outcome.transaction_id is not None
    assert not store.transaction_committed(outcome.transaction_id)
    assert not list(manager.journal_dir.glob("*.json"))
    manager.recover()
    assert outcome.status == "committed"


@pytest.mark.parametrize("callback_raises", [False, True])
def test_later_recovery_updates_pending_outcome_using_positive_sql_proof(publication, monkeypatch, callback_raises):
    store, manager, original, candidate = publication
    outcome = PublicationOutcome()
    read_marker = store.transaction_committed
    failures = 1 if callback_raises else 2

    def temporarily_unreadable(transaction_id):
        nonlocal failures
        if failures:
            failures -= 1
            raise sqlite3.OperationalError("temporary marker read failure")
        return read_marker(transaction_id)

    def commit(transaction_id):
        result = _commit(store, transaction_id)
        if callback_raises:
            raise RuntimeError("callback raised after actual SQL commit")
        return result

    monkeypatch.setattr(store, "transaction_committed", temporarily_unreadable)
    with pytest.raises(sqlite3.OperationalError):
        manager.publish([(candidate, original)], commit, outcome=outcome)
    assert outcome.status == "pending"
    assert original.read_bytes() == b"new"
    with DocumentStore(store.path) as reopened:
        assert reopened.get_chunk("chunk").text == "new"
    assert manager.recover() == 1
    assert outcome.status == "committed"
    assert not read_marker(outcome.transaction_id)
    manager.recover()
    assert outcome.status == "committed"


def test_unreadable_marker_keeps_outcome_pending_until_recovery_succeeds(publication, monkeypatch):
    store, manager, original, candidate = publication
    outcome = PublicationOutcome()
    read_marker = store.transaction_committed

    def unavailable(_transaction_id):
        raise sqlite3.OperationalError("marker unavailable")

    monkeypatch.setattr(store, "transaction_committed", unavailable)
    with pytest.raises(sqlite3.OperationalError):
        manager.publish([(candidate, original)], lambda tx: _commit(store, tx), outcome=outcome)
    with pytest.raises(sqlite3.OperationalError):
        manager.recover()
    assert outcome.status == "pending"
    assert list(manager.journal_dir.glob("*.json"))
    monkeypatch.setattr(store, "transaction_committed", read_marker)
    manager.recover()
    assert outcome.status == "committed"


def test_publication_input_failure_is_known_unpublished_and_keeps_storage_usable(publication):
    store, manager, original, _candidate = publication
    outcome = PublicationOutcome()

    def broken_input():
        raise RuntimeError("input enumeration failed")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="enumeration"):
        manager.publish(broken_input(), lambda tx: _commit(store, tx), outcome=outcome)
    assert outcome.status == "rolled_back"
    assert original.read_bytes() == b"old"
    assert manager.recover() == 0
    assert store.get_chunk("chunk").text == "old"


@pytest.mark.parametrize("failure_phase", ["preparing", "prepared", "rollback_complete", "cleanup:0"])
def test_rollback_outcome_never_claims_commit_when_marker_absent(publication, failure_phase):
    store, manager, original, candidate = publication
    outcome = PublicationOutcome()
    fired = False

    def fail(phase):
        nonlocal fired
        if phase == failure_phase and not fired:
            fired = True
            raise OSError("injected interruption")

    def abort(_transaction_id):
        raise RuntimeError("SQL never committed")

    manager.failpoint = fail
    with pytest.raises((RuntimeError, OSError)):
        manager.publish([(candidate, original)], abort, outcome=outcome)
    assert fired
    assert outcome.status == "rolled_back"
    assert original.read_bytes() == b"old"
    assert store.get_chunk("chunk").text == "old"
    manager.recover()
    assert outcome.status == "rolled_back"
