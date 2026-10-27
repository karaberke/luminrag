"""Kill real publishing/recovery processes; verify restart sees one full revision."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.db.document_store import DocumentStore
from backend.tests.crash_recovery_worker import (
    CRASH_EXIT,
    DATABASE,
    JOURNAL,
    MIXED_FILES,
    NEW_FILES,
    OLD_FILES,
    new_chunks,
    old_chunks,
    seed,
)


REPOSITORY = Path(__file__).resolve().parents[2]
FILE_COUNT = len(NEW_FILES)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    seed(tmp_path)
    return tmp_path


def _child(root: Path, mode: str, phase: str = "never") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "backend.tests.crash_recovery_worker", str(root), mode, phase],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _crash(root: Path, mode: str, phase: str) -> None:
    result = _child(root, mode, phase)
    assert result.returncode == CRASH_EXIT, (
        f"Did not reach {mode} failpoint {phase}: exit={result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    assert (root / "crashed-at.txt").read_text(encoding="utf-8") == phase


def _recover(root: Path) -> None:
    result = _child(root, "recover")
    assert result.returncode == 0, f"Recovery failed:\n{result.stdout}\n{result.stderr}"


def _assert_revision(root: Path, revision: str) -> None:
    if revision == "empty":
        expected_files = dict.fromkeys(OLD_FILES)
        expected_chunks = []
    elif revision in {"new", "mixed"}:
        expected_files = MIXED_FILES if revision == "mixed" else NEW_FILES
        expected_chunks = [old_chunks()[1], *new_chunks()]
    else:
        expected_files = OLD_FILES
        expected_chunks = old_chunks()

    for relative, expected in expected_files.items():
        destination = root / relative
        if expected is None:
            assert not destination.exists(), f"Unexpected surviving {relative} after {revision} recovery"
        else:
            assert destination.read_bytes() == expected, f"Mixed file revision at {relative}"

    with DocumentStore(root / DATABASE) as store:
        rows = sorted((chunk.id, chunk.source_id, chunk.text) for chunk in store.get_all_chunks())
    assert rows == sorted((chunk.id, chunk.source_id, chunk.text) for chunk in expected_chunks)


def _recover_twice(root: Path, revision: str) -> None:
    # Both recoveries run in fresh processes with fresh SQLite connections.
    for _ in range(2):
        _recover(root)
        _assert_revision(root, revision)
        assert not list(root.rglob("*.luminrag-*")), "Recovery must clean transaction companions"
        assert not list((root / JOURNAL).glob("*.json"))
        with DocumentStore(root / DATABASE) as store:
            assert store._conn.execute("SELECT COUNT(*) FROM storage_transactions").fetchone()[0] == 0


@pytest.mark.parametrize("phase", [
    "preparing",
    *(f"{event}:{index}" for event in ("before_backup", "after_backup")
      for index, content in enumerate(OLD_FILES.values()) if content is not None),
    *(f"{event}:{index}" for event in ("before_copy", "after_copy") for index in range(FILE_COUNT)),
    "during_backup", "during_copy",
])
def test_crash_during_preparation_keeps_old_state_and_cleans_partial_copies(dataset, phase):
    _crash(dataset, "publish", phase)
    _assert_revision(dataset, "old")
    _recover_twice(dataset, "old")


def test_crash_in_middle_of_restoring_backup_recovers_from_original_copy(dataset):
    _crash(dataset, "publish", "published:0")
    _crash(dataset, "recover", "during_restore")
    _recover_twice(dataset, "old")


@pytest.mark.parametrize("phase", [
    "prepared",
    *(f"published:{index}" for index in range(FILE_COUNT)),
    "before_sql_commit",
    "sql_rows_uncommitted",
])
def test_uncommitted_publication_recovers_entire_old_revision(dataset, phase):
    _crash(dataset, "publish", phase)
    _recover_twice(dataset, "old")


@pytest.mark.parametrize("phase", [
    "after_sql_commit",
    "sql_committed_before_manager_resumes",
    *(f"cleanup:{index}" for index in range(FILE_COUNT)),
    "journal_removed",
    "cleanup_complete",
])
def test_committed_publication_recovers_entire_new_revision(dataset, phase):
    _crash(dataset, "publish", phase)
    _recover_twice(dataset, "new")


@pytest.mark.parametrize("phase", [
    *(f"rollback:{index}" for index in range(FILE_COUNT)),
    "rollback_complete",
    "cleanup:0",
    "cleanup_complete",
])
def test_crash_during_rollback_or_its_cleanup_recovers_old_revision(dataset, phase):
    _crash(dataset, "rollback", phase)
    _recover_twice(dataset, "old")


@pytest.mark.parametrize("first_mode,first_phase,recovery_phase,revision", [
    ("publish", "published:2", "rollback:0", "old"),
    ("publish", "published:5", "rollback:4", "old"),
    ("rollback", "rollback:2", "rollback:0", "old"),
    ("publish", "after_sql_commit", "cleanup:0", "new"),
])
def test_recovery_can_itself_crash_and_restart(dataset, first_mode, first_phase, recovery_phase, revision):
    _crash(dataset, first_mode, first_phase)
    _crash(dataset, "recover", recovery_phase)
    _recover_twice(dataset, revision)


@pytest.mark.parametrize("phase,revision", [
    ("published:1", "old"),
    ("before_sql_commit", "old"),
    ("sql_rows_uncommitted", "old"),
    ("after_sql_commit", "empty"),
    ("sql_committed_before_manager_resumes", "empty"),
    ("cleanup:0", "empty"),
    ("journal_removed", "empty"),
])
def test_clear_crash_recovers_whole_dataset_or_whole_clear(dataset, phase, revision):
    _crash(dataset, "clear", phase)
    _recover_twice(dataset, revision)


def test_pending_committed_upload_cannot_resurrect_data_after_clear(dataset):
    _crash(dataset, "publish", "cleanup:0")
    _assert_revision(dataset, "new")
    # publish(clear) must first settle the older upload journal. Its stale
    # backups must never be replayed after the clear's SQLite marker commits.
    _crash(dataset, "clear", "after_sql_commit")
    _recover_twice(dataset, "empty")


def test_missing_backup_blocks_recovery_instead_of_serving_mixed_state(dataset):
    _crash(dataset, "publish", "published:0")
    journal_path, = (dataset / JOURNAL).glob("*.json")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    Path(journal["operations"][0]["backup"]).unlink()
    result = _child(dataset, "recover")
    assert result.returncode not in (0, CRASH_EXIT), "Missing durable state must fail startup closed"
    assert journal_path.exists(), "Recovery evidence must survive a failed restore"


@pytest.mark.parametrize("corruption", [
    "invalid_json", "invalid_version", "invalid_flags", "duplicate_destination",
    "outside_destination", "outside_companion", "database_destination", "directory_destination",
])
def test_invalid_journal_refuses_recovery_without_touching_data(dataset, corruption):
    _crash(dataset, "publish", "published:0")
    journal_path, = (dataset / JOURNAL).glob("*.json")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    operation = journal["operations"][0]
    external = dataset.parent / f"{dataset.name}-unrelated.txt"
    external.write_bytes(b"PRIVATE UNRELATED DATA")
    if corruption == "invalid_version":
        journal["version"] = 99
    elif corruption == "invalid_flags":
        operation["existed"] = "false"
    elif corruption == "duplicate_destination":
        journal["operations"].append(dict(operation))
    elif corruption == "outside_destination":
        operation["destination"] = str(external)
    elif corruption == "outside_companion":
        operation["backup"] = str(external)
    elif corruption == "database_destination":
        operation["destination"] = str(dataset / DATABASE)
    elif corruption == "directory_destination":
        operation["destination"] = str(dataset / "raw")
    journal_path.write_text("{" if corruption == "invalid_json" else json.dumps(journal), encoding="utf-8")

    before = {
        relative: (dataset / relative).read_bytes() if (dataset / relative).exists() else None
        for relative in OLD_FILES
    }
    result = _child(dataset, "recover")
    assert result.returncode not in (0, CRASH_EXIT), "Invalid storage evidence must fail startup closed"
    for relative, content in before.items():
        destination = dataset / relative
        assert (destination.read_bytes() if destination.exists() else None) == content
    assert external.read_bytes() == b"PRIVATE UNRELATED DATA"
    assert journal_path.exists()
    with DocumentStore(dataset / DATABASE) as store:
        assert {chunk.id for chunk in store.get_all_chunks()} == {chunk.id for chunk in old_chunks()}


def test_recovery_path_that_is_a_regular_file_blocks_startup(dataset):
    (dataset / JOURNAL).write_text("unreadable recovery directory", encoding="utf-8")
    result = _child(dataset, "recover")
    assert result.returncode not in (0, CRASH_EXIT), "An invalid recovery directory must fail startup closed"
    _assert_revision(dataset, "old")


@pytest.mark.parametrize("phase", [
    "preparing", "prepared",
    *(f"{event}:{index}" for event in ("before_delete_move", "after_delete_move")
      for index, content in enumerate(OLD_FILES.values()) if content is not None),
    "delete_move_applied",
])
def test_copy_free_clear_crash_before_or_after_atomic_rename_keeps_old_revision(dataset, phase):
    _crash(dataset, "clear", phase)
    _recover_twice(dataset, "old")


@pytest.mark.parametrize("phase", [
    "before_delete_restore:0", "after_delete_restore:0",
    "before_delete_restore:5", "after_delete_restore:5",
    "rollback:2", "rollback_complete", "cleanup:0", "journal_removed",
])
def test_copy_free_clear_rollback_can_crash_and_resume(dataset, phase):
    _crash(dataset, "clear_rollback", phase)
    _recover_twice(dataset, "old")


def test_clear_recovery_can_exit_after_backup_is_renamed_back_before_hook(dataset):
    _crash(dataset, "clear", "before_sql_commit")
    _crash(dataset, "recover", "delete_restore_applied")
    _recover_twice(dataset, "old")


@pytest.mark.parametrize("phase,revision", [
    ("prepared", "old"), ("published:1", "old"), ("before_sql_commit", "old"),
    ("after_sql_commit", "mixed"), ("cleanup:0", "mixed"),
])
def test_replacements_and_renamed_deletions_share_atomic_recovery(dataset, phase, revision):
    _crash(dataset, "mixed", phase)
    _recover_twice(dataset, revision)


def test_mixed_rollback_can_restart_after_a_delete_backup_has_been_consumed(dataset):
    _crash(dataset, "mixed_rollback", "after_delete_restore:1")
    _recover_twice(dataset, "old")


@pytest.mark.parametrize("rollback", [False, True])
def test_clear_and_rollback_never_copy_corpus_content(dataset, monkeypatch, rollback):
    from unittest.mock import Mock
    from backend.utils import recovery

    copy = Mock(side_effect=OSError("no space for a duplicate corpus"))
    monkeypatch.setattr(recovery, "copy_file_durable", copy)
    with DocumentStore(dataset / DATABASE) as store:
        manager = recovery.RecoveryManager(dataset / JOURNAL, [dataset], store)

        def commit(transaction_id):
            if rollback:
                raise RuntimeError("SQLite clear rejected")
            return store.clear_all(transaction_id=transaction_id)

        files = [(None, dataset / relative) for relative in OLD_FILES]
        if rollback:
            with pytest.raises(RuntimeError, match="SQLite clear rejected"):
                manager.publish(files, commit)
        else:
            manager.publish(files, commit)
    copy.assert_not_called()
    _recover_twice(dataset, "old" if rollback else "empty")


def test_missing_both_deleted_source_and_renamed_backup_blocks_recovery(dataset):
    _crash(dataset, "clear", "after_delete_move:0")
    journal_path, = (dataset / JOURNAL).glob("*.json")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    Path(journal["operations"][0]["backup"]).unlink()
    result = _child(dataset, "recover")
    assert result.returncode not in (0, CRASH_EXIT)
    assert journal_path.exists()


@pytest.mark.parametrize("committed", [False, True])
def test_version_one_copy_backed_clear_journals_remain_recoverable(dataset, committed):
    # The previous journal format prepared copied backups before unlinking.
    # Manufacture that exact durable state from a prepared replacement plan.
    _crash(dataset, "publish", "prepared")
    journal_path, = (dataset / JOURNAL).glob("*.json")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["version"] = 1
    for operation in journal["operations"]:
        operation["replacement"] = False
        Path(operation["destination"]).unlink(missing_ok=True)
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    if committed:
        with DocumentStore(dataset / DATABASE) as store:
            store.clear_all(transaction_id=journal["transaction_id"])
    _recover_twice(dataset, "empty" if committed else "old")


def test_version_one_missing_backup_still_fails_even_when_original_survives(dataset):
    _crash(dataset, "publish", "prepared")
    journal_path, = (dataset / JOURNAL).glob("*.json")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["version"] = 1
    for operation in journal["operations"]:
        operation["replacement"] = False
    Path(journal["operations"][0]["backup"]).unlink()
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    result = _child(dataset, "recover")
    assert result.returncode not in (0, CRASH_EXIT), "Only v2 rename-backed deletes may accept an untouched original"
    assert journal_path.exists()
