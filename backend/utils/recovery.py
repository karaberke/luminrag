"""Restart-safe coordination of published files with a SQLite commit marker.

Call under the application's mutation lock. This supports one writer process.
Recovery is deliberately fail-closed: a malformed journal or failed restore
prevents loading mixed state. File contents are fsynced on all platforms;
directory fsync is additionally used where the operating system supports it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from backend.db.document_store import DocumentStore
from backend.utils.files import atomic_write_text, copy_file_durable, ensure_directory_durable, fsync_directory

logger = logging.getLogger(__name__)
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$")
_MAX_JOURNAL_BYTES = 64 * 1024 * 1024
_MAX_OPERATIONS = 100_000


class RecoveryError(RuntimeError):
    """Recovery cannot safely determine or restore the committed state."""


@dataclass
class PublicationOutcome:
    """Retain positive commit/rollback proof after recovery removes its files.

    A pending transaction may have committed even when a marker read failed.
    Callers must preserve its candidate assets until recovery establishes a
    terminal outcome; absence of a marker alone never proves a commit.
    """

    transaction_id: str | None = None
    status: Literal["not_started", "pending", "committed", "rolled_back"] = "not_started"


class RecoveryManager:
    def __init__(
        self,
        journal_dir: Path,
        allowed_roots: Iterable[Path],
        store: DocumentStore,
        *,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        self.roots = tuple(Path(root).resolve() for root in allowed_roots)
        if not self.roots:
            raise RecoveryError("At least one storage root is required.")
        if any(root.exists() and not root.is_dir() for root in self.roots):
            raise RecoveryError("Storage roots must be directories.")
        self.journal_dir = Path(journal_dir).absolute()
        self.store = store
        self.failpoint = failpoint
        self._outcome: PublicationOutcome | None = None
        self._validate_path(self.journal_dir)
        if self.journal_dir.exists() and not self.journal_dir.is_dir():
            raise RecoveryError("The recovery journal path must be a directory.")
        database = store.path.resolve()
        self._forbidden = {database, Path(str(database) + "-wal"), Path(str(database) + "-shm")}

    def _hook(self, phase: str) -> None:
        if self.failpoint is not None:
            self.failpoint(phase)

    def _record_outcome(self, transaction_id: str, status: Literal["committed", "rolled_back"]) -> None:
        if self._outcome is not None and self._outcome.transaction_id == transaction_id:
            self._outcome.status = status

    def _validate_path(self, path: Path) -> Path:
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts:
            raise RecoveryError("Recovery paths must be absolute and confined to storage roots.")
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.roots):
            raise RecoveryError(f"Path outside configured storage roots: {path}")
        for ancestor in (path, *path.parents):
            if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
                raise RecoveryError(f"Storage paths must not traverse links: {path}")
        if path.exists() and not path.is_file() and not path.is_dir():
            raise RecoveryError(f"Unsupported storage file: {path}")
        return resolved

    def _destination(self, path: Path) -> Path:
        path = self._validate_path(Path(path).absolute())
        if path in self._forbidden or path == self.journal_dir or self.journal_dir in path.parents:
            raise RecoveryError("A publication cannot replace its database or recovery journal.")
        if path.is_dir():
            raise RecoveryError("Recovery operates on regular files, never directories.")
        return path

    @staticmethod
    def _companions(destination: Path, transaction_id: str) -> dict[str, Path]:
        stem = f".{destination.name}.luminrag-{transaction_id}"
        return {kind: destination.with_name(f"{stem}.{kind}") for kind in ("new", "bak", "restore")}

    def _journal_path(self, transaction_id: str) -> Path:
        return self.journal_dir / f"{transaction_id}.json"

    def _write_journal(self, journal: dict) -> None:
        payload = json.dumps(journal)
        if len(payload.encode("utf-8")) > _MAX_JOURNAL_BYTES - 128:
            raise RecoveryError("Storage transaction journal is too large.")
        atomic_write_text(self._journal_path(journal["transaction_id"]), payload)

    def _read_journal(self, path: Path) -> dict:
        self._validate_path(path)
        if path.is_symlink() or path.stat().st_size > _MAX_JOURNAL_BYTES:
            raise RecoveryError("Invalid recovery journal file.")
        try:
            journal = json.loads(path.read_text(encoding="utf-8"))
            transaction_id = journal["transaction_id"]
            if journal.get("version") not in {1, 2} or not _TRANSACTION_ID.fullmatch(transaction_id):
                raise ValueError("Invalid transaction ID or journal version")
            if path.name != f"{transaction_id}.json" or journal["phase"] not in {"PREPARING", "PREPARED", "ROLLED_BACK"}:
                raise ValueError("Invalid journal name or phase")
            operations = journal["operations"]
            if not isinstance(operations, list) or len(operations) > _MAX_OPERATIONS:
                raise ValueError("Invalid operation list")
            seen = set()
            for operation in operations:
                destination = self._destination(Path(operation["destination"]))
                if destination in seen:
                    raise ValueError("Duplicate publication destination")
                seen.add(destination)
                companions = self._companions(destination, transaction_id)
                for kind, field in (("new", "prepared"), ("bak", "backup"), ("restore", "restore")):
                    if Path(operation[field]) != companions[kind]:
                        raise ValueError("Invalid recovery companion path")
                    self._validate_path(companions[kind])
                if type(operation["existed"]) is not bool or type(operation["replacement"]) is not bool:
                    raise ValueError("Invalid operation flags")
            return journal
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RecoveryError(f"Malformed recovery journal: {path.name}") from error

    def _unlink(self, path: Path) -> None:
        self._validate_path(path)
        if path.exists():
            path.unlink()
            fsync_directory(path.parent)

    def _rollback(self, journal: dict) -> None:
        for index, operation in enumerate(journal["operations"]):
            destination = self._destination(Path(operation["destination"]))
            if journal["version"] == 2 and not operation["replacement"]:
                self._restore_deleted_file(operation, destination, index)
            elif operation["existed"]:
                backup, restore = Path(operation["backup"]), Path(operation["restore"])
                self._validate_path(backup)
                self._validate_path(restore)
                if not backup.is_file():
                    raise RecoveryError(f"Required recovery backup missing: {backup}")
                self._unlink(restore)
                copy_file_durable(backup, restore)
                os.replace(restore, destination)
                fsync_directory(destination.parent)
            else:
                self._unlink(destination)
            self._hook(f"rollback:{index}")
        # Backups remain necessary until this terminal phase is durable.
        journal["phase"] = "ROLLED_BACK"
        self._write_journal(journal)
        self._record_outcome(journal["transaction_id"], "rolled_back")
        self._hook("rollback_complete")

    def _restore_deleted_file(self, operation: dict, destination: Path, index: int) -> None:
        """Undo a v2 deletion without copying or deleting an untouched original.

        The rename is atomic. A missing backup with a surviving destination
        means the deletion was never applied or a prior recovery restored it.
        Keeping the journal until every entry is restored makes both cases
        safe after repeated process termination, without per-file rewrites.
        """
        if not operation["existed"]:
            self._unlink(destination)
            return
        backup = Path(operation["backup"])
        self._validate_path(backup)
        if backup.is_file():
            self._hook(f"before_delete_restore:{index}")
            os.replace(backup, destination)
            fsync_directory(destination.parent)
            self._hook(f"after_delete_restore:{index}")
        elif not destination.is_file():
            raise RecoveryError(f"Both deleted source and its recovery backup are missing: {destination}")

    def _cleanup(self, journal: dict) -> None:
        for index, operation in enumerate(journal["operations"]):
            for field in ("prepared", "backup", "restore"):
                self._unlink(Path(operation[field]))
            self._hook(f"cleanup:{index}")
        self._unlink(self._journal_path(journal["transaction_id"]))
        # Keep the SQL marker until the journal removal has been flushed.
        self._hook("journal_removed")
        self.store.forget_transaction(journal["transaction_id"])
        self._hook("cleanup_complete")

    def _recover_journal(self, journal: dict) -> None:
        committed = self.store.transaction_committed(journal["transaction_id"])
        if journal["phase"] in {"PREPARING", "ROLLED_BACK"}:
            if committed:
                raise RecoveryError("An unpublished or rolled-back journal unexpectedly has a committed SQL marker.")
            self._record_outcome(journal["transaction_id"], "rolled_back")
        elif not committed:
            self._rollback(journal)
        else:
            self._record_outcome(journal["transaction_id"], "committed")
        self._cleanup(journal)

    def recover(self) -> int:
        """Resolve pending journals before reading an index, graph, or raw source."""
        if not self.journal_dir.exists():
            self.store.prune_orphaned_transactions()
            return 0
        self._validate_path(self.journal_dir)
        if not self.journal_dir.is_dir():
            raise RecoveryError("The recovery journal path must be a directory.")
        paths = sorted(self.journal_dir.glob("*.json"))
        # Validate every journal before making any recovery change.
        journals = [self._read_journal(path) for path in paths]
        for journal in journals:
            self._recover_journal(journal)
        # An exit after durable journal removal can leave only its SQL marker.
        # No active transaction exists here under the single-writer lock, and
        # all validated journals have now been resolved successfully.
        self.store.prune_orphaned_transactions()
        return len(journals)

    def publish(
        self, files: Iterable[tuple[Path | None, Path]], commit: Callable[[str], Any],
        *, outcome: PublicationOutcome | None = None,
    ) -> Any:
        """Prepare durable recovery material, publish files, and commit SQL once."""
        self.recover()
        transaction_id = uuid.uuid4().hex
        self._outcome = outcome if outcome is not None else PublicationOutcome()
        self._outcome.transaction_id = transaction_id
        self._outcome.status = "pending"
        journal = {"version": 2, "transaction_id": transaction_id, "phase": "PREPARING", "operations": []}
        try:
            material = list(files)
            if len(material) > _MAX_OPERATIONS:
                raise RecoveryError("Too many files in a single storage transaction.")
        except BaseException:
            # Enumeration/limits run before creating any recovery material or
            # changing a live file, so this outcome is positively known.
            self._record_outcome(transaction_id, "rolled_back")
            raise
        seen = set()
        sources: list[Path | None] = []
        try:
            for source, target in material:
                destination = self._destination(Path(target))
                if destination in seen:
                    raise RecoveryError("Duplicate publication destination.")
                seen.add(destination)
                companions = self._companions(destination, transaction_id)
                operation = {
                    "destination": str(destination), "prepared": str(companions["new"]),
                    "backup": str(companions["bak"]), "restore": str(companions["restore"]),
                    "existed": destination.exists(), "replacement": source is not None,
                }
                journal["operations"].append(operation)
                if source is not None:
                    source = self._validate_path(Path(source).absolute())
                    if not source.is_file():
                        raise RecoveryError("Publication source must be a regular file.")
                sources.append(source)
            ensure_directory_durable(self.journal_dir)
            # Track preparation itself: an exit during a large copy must not
            # leave untracked old-data backups or partial candidate files.
            self._write_journal(journal)
            self._hook("preparing")
            for index, (operation, source) in enumerate(zip(journal["operations"], sources)):
                destination = Path(operation["destination"])
                ensure_directory_durable(destination.parent)
                # Replacements need a copied old revision before publication.
                # Deletions retain the original inode via a same-directory
                # rename after PREPARED, avoiding a second copy of the corpus.
                if operation["existed"] and operation["replacement"]:
                    self._hook(f"before_backup:{index}")
                    copy_file_durable(destination, Path(operation["backup"]))
                    self._hook(f"after_backup:{index}")
                if source is not None:
                    self._hook(f"before_copy:{index}")
                    copy_file_durable(source, Path(operation["prepared"]))
                    self._hook(f"after_copy:{index}")
            journal["phase"] = "PREPARED"
            self._write_journal(journal)
        except BaseException:
            # No live file changes precede a durable journal. If writing the
            # journal itself failed after replacement, retain recovery material.
            if self._journal_path(transaction_id).exists():
                self._recover_journal(self._read_journal(self._journal_path(transaction_id)))
            else:
                for operation in journal["operations"]:
                    for field in ("prepared", "backup", "restore"):
                        self._unlink(Path(operation[field]))
                self._record_outcome(transaction_id, "rolled_back")
            raise

        result = None
        try:
            self._hook("prepared")
            for index, operation in enumerate(journal["operations"]):
                destination = Path(operation["destination"])
                if operation["replacement"]:
                    os.replace(Path(operation["prepared"]), destination)
                    fsync_directory(destination.parent)
                elif operation["existed"]:
                    self._hook(f"before_delete_move:{index}")
                    os.replace(destination, Path(operation["backup"]))
                    fsync_directory(destination.parent)
                    self._hook(f"after_delete_move:{index}")
                self._hook(f"published:{index}")
            self._hook("before_sql_commit")
            result = commit(transaction_id)
            if not self.store.transaction_committed(transaction_id):
                raise RecoveryError("SQL publication did not commit its recovery marker.")
            self._record_outcome(transaction_id, "committed")
            self._hook("after_sql_commit")
        except BaseException as error:
            # A callback can raise after SQLite actually committed; the marker,
            # rather than the Python return path, decides which version is valid.
            if not self.store.transaction_committed(transaction_id):
                self._recover_journal(journal)
                raise
            self._record_outcome(transaction_id, "committed")
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning("Storage committed despite a post-commit error: %s", error)
        try:
            self._cleanup(journal)
        except Exception:
            # State is committed. Keep remaining journal/marker material for
            # restart cleanup instead of falsely reporting a rolled-back upload.
            logger.exception("Committed storage cleanup deferred until recovery")
        return result
