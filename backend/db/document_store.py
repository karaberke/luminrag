"""
Stage 1 - Multimodal Ingestion: document_store.py

SQLite-backed persistent store for all Chunk objects produced by the
ingestion pipeline. Downstream stages (graph construction, retrieval)
use this store to trace graph nodes back to source text.

Schema (single table `chunks`):
    chunk_id  TEXT PRIMARY KEY
    source_id TEXT
    modality  TEXT
    text      TEXT
    metadata  TEXT   -- JSON-serialised Chunk.metadata dict

Public API:
    store = DocumentStore(db_path)
    store.save_chunks(chunks)
    store.replace_sources(chunks)   -> list[str] (old chunk ids of replaced sources)
    store.get_chunk(chunk_id)
    store.get_chunk_sources(chunk_ids) -> dict[str, str] (chunk id -> source id)
    store.get_chunks_by_source(source_id)
    store.get_first_chunk_by_source(source_id)
    store.get_all_chunks()
    store.delete_source(source_id)  -> int (rows deleted)
    store.count()                   -> int
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from backend.schemas import Chunk


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    modality  TEXT NOT NULL,
    text      TEXT NOT NULL,
    metadata  TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_source_id ON chunks (source_id);
"""

_CREATE_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS storage_transactions (
    transaction_id TEXT PRIMARY KEY,
    committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_INSERT = (
    "INSERT OR REPLACE INTO chunks "
    "(chunk_id, source_id, modality, text, metadata) "
    "VALUES (?, ?, ?, ?, ?);"
)


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["chunk_id"],
        source_id=row["source_id"],
        modality=row["modality"],
        text=row["text"],
        metadata=json.loads(row["metadata"]),
    )


def _chunk_to_row(chunk: Chunk) -> tuple:
    return (
        chunk.id,
        chunk.source_id,
        chunk.modality,
        chunk.text,
        json.dumps(chunk.metadata),
    )


class DocumentStore:
    """
    Thin SQLite wrapper for storing and retrieving Chunk objects.

    The database file is created automatically on first use.
    All writes use INSERT OR REPLACE so re-ingesting a source is safe.
    The connection is shared across threads, so every operation is
    serialised by a per-connection RLock.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")  # safe concurrent reads
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.executescript(_CREATE_TABLE + _CREATE_INDEX + _CREATE_TRANSACTIONS)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """
        Persist a list of Chunks. Upserts on chunk_id — calling this again
        with the same chunks after re-ingestion will overwrite, not duplicate.
        Does not remove stale chunks; use replace_sources() for re-ingestion.
        """
        rows = [_chunk_to_row(chunk) for chunk in chunks]
        with self._lock:
            self._conn.executemany(_INSERT, rows)
            self._conn.commit()

    def replace_sources(
        self, chunks: list[Chunk], *, replaced_source_ids: Iterable[str] = (), transaction_id: str | None = None
    ) -> list[str]:
        """
        Atomically replace every source present in *chunks*: all prior chunks
        of those source_ids are deleted and the new chunks inserted in one
        transaction. On any error nothing changes.

        Returns the chunk IDs that existed for the replaced sources before
        the call (callers use them to clean graph evidence references).
        """
        source_ids = list(dict.fromkeys([*(c.source_id for c in chunks), *replaced_source_ids]))
        rows = [_chunk_to_row(chunk) for chunk in chunks]
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE;")
                old_ids: list[str] = []
                for sid in source_ids:
                    old_ids.extend(
                        r[0]
                        for r in self._conn.execute(
                            "SELECT chunk_id FROM chunks WHERE source_id = ?;", (sid,)
                        ).fetchall()
                    )
                    self._conn.execute("DELETE FROM chunks WHERE source_id = ?;", (sid,))
                self._conn.executemany(_INSERT, rows)
                if transaction_id is not None:
                    self._conn.execute("INSERT INTO storage_transactions (transaction_id) VALUES (?);", (transaction_id,))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return old_ids

    def clear_all(self, *, transaction_id: str | None = None) -> int:
        """Delete every chunk in the store. Returns the number of rows deleted."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE;")
                cur = self._conn.execute("DELETE FROM chunks;")
                if transaction_id is not None:
                    self._conn.execute("INSERT INTO storage_transactions (transaction_id) VALUES (?);", (transaction_id,))
                self._conn.commit()
                return cur.rowcount
            except BaseException:
                self._conn.rollback()
                raise

    @property
    def path(self) -> Path:
        return self._path

    def transaction_committed(self, transaction_id: str) -> bool:
        """The marker is committed atomically with the corresponding chunk update."""
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM storage_transactions WHERE transaction_id = ?;", (transaction_id,)
            ).fetchone() is not None

    def forget_transaction(self, transaction_id: str) -> None:
        """Only safe after the associated journal has been durably removed."""
        with self._lock:
            self._conn.execute("DELETE FROM storage_transactions WHERE transaction_id = ?;", (transaction_id,))
            self._conn.commit()

    def prune_orphaned_transactions(self) -> int:
        """Remove markers only after the recovery manager resolved every journal.

        The single-writer mutation lock must exclude an active publication.
        Avoid an unnecessary commit when there are no orphaned markers.
        """
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM storage_transactions;").fetchone()[0]
            if not count:
                return 0
            try:
                self._conn.execute("DELETE FROM storage_transactions;")
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return count

    def delete_source(self, source_id: str) -> int:
        """
        Remove all chunks belonging to source_id.
        Returns the number of rows deleted.
        Useful for cleanly re-ingesting a document.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE source_id = ?;", (source_id,)
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return the Chunk with the given id, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?;", (chunk_id,)
            ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        """Fetch distinct IDs in their requested order, omitting missing chunks.

        Bound each IN clause below SQLite's portable parameter limit rather
        than issuing a query for every graph node or vector search hit.
        """
        ids = list(dict.fromkeys(chunk_ids))
        rows_by_id: dict[str, sqlite3.Row] = {}
        with self._lock:
            for offset in range(0, len(ids), 400):
                batch = ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders});", batch
                ).fetchall()
                rows_by_id.update((row["chunk_id"], row) for row in rows)
        return [_row_to_chunk(rows_by_id[cid]) for cid in ids if cid in rows_by_id]

    def get_chunk_sources(self, chunk_ids: Iterable[str]) -> dict[str, str]:
        """Map distinct existing chunk IDs to their stored source_id, in request order.

        Reads only the two metadata columns in bounded IN batches, so grouping
        graph nodes by document never loads passage text. Empty input runs no SQL.
        """
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        sources: dict[str, str] = {}
        with self._lock:
            for offset in range(0, len(ids), 400):
                batch = ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"SELECT chunk_id, source_id FROM chunks WHERE chunk_id IN ({placeholders});", batch
                ).fetchall()
                sources.update((row["chunk_id"], row["source_id"]) for row in rows)
        return {cid: sources[cid] for cid in ids if cid in sources}

    def get_chunks_by_source(self, source_id: str) -> list[Chunk]:
        """Return all Chunks belonging to source_id, in insertion order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE source_id = ?;", (source_id,)
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_first_chunk_by_source(self, source_id: str) -> Chunk | None:
        """Return one Chunk for source_id (cheap provenance lookup), or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE source_id = ? LIMIT 1;", (source_id,)
            ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_all_chunks(self) -> list[Chunk]:
        """Return every Chunk in the store."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM chunks;").fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count(self) -> int:
        """Return total number of chunks in the store."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()
