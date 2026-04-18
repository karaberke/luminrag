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
    store.get_chunk(chunk_id)
    store.get_chunks_by_source(source_id)
    store.get_all_chunks()
    store.delete_source(source_id)  -> int (rows deleted)
    store.count()                   -> int
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["chunk_id"],
        source_id=row["source_id"],
        modality=row["modality"],
        text=row["text"],
        metadata=json.loads(row["metadata"]),
    )


class DocumentStore:
    """
    Thin SQLite wrapper for storing and retrieving Chunk objects.

    The database file is created automatically on first use.
    All writes use INSERT OR REPLACE so re-ingesting a source is safe.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")  # safe concurrent reads
        self._conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """
        Persist a list of Chunks. Upserts on chunk_id — calling this again
        with the same chunks after re-ingestion will overwrite, not duplicate.
        """
        rows = [
            (
                chunk.id,
                chunk.source_id,
                chunk.modality,
                chunk.text,
                json.dumps(chunk.metadata),
            )
            for chunk in chunks
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, source_id, modality, text, metadata) "
            "VALUES (?, ?, ?, ?, ?);",
            rows,
        )
        self._conn.commit()

    def clear_all(self) -> int:
        """Delete every chunk in the store. Returns the number of rows deleted."""
        cur = self._conn.execute("DELETE FROM chunks;")
        self._conn.commit()
        return cur.rowcount

    def delete_source(self, source_id: str) -> int:
        """
        Remove all chunks belonging to source_id.
        Returns the number of rows deleted.
        Useful for cleanly re-ingesting a document.
        """
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
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?;", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_source(self, source_id: str) -> list[Chunk]:
        """Return all Chunks belonging to source_id, in insertion order."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE source_id = ?;", (source_id,)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_all_chunks(self) -> list[Chunk]:
        """Return every Chunk in the store."""
        rows = self._conn.execute("SELECT * FROM chunks;").fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count(self) -> int:
        """Return total number of chunks in the store."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()
