"""
Tests for backend/db/document_store.py

All tests use tmp_path so the SQLite file is isolated and cleaned up
automatically. No mocking needed — sqlite3 is stdlib with no I/O side effects
worth faking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    id: str = "src_chunk_0",
    source_id: str = "lecture_01",
    modality: str = "video",
    text: str = "Enzymes lower activation energy.",
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        id=id,
        source_id=source_id,
        modality=modality,
        text=text,
        metadata=metadata or {"start_time": 0.0, "end_time": 5.0},
    )


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    with DocumentStore(tmp_path / "test.db") as s:
        yield s


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "sub" / "chunks.db"
        with DocumentStore(db_path):
            pass
        assert db_path.exists()

    def test_empty_store_count_is_zero(self, store):
        assert store.count() == 0


# ---------------------------------------------------------------------------
# save_chunks / get_chunk
# ---------------------------------------------------------------------------

class TestSaveAndRetrieve:
    def test_save_and_retrieve_by_id(self, store):
        c = _chunk()
        store.save_chunks([c])
        result = store.get_chunk(c.id)
        assert result is not None
        assert result.id == c.id
        assert result.text == c.text

    def test_all_fields_round_trip(self, store):
        c = _chunk(
            id="vid_chunk_7",
            source_id="biochem_lecture",
            modality="video",
            text="Michaelis-Menten kinetics.",
            metadata={"start_time": 12.5, "end_time": 30.0, "keyframe_path": None},
        )
        store.save_chunks([c])
        result = store.get_chunk(c.id)
        assert result.id == c.id
        assert result.source_id == c.source_id
        assert result.modality == c.modality
        assert result.text == c.text
        assert result.metadata == c.metadata

    def test_metadata_round_trips_correctly(self, store):
        meta = {"page_number": 42, "slide_title": "Enzyme Kinetics", "nested": {"a": 1}}
        c = _chunk(metadata=meta)
        store.save_chunks([c])
        assert store.get_chunk(c.id).metadata == meta

    def test_get_nonexistent_chunk_returns_none(self, store):
        assert store.get_chunk("does_not_exist") is None

    def test_upsert_does_not_duplicate(self, store):
        c = _chunk()
        store.save_chunks([c])
        store.save_chunks([c])
        assert store.count() == 1

    def test_upsert_updates_text(self, store):
        c = _chunk(text="Original text.")
        store.save_chunks([c])
        updated = _chunk(text="Updated text.")
        store.save_chunks([updated])
        assert store.get_chunk(c.id).text == "Updated text."

    def test_save_empty_list_is_noop(self, store):
        store.save_chunks([])
        assert store.count() == 0


# ---------------------------------------------------------------------------
# get_all_chunks
# ---------------------------------------------------------------------------

class TestGetAllChunks:
    def test_returns_all_saved_chunks(self, store):
        chunks = [_chunk(id=f"src_chunk_{i}", text=f"Text {i}") for i in range(5)]
        store.save_chunks(chunks)
        result = store.get_all_chunks()
        assert len(result) == 5

    def test_empty_store_returns_empty_list(self, store):
        assert store.get_all_chunks() == []

    def test_all_returned_are_chunk_instances(self, store):
        store.save_chunks([_chunk(id=f"c{i}") for i in range(3)])
        assert all(isinstance(c, Chunk) for c in store.get_all_chunks())


# ---------------------------------------------------------------------------
# get_chunks_by_source
# ---------------------------------------------------------------------------

class TestGetChunksBySource:
    def test_returns_only_matching_source(self, store):
        store.save_chunks([
            _chunk(id="a_0", source_id="doc_a"),
            _chunk(id="a_1", source_id="doc_a"),
            _chunk(id="b_0", source_id="doc_b"),
        ])
        result = store.get_chunks_by_source("doc_a")
        assert len(result) == 2
        assert all(c.source_id == "doc_a" for c in result)

    def test_unknown_source_returns_empty(self, store):
        store.save_chunks([_chunk(source_id="doc_x")])
        assert store.get_chunks_by_source("doc_y") == []

    def test_does_not_return_other_sources(self, store):
        store.save_chunks([
            _chunk(id="x_0", source_id="x"),
            _chunk(id="y_0", source_id="y"),
        ])
        ids = [c.id for c in store.get_chunks_by_source("x")]
        assert "y_0" not in ids


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------

class TestDeleteSource:
    def test_removes_correct_chunks(self, store):
        store.save_chunks([
            _chunk(id="a_0", source_id="doc_a"),
            _chunk(id="b_0", source_id="doc_b"),
        ])
        store.delete_source("doc_a")
        assert store.get_chunk("a_0") is None
        assert store.get_chunk("b_0") is not None

    def test_returns_deleted_row_count(self, store):
        store.save_chunks([
            _chunk(id="a_0", source_id="doc_a"),
            _chunk(id="a_1", source_id="doc_a"),
            _chunk(id="b_0", source_id="doc_b"),
        ])
        deleted = store.delete_source("doc_a")
        assert deleted == 2

    def test_delete_nonexistent_source_returns_zero(self, store):
        assert store.delete_source("ghost") == 0

    def test_delete_then_resave_works(self, store):
        c = _chunk()
        store.save_chunks([c])
        store.delete_source(c.source_id)
        store.save_chunks([c])
        assert store.count() == 1


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_matches_saved(self, store):
        store.save_chunks([_chunk(id=f"c{i}") for i in range(7)])
        assert store.count() == 7

    def test_count_after_delete(self, store):
        store.save_chunks([
            _chunk(id="a_0", source_id="a"),
            _chunk(id="b_0", source_id="b"),
            _chunk(id="b_1", source_id="b"),
        ])
        store.delete_source("b")
        assert store.count() == 1


# ---------------------------------------------------------------------------
# Mixed modalities
# ---------------------------------------------------------------------------

class TestMixedModalities:
    def test_all_modalities_stored_correctly(self, store):
        chunks = [
            _chunk(id="v0", modality="video"),
            _chunk(id="s0", modality="slide"),
            _chunk(id="p0", modality="pdf"),
        ]
        store.save_chunks(chunks)
        assert store.get_chunk("v0").modality == "video"
        assert store.get_chunk("s0").modality == "slide"
        assert store.get_chunk("p0").modality == "pdf"

    def test_get_all_preserves_modality(self, store):
        chunks = [
            _chunk(id="v0", modality="video"),
            _chunk(id="s0", modality="slide"),
        ]
        store.save_chunks(chunks)
        result = {c.id: c for c in store.get_all_chunks()}
        assert result["v0"].modality == "video"
        assert result["s0"].modality == "slide"
