"""SQLite round trips and source-scoped editing using real isolated databases."""
from pathlib import Path

import pytest

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk


@pytest.fixture
def store(tmp_path: Path):
    with DocumentStore(tmp_path / "test.db") as value:
        yield value


def _chunk(identifier="source-0", source="lecture", text="Enzymes lower activation energy.", modality="video", metadata=None):
    return Chunk(id=identifier, source_id=source, text=text, modality=modality,
                 metadata=metadata if metadata is not None else {"start_time": 12.5, "end_time": 30.0, "keyframe_path": None})


def test_new_store_creates_parent_directory_and_empty_operations_are_noops(tmp_path):
    path = tmp_path / "nested" / "chunks.db"
    with DocumentStore(path) as store:
        assert path.exists() and store.count() == 0
        store.save_chunks([])
        assert store.get_all_chunks() == []
        assert store.get_chunk("missing") is None
        assert store.get_chunks_by_source("missing") == []
        assert store.delete_source("missing") == 0
        assert store.count() == 0


def test_chunk_round_trip_preserves_all_fields_and_mixed_modalities(store):
    chunks = [
        _chunk("video-0", modality="video"),
        _chunk("slide-0", modality="slide", metadata={"slide_title": "Enzyme Kinetics", "nested": {"a": 1}}),
        _chunk("pdf-0", modality="pdf", metadata={"page_number": 42}),
    ]
    store.save_chunks(chunks)
    assert store.count() == len(chunks)
    assert {chunk.id: chunk.model_dump() for chunk in store.get_all_chunks()} == {chunk.id: chunk.model_dump() for chunk in chunks}
    for expected in chunks:
        assert store.get_chunk(expected.id).model_dump() == expected.model_dump()


def test_upsert_replaces_same_id_without_duplicating_or_touching_other_chunks(store):
    original, other = _chunk(text="Original text."), _chunk("other", source="other")
    store.save_chunks([original, other])
    store.save_chunks([original])
    assert store.count() == 2
    updated = original.model_copy(update={"text": "Updated text."})
    store.save_chunks([updated])
    assert store.count() == 2
    assert store.get_chunk(original.id).text == "Updated text."
    assert store.get_chunk(other.id).model_dump() == other.model_dump()


def test_source_filtering_and_deletion_preserve_other_sources_and_allow_resave(store):
    chunks = [_chunk("a-0", "a"), _chunk("a-1", "a"), _chunk("b-0", "b")]
    store.save_chunks(chunks)
    assert {chunk.id for chunk in store.get_chunks_by_source("a")} == {"a-0", "a-1"}
    assert store.get_chunks_by_source("missing") == []
    assert store.delete_source("a") == 2
    assert store.count() == 1 and store.get_chunks_by_source("a") == []
    assert store.get_chunk("a-0") is None
    assert store.get_chunk("b-0").model_dump() == chunks[2].model_dump()
    store.save_chunks(chunks[:2])
    assert store.count() == 3
    assert {chunk.id for chunk in store.get_chunks_by_source("a")} == {"a-0", "a-1"}
