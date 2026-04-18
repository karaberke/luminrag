"""Tests for backend/schemas.py — validates all four core data models."""

import pytest
from pydantic import ValidationError

from backend.schemas import Chunk, GraphTriple, ReflectionVerdict, RetrievalResult


class TestChunk:
    def test_valid_video_chunk(self):
        chunk = Chunk(
            text="Enzymes lower activation energy.",
            source_id="lecture_01",
            modality="video",
            metadata={"start_time": 12.5, "end_time": 30.0, "keyframe_path": None},
        )
        assert chunk.modality == "video"
        assert chunk.id  # auto-generated UUID

    def test_valid_pdf_chunk(self):
        chunk = Chunk(
            text="Chapter 3: Thermodynamics",
            source_id="textbook_ch3",
            modality="pdf",
            metadata={"page_number": 42},
        )
        assert chunk.modality == "pdf"

    def test_id_is_unique(self):
        a = Chunk(text="a", source_id="s", modality="video")
        b = Chunk(text="b", source_id="s", modality="video")
        assert a.id != b.id

    def test_explicit_id_is_preserved(self):
        chunk = Chunk(id="my-id-123", text="x", source_id="s", modality="slide")
        assert chunk.id == "my-id-123"

    def test_invalid_modality(self):
        with pytest.raises(ValidationError):
            Chunk(text="x", source_id="s", modality="unknown")

    def test_metadata_defaults_to_empty_dict(self):
        chunk = Chunk(text="x", source_id="s", modality="audio")
        assert chunk.metadata == {}

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            Chunk(source_id="s", modality="video")  # text is required


class TestGraphTriple:
    def test_valid_triple(self):
        triple = GraphTriple(
            head="Enzyme",
            relation="CAUSES",
            tail="ReactionRate",
            source_chunk_ids=["lecture_01_chunk_0", "lecture_01_chunk_3"],
        )
        assert triple.relation == "CAUSES"
        assert len(triple.source_chunk_ids) == 2

    def test_empty_source_chunk_ids(self):
        triple = GraphTriple(head="A", relation="PART_OF", tail="B", source_chunk_ids=[])
        assert triple.source_chunk_ids == []

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            GraphTriple(head="A", relation="PART_OF", source_chunk_ids=[])  # tail missing


class TestRetrievalResult:
    def test_valid_dense_result(self):
        chunk = Chunk(text="context", source_id="src", modality="pdf")
        result = RetrievalResult(chunks=[chunk], subgraph=[], routing_mode="dense")
        assert result.routing_mode == "dense"
        assert len(result.chunks) == 1

    def test_valid_graph_result(self):
        triple = GraphTriple(head="A", relation="EXPLAINS", tail="B", source_chunk_ids=["c1"])
        result = RetrievalResult(chunks=[], subgraph=[triple], routing_mode="graph")
        assert result.subgraph[0].relation == "EXPLAINS"

    def test_invalid_routing_mode(self):
        with pytest.raises(ValidationError):
            RetrievalResult(chunks=[], subgraph=[], routing_mode="hybrid")


class TestReflectionVerdict:
    def test_valid_verdict(self):
        verdict = ReflectionVerdict(
            needs_retrieval=True,
            is_relevant=True,
            is_supported=False,
            is_useful=True,
            reasoning="Answer partially supported but missing enzyme kinetics detail.",
        )
        assert verdict.is_supported is False

    def test_all_false(self):
        verdict = ReflectionVerdict(
            needs_retrieval=False,
            is_relevant=False,
            is_supported=False,
            is_useful=False,
            reasoning="Off-topic question.",
        )
        assert verdict.needs_retrieval is False

    def test_missing_reasoning(self):
        with pytest.raises(ValidationError):
            ReflectionVerdict(
                needs_retrieval=True,
                is_relevant=True,
                is_supported=True,
                is_useful=True,
            )  # reasoning is required
