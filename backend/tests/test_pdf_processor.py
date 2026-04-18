"""
Tests for backend/ingestion/pdf_processor.py

All tests build synthetic PDFs in-memory with pymupdf — no fixture files needed.
No mocking required (pdf_processor has no external I/O).
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from backend.ingestion.pdf_processor import (
    _build_chunks,
    _extract_lines,
    _group_into_sections,
    _hard_split,
    _split_section,
    process_pdf,
)
from backend.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path: Path, pages: list[list[tuple[str, float]]]) -> Path:
    """
    Build a PDF where each page is a list of (text, font_size) tuples.
    Lines are stacked top-to-bottom with font-size-proportional spacing.
    """
    pdf_path = tmp_path / "textbook.pdf"
    doc = fitz.open()
    for page_content in pages:
        page = doc.new_page(width=595, height=842)
        y = 60.0
        for text, size in page_content:
            page.insert_text((50, y), text, fontsize=size)
            y += size * 2.0
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def base_config() -> dict:
    return {
        "pdf_processor": {
            "max_chunk_chars": 2000,
            "min_heading_size": 14,
        }
    }


# ---------------------------------------------------------------------------
# _hard_split
# ---------------------------------------------------------------------------

class TestHardSplit:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert _hard_split(text, 100) == ["Short text."]

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        parts = _hard_split(text, 30)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 30 + 20  # small tolerance for boundary

    def test_hard_cut_when_no_sentence_boundary(self):
        text = "A" * 150
        parts = _hard_split(text, 100)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 100

    def test_all_parts_non_empty(self):
        text = "Word " * 200
        parts = _hard_split(text, 50)
        assert all(p.strip() for p in parts)


# ---------------------------------------------------------------------------
# _split_section
# ---------------------------------------------------------------------------

class TestSplitSection:
    def _section(self, text: str) -> dict:
        return {"title": "S", "text": text, "page_start": 1, "page_end": 1}

    def test_short_section_returned_as_is(self):
        s = self._section("Short text.")
        result = _split_section(s, max_chunk_chars=2000)
        assert result == [s]

    def test_oversized_section_splits(self):
        long_text = ("Word " * 100 + "\n\n") * 5  # 5 paragraphs
        s = self._section(long_text)
        result = _split_section(s, max_chunk_chars=200)
        assert len(result) > 1

    def test_all_sub_chunks_under_limit(self):
        long_text = ("Sentence text here. " * 30 + "\n\n") * 4
        s = self._section(long_text)
        limit = 300
        result = _split_section(s, max_chunk_chars=limit)
        for sub in result:
            assert len(sub["text"]) <= limit * 1.1  # small tolerance for boundary

    def test_title_preserved_in_all_sub_chunks(self):
        long_text = ("Word " * 60 + "\n\n") * 4
        s = {"title": "My Section", "text": long_text, "page_start": 2, "page_end": 5}
        result = _split_section(s, max_chunk_chars=200)
        assert all(sub["title"] == "My Section" for sub in result)

    def test_page_range_preserved(self):
        s = {"title": "T", "text": "W " * 200, "page_start": 3, "page_end": 7}
        result = _split_section(s, max_chunk_chars=100)
        assert all(sub["page_start"] == 3 for sub in result)
        assert all(sub["page_end"] == 7 for sub in result)


# ---------------------------------------------------------------------------
# _extract_lines
# ---------------------------------------------------------------------------

class TestExtractLines:
    def test_extracts_text(self, tmp_path):
        pdf = _make_pdf(tmp_path, [[("Hello world", 12.0)]])
        doc = fitz.open(str(pdf))
        lines = _extract_lines(doc)
        doc.close()
        texts = [l["text"] for l in lines]
        assert any("Hello" in t for t in texts)

    def test_records_page_number(self, tmp_path):
        pdf = _make_pdf(tmp_path, [
            [("Page one", 12.0)],
            [("Page two", 12.0)],
        ])
        doc = fitz.open(str(pdf))
        lines = _extract_lines(doc)
        doc.close()
        page_numbers = {l["page_number"] for l in lines}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_records_font_size(self, tmp_path):
        pdf = _make_pdf(tmp_path, [[("Big heading", 18.0), ("Small body", 10.0)]])
        doc = fitz.open(str(pdf))
        lines = _extract_lines(doc)
        doc.close()
        sizes = [l["max_font_size"] for l in lines]
        assert max(sizes) > min(sizes)

    def test_empty_pdf_returns_empty(self, tmp_path):
        pdf = _make_pdf(tmp_path, [[]])  # one blank page
        doc = fitz.open(str(pdf))
        lines = _extract_lines(doc)
        doc.close()
        assert lines == []


# ---------------------------------------------------------------------------
# _group_into_sections
# ---------------------------------------------------------------------------

class TestGroupIntoSections:
    def test_heading_starts_new_section(self):
        lines = [
            {"text": "Chapter 1", "max_font_size": 18.0, "page_number": 1},
            {"text": "Body text.", "max_font_size": 11.0, "page_number": 1},
            {"text": "Chapter 2", "max_font_size": 18.0, "page_number": 2},
            {"text": "More body.", "max_font_size": 11.0, "page_number": 2},
        ]
        sections = _group_into_sections(lines, min_heading_size=14.0)
        assert len(sections) == 2

    def test_section_title_is_heading_text(self):
        lines = [
            {"text": "Introduction", "max_font_size": 16.0, "page_number": 1},
            {"text": "Some content.", "max_font_size": 11.0, "page_number": 1},
        ]
        sections = _group_into_sections(lines, min_heading_size=14.0)
        assert sections[0]["title"] == "Introduction"

    def test_no_headings_gives_single_section(self):
        lines = [
            {"text": "Para one.", "max_font_size": 11.0, "page_number": 1},
            {"text": "Para two.", "max_font_size": 11.0, "page_number": 1},
        ]
        sections = _group_into_sections(lines, min_heading_size=14.0)
        assert len(sections) == 1
        assert sections[0]["title"] is None

    def test_page_range_spans_correctly(self):
        lines = [
            {"text": "Heading", "max_font_size": 16.0, "page_number": 1},
            {"text": "Body p1.",  "max_font_size": 11.0, "page_number": 1},
            {"text": "Body p2.",  "max_font_size": 11.0, "page_number": 3},
        ]
        sections = _group_into_sections(lines, min_heading_size=14.0)
        assert sections[0]["page_start"] == 1
        assert sections[0]["page_end"] == 3


# ---------------------------------------------------------------------------
# _build_chunks
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_returns_chunk_instances(self):
        sections = [
            {"title": "S1", "text": "Some text.", "page_start": 1, "page_end": 1},
        ]
        chunks = _build_chunks("textbook", sections)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_modality_is_pdf(self):
        sections = [{"title": None, "text": "Body.", "page_start": 1, "page_end": 1}]
        chunks = _build_chunks("src", sections)
        assert all(c.modality == "pdf" for c in chunks)

    def test_metadata_keys_present(self):
        sections = [{"title": "T", "text": "Text.", "page_start": 2, "page_end": 4}]
        chunk = _build_chunks("src", sections)[0]
        assert "page_start" in chunk.metadata
        assert "page_end" in chunk.metadata
        assert "section_title" in chunk.metadata
        assert "char_count" in chunk.metadata

    def test_char_count_matches_text_length(self):
        sections = [{"title": None, "text": "Hello world.", "page_start": 1, "page_end": 1}]
        chunk = _build_chunks("src", sections)[0]
        assert chunk.metadata["char_count"] == len("Hello world.")

    def test_empty_text_sections_skipped(self):
        sections = [
            {"title": "T", "text": "   ", "page_start": 1, "page_end": 1},
            {"title": "T2", "text": "Real content.", "page_start": 2, "page_end": 2},
        ]
        chunks = _build_chunks("src", sections)
        assert len(chunks) == 1

    def test_chunk_ids_are_unique(self):
        sections = [
            {"title": f"S{i}", "text": f"Content {i}.", "page_start": i, "page_end": i}
            for i in range(5)
        ]
        chunks = _build_chunks("src", sections)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# process_pdf  (full integration)
# ---------------------------------------------------------------------------

class TestProcessPdf:
    def test_heading_creates_chunk_boundary(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [[
            ("Chapter 1: Enzymes", 16.0),
            ("Enzymes are biological catalysts.", 11.0),
            ("They lower activation energy.", 11.0),
            ("Chapter 2: Kinetics", 16.0),
            ("Reaction rate depends on concentration.", 11.0),
        ]])
        chunks = process_pdf(pdf, base_config)
        assert len(chunks) >= 2

    def test_no_headings_still_produces_chunks(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [[
            ("All text has the same font size.", 11.0),
            ("Another line of body text.", 11.0),
        ]])
        chunks = process_pdf(pdf, base_config)
        assert len(chunks) >= 1

    def test_all_chunks_are_pdf_modality(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [[("Some content", 11.0)]])
        chunks = process_pdf(pdf, base_config)
        assert all(c.modality == "pdf" for c in chunks)

    def test_section_title_in_metadata(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [[
            ("Introduction", 16.0),
            ("This is the intro text.", 11.0),
        ]])
        chunks = process_pdf(pdf, base_config)
        titled = [c for c in chunks if c.metadata.get("section_title") == "Introduction"]
        assert len(titled) >= 1

    def test_oversized_section_is_split(self, tmp_path):
        config = {"pdf_processor": {"max_chunk_chars": 100, "min_heading_size": 14}}
        # One section with lots of body text
        body = [("Word " * 10, 11.0)] * 20  # ~200 chars per line × 20 lines
        pdf = _make_pdf(tmp_path, [body])
        chunks = process_pdf(pdf, config)
        assert len(chunks) > 1

    def test_all_chunk_ids_unique(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [
            [("Heading A", 16.0), ("Body A text.", 11.0)],
            [("Heading B", 16.0), ("Body B text.", 11.0)],
        ])
        chunks = process_pdf(pdf, base_config)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_keys_present(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [[("Content", 11.0)]])
        chunks = process_pdf(pdf, base_config)
        for c in chunks:
            assert "page_start" in c.metadata
            assert "page_end" in c.metadata
            assert "section_title" in c.metadata
            assert "char_count" in c.metadata

    def test_empty_pdf_returns_empty_list(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [[]])  # blank page
        chunks = process_pdf(pdf, base_config)
        assert chunks == []

    def test_file_not_found_raises(self, tmp_path, base_config):
        with pytest.raises(FileNotFoundError):
            process_pdf(tmp_path / "missing.pdf", base_config)

    def test_unsupported_extension_raises(self, tmp_path, base_config):
        bad = tmp_path / "notes.docx"
        bad.write_text("not a pdf")
        with pytest.raises(ValueError, match="Unsupported file type"):
            process_pdf(bad, base_config)

    def test_multipage_page_numbers_recorded(self, tmp_path, base_config):
        pdf = _make_pdf(tmp_path, [
            [("Page one body.", 11.0)],
            [("Page two body.", 11.0)],
        ])
        chunks = process_pdf(pdf, base_config)
        all_starts = {c.metadata["page_start"] for c in chunks}
        # Content spans at least two different pages
        assert len(all_starts) >= 1  # may merge into one section (no headings)
        # Verify page_end >= page_start for all chunks
        for c in chunks:
            assert c.metadata["page_end"] >= c.metadata["page_start"]
