"""
Tests for backend/ingestion/slide_processor.py

Strategy:
  - All tests use a real synthetic PDF built with pymupdf (no external files needed).
  - The captioner is always mocked — no Ollama/Anthropic calls during tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest

from backend.ingestion.slide_processor import (
    _extract_title,
    _process_page,
    _render_page,
    process_slides,
)
from backend.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path: Path, pages: list[str]) -> Path:
    """Create a minimal PDF where each string in pages is one page's text."""
    pdf_path = tmp_path / "slides.pdf"
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        if text:
            page.insert_text((50, 100), text, fontsize=14)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def stub_captioner() -> MagicMock:
    captioner = MagicMock()
    captioner.caption.return_value = "A slide showing enzyme kinetics."
    return captioner


@pytest.fixture
def base_config() -> dict:
    return {
        "captioning": {
            "provider": "ollama",
            "ollama_model": "llama3.2-vision",
            "ollama_base_url": "http://localhost:11434",
            "anthropic_model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
        },
        "slide_processor": {"image_dpi": 72},  # low DPI keeps tests fast
    }


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_returns_first_non_empty_line(self):
        assert _extract_title("  \nIntroduction\nSome body text") == "Introduction"

    def test_returns_none_for_empty_text(self):
        assert _extract_title("") is None

    def test_returns_none_for_whitespace_only(self):
        assert _extract_title("   \n\n  ") is None

    def test_strips_whitespace(self):
        assert _extract_title("  My Title  \nBody") == "My Title"


# ---------------------------------------------------------------------------
# _render_page
# ---------------------------------------------------------------------------

class TestRenderPage:
    def test_creates_jpeg_file(self, tmp_path):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Hello", fontsize=14)

        out = tmp_path / "slide.jpg"
        _render_page(page, out, dpi=72)
        doc.close()

        assert out.exists()
        assert out.stat().st_size > 0

    def test_higher_dpi_produces_larger_file(self, tmp_path):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Hello", fontsize=14)

        lo = tmp_path / "lo.jpg"
        hi = tmp_path / "hi.jpg"
        _render_page(page, lo, dpi=72)
        _render_page(page, hi, dpi=150)
        doc.close()

        assert hi.stat().st_size > lo.stat().st_size


# ---------------------------------------------------------------------------
# _process_page
# ---------------------------------------------------------------------------

class TestProcessPage:
    def test_returns_chunk(self, tmp_path, stub_captioner):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Enzyme kinetics", fontsize=14)

        chunk = _process_page(page, 1, "lecture", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert isinstance(chunk, Chunk)

    def test_modality_is_slide(self, tmp_path, stub_captioner):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Content", fontsize=14)

        chunk = _process_page(page, 1, "src", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert chunk.modality == "slide"

    def test_chunk_id_format(self, tmp_path, stub_captioner):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Content", fontsize=14)

        chunk = _process_page(page, 3, "lecture_01", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert chunk.id == "lecture_01_slide_3"

    def test_caption_injected_in_text(self, tmp_path, stub_captioner):
        stub_captioner.caption.return_value = "Diagram of ATP synthesis."
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Mitochondria text", fontsize=14)

        chunk = _process_page(page, 1, "src", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert "[Visual: Diagram of ATP synthesis.]" in chunk.text

    def test_metadata_keys_present(self, tmp_path, stub_captioner):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Title\nBody text", fontsize=14)

        chunk = _process_page(page, 2, "src", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert "page_number" in chunk.metadata
        assert "slide_image_path" in chunk.metadata
        assert "slide_title" in chunk.metadata
        assert chunk.metadata["page_number"] == 2

    def test_slide_title_is_first_line(self, tmp_path, stub_captioner):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "My Title\nSome body", fontsize=14)

        chunk = _process_page(page, 1, "src", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert chunk.metadata["slide_title"] == "My Title"

    def test_captioner_failure_gives_text_only_chunk(self, tmp_path):
        failing_captioner = MagicMock()
        failing_captioner.caption.side_effect = RuntimeError("Ollama down")

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Some slide content", fontsize=14)

        chunk = _process_page(page, 1, "src", tmp_path, failing_captioner, dpi=72)
        doc.close()

        assert chunk is not None
        assert "[Visual:" not in chunk.text

    def test_empty_page_returns_none(self, tmp_path, stub_captioner):
        stub_captioner.caption.return_value = ""  # captioner also returns nothing
        doc = fitz.open()
        page = doc.new_page()  # blank page, no text

        chunk = _process_page(page, 1, "src", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert chunk is None

    def test_image_file_is_created(self, tmp_path, stub_captioner):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Content", fontsize=14)

        _process_page(page, 5, "src", tmp_path, stub_captioner, dpi=72)
        doc.close()

        assert (tmp_path / "slide_0005.jpg").exists()


# ---------------------------------------------------------------------------
# process_slides (integration over the full PDF)
# ---------------------------------------------------------------------------

class TestProcessSlides:
    def test_returns_one_chunk_per_page(self, tmp_path, base_config, stub_captioner):
        pdf = _make_pdf(tmp_path, ["Slide one text", "Slide two text", "Slide three text"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.slide_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_slides(pdf, base_config)

        assert len(chunks) == 3

    def test_all_chunks_are_slide_modality(self, tmp_path, base_config, stub_captioner):
        pdf = _make_pdf(tmp_path, ["Page A", "Page B"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.slide_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_slides(pdf, base_config)

        assert all(c.modality == "slide" for c in chunks)

    def test_chunk_ids_are_unique(self, tmp_path, base_config, stub_captioner):
        pdf = _make_pdf(tmp_path, ["A", "B", "C"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.slide_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_slides(pdf, base_config)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_captioner_called_once_per_page(self, tmp_path, base_config, stub_captioner):
        pdf = _make_pdf(tmp_path, ["Page 1", "Page 2", "Page 3"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.slide_processor.build_captioner",
                lambda _: stub_captioner,
            )
            process_slides(pdf, base_config)

        assert stub_captioner.caption.call_count == 3

    def test_file_not_found_raises(self, tmp_path, base_config):
        with pytest.raises(FileNotFoundError):
            process_slides(tmp_path / "nonexistent.pdf", base_config)

    def test_unsupported_extension_raises(self, tmp_path, base_config):
        bad_file = tmp_path / "deck.docx"
        bad_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="Unsupported file type"):
            process_slides(bad_file, base_config)

    def test_slide_images_saved_to_disk(self, tmp_path, base_config, stub_captioner):
        pdf = _make_pdf(tmp_path, ["Slide 1", "Slide 2"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.slide_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_slides(pdf, base_config)

        for chunk in chunks:
            assert Path(chunk.metadata["slide_image_path"]).exists()
