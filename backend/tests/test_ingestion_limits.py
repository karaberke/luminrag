"""Reject pathological documents before allocating unbounded pixel buffers."""

from types import SimpleNamespace

import pymupdf
import pytest

from backend.utils.ingestion_limits import (
    MAX_RENDER_PIXELS, check_document_pages, check_image_dimensions, page_render_scale,
)
from backend.ingestion.slide_processor import _render_page


def test_huge_page_is_scaled_before_rendering(tmp_path):
    with pymupdf.open() as document:
        page = document.new_page(width=20000, height=20000)
        output = tmp_path / "huge.jpg"
        _render_page(page, output, 300)
    pixmap = pymupdf.Pixmap(str(output))
    assert pixmap.width * pixmap.height <= MAX_RENDER_PIXELS


@pytest.mark.parametrize("width,height", [(0, 4), (4, 0), (-1, 4), (50000, 50000)])
def test_image_dimensions_are_bounded_before_decode(monkeypatch, width, height):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    with pytest.raises(ValueError):
        check_image_dimensions(width, height)


def test_hosted_document_page_count_is_bounded(monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    class Document:
        def __len__(self): return 501
    with pytest.raises(ValueError):
        check_document_pages(Document())


def test_nonfinite_page_dimensions_rejected():
    page = SimpleNamespace(rect=SimpleNamespace(width=float("inf"), height=100))
    with pytest.raises(ValueError):
        page_render_scale(page, 150)
