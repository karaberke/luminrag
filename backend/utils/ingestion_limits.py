"""Shared document budgets, checked before expensive image allocations."""

from __future__ import annotations

import math
from pathlib import Path

from backend.credentials import is_hosted

MAX_DOCUMENT_PAGES = 500
MAX_EXTRACTED_IMAGES = 300
MAX_IMAGE_PIXELS = 16_000_000
MAX_RENDER_PIXELS = 4_000_000
MAX_DOCUMENT_ASSET_BYTES = 100 * 1024 * 1024
MAX_DOCUMENT_TEXT_CHARS = 20_000_000


def check_pdf_file(path: Path) -> None:
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("The uploaded file is not a PDF document.")


def check_document_pages(document) -> None:
    if is_hosted() and len(document) > MAX_DOCUMENT_PAGES:
        raise ValueError(f"Documents may contain at most {MAX_DOCUMENT_PAGES} pages on this server.")


def check_image_dimensions(width: int, height: int) -> None:
    maximum = MAX_IMAGE_PIXELS if is_hosted() else 64_000_000
    if width <= 0 or height <= 0 or width * height > maximum:
        raise ValueError("Image dimensions exceed the processing limit.")


def page_render_scale(page, dpi: int) -> float:
    width, height = page.rect.width, page.rect.height
    if not math.isfinite(width * height) or width <= 0 or height <= 0:
        raise ValueError("Invalid PDF page dimensions.")
    # Scale large pages down before PyMuPDF creates their pixel buffers.
    return min(max(1, min(int(dpi), 600)) / 72, math.sqrt(MAX_RENDER_PIXELS / (width * height)) * 0.999)
