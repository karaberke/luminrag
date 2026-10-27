"""
Stage 1 - Multimodal Ingestion: slide_processor.py

Processes a PDF slide deck into a list of Chunk objects.
One chunk per page: text extracted by pymupdf + image caption from the
rendered slide image.

Pipeline per page:
  1. Extract text          (fitz page.get_text())
  2. Render page to JPEG   (fitz page.get_pixmap() at configured DPI)
  3. Caption image         (BaseCaptioner — Ollama or Anthropic)
  4. Assemble Chunk        (modality="slide")

Public API:
    chunks = process_slides(pdf_path, config)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pymupdf as fitz

from backend.ingestion.captioners import BaseCaptioner, build_captioner, safe_caption
from backend.schemas import Chunk
from backend.utils.paths import processed_data_dir
from backend.utils.ingestion_limits import (
    MAX_DOCUMENT_ASSET_BYTES, MAX_DOCUMENT_TEXT_CHARS,
    check_document_pages, check_pdf_file, page_render_scale,
)
from backend.utils.processes import publish_parser_assets, run_parser

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf"}


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def _render_page(page: fitz.Page, out_path: Path, dpi: int) -> None:
    """Render a fitz page to a JPEG image at the given DPI."""
    scale = page_render_scale(page, dpi)
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    pixmap.save(str(out_path), output="jpeg")


def _extract_title(text: str) -> str | None:
    """Return the first non-empty line of a page's text as its slide title."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _process_page(
    page: fitz.Page,
    page_number: int,
    source_id: str,
    slide_dir: Path,
    captioner: BaseCaptioner,
    dpi: int,
) -> Chunk | None:
    """
    Convert one PDF page into a Chunk.
    Returns None if the page has no text and captioning fails.
    """
    text = page.get_text().strip()

    # Render and caption
    img_path = slide_dir / f"slide_{page_number:04d}.jpg"
    _render_page(page, img_path, dpi)

    return _assemble_slide(text, img_path, page_number, source_id, captioner)


def _assemble_slide(text: str, img_path: Path, page_number: int, source_id: str,
                    captioner: BaseCaptioner) -> Chunk | None:

    caption = safe_caption(captioner, img_path, f"{source_id} page {page_number}")

    full_text = f"[Visual: {caption}] {text}" if caption else text

    if not full_text.strip():
        logger.debug(f"[{source_id}] Skipping empty page {page_number}")
        return None

    return Chunk(
        id=f"{source_id}_slide_{page_number}",
        text=full_text.strip(),
        source_id=source_id,
        modality="slide",
        metadata={
            "page_number": page_number,
            "slide_image_path": str(img_path),
            "slide_title": _extract_title(text),
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_slides(pdf_path: str | Path, config: dict) -> list[Chunk]:
    """
    Process a PDF slide deck into a list of Chunk objects.

    Args:
        pdf_path: Path to the source PDF file.
        config:   Parsed contents of config/llm.yaml.

    Returns:
        Ordered list of Chunk objects with modality="slide", one per page.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{pdf_path.suffix}'. "
            f"slide_processor only handles: {', '.join(_SUPPORTED_EXTENSIONS)}"
        )

    source_id = pdf_path.stem
    dpi = config.get("slide_processor", {}).get("image_dpi", 150)
    check_pdf_file(pdf_path)
    slide_dir = processed_data_dir(config) / "slides" / source_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    captioner = build_captioner(config)
    chunks: list[Chunk] = []
    text_chars = 0
    with tempfile.TemporaryDirectory(prefix="slides-", dir=slide_dir) as temporary:
        pages = run_parser("slides", pdf_path, options={"image_dir": temporary, "image_dpi": dpi})
        mapping = publish_parser_assets([page["image_path"] for page in pages], Path(temporary), slide_dir,
                                        max_bytes=MAX_DOCUMENT_ASSET_BYTES)
        for page in pages:
            chunk = _assemble_slide(page["text"], Path(mapping[page["image_path"]]), page["page_number"], source_id, captioner)
            if chunk:
                text_chars += len(chunk.text)
                if text_chars > MAX_DOCUMENT_TEXT_CHARS:
                    raise ValueError("Extracted document text exceeds the processing limit.")
                chunks.append(chunk)
    logger.info(f"[{source_id}] Done — {len(chunks)} slide chunks produced")
    return chunks


def _extract_slides_local(pdf_path: Path, options: dict) -> list[dict]:
    """Worker-only page extraction/rendering; provider calls stay in parent."""
    check_pdf_file(pdf_path)
    pages = []
    asset_bytes = text_chars = 0
    slide_dir = Path(options["image_dir"])
    with fitz.open(str(pdf_path), filetype="pdf") as doc:
        if not doc.is_pdf:
            raise ValueError("The uploaded file is not a PDF document.")
        check_document_pages(doc)
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            text_chars += len(text)
            if text_chars > MAX_DOCUMENT_TEXT_CHARS:
                raise ValueError("Extracted document text exceeds the processing limit.")
            image_path = slide_dir / f"slide_{page_number:04d}.jpg"
            _render_page(page, image_path, options.get("image_dpi", 150))
            asset_bytes += image_path.stat().st_size
            if asset_bytes > MAX_DOCUMENT_ASSET_BYTES:
                raise ValueError("Rendered document images exceed the processing limit.")
            pages.append({"page_number": page_number, "text": text, "image_path": str(image_path)})
    return pages
