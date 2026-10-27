"""
Stage 1 - Multimodal Ingestion: pdf_processor.py

Processes a textbook or lecture-notes PDF into semantically coherent Chunks.
Unlike slide_processor (one chunk per page), this module groups text into
sections using heading detection and then splits oversized sections at
paragraph boundaries.

No captioning — textbook PDFs are text-dominant; per-page images add noise.

Pipeline:
  1. Extract all text lines with font-size metadata  (_extract_lines)
  2. Detect headings and group lines into sections   (_group_into_sections)
  3. Split oversized sections at paragraph breaks    (_split_section)
  4. Assemble Chunk objects                          (_build_chunks)

Public API:
    chunks = process_pdf(pdf_path, config)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pymupdf as fitz

from backend.schemas import Chunk
from backend.utils.paths import processed_data_dir
from backend.utils.ingestion_limits import (
    MAX_DOCUMENT_ASSET_BYTES, MAX_DOCUMENT_TEXT_CHARS, MAX_EXTRACTED_IMAGES,
    check_document_pages, check_image_dimensions, check_pdf_file,
)
from backend.utils.processes import publish_parser_assets, run_parser
from backend.utils.chunking import _hard_split, _split_section

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf"}
_MIN_IMAGE_DIM = 50  # px — skip hairlines and decorative rule images


# ---------------------------------------------------------------------------
# Step 1: Line extraction with font metadata
# ---------------------------------------------------------------------------

def _extract_lines(doc: fitz.Document) -> list[dict]:
    """
    Walk every page → block → line → span and emit one record per line.

    Returns list of:
        {text: str, max_font_size: float, page_number: int}
    """
    lines: list[dict] = []
    text_chars = 0

    for page_num, page in enumerate(doc, start=1):
        # Dictionary extraction otherwise decodes embedded images even though
        # this pass only needs text. Assets have a separately bounded pass.
        page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:   # type 0 = text block
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                text_chars += len(text)
                if text_chars > MAX_DOCUMENT_TEXT_CHARS:
                    raise ValueError("Extracted document text exceeds the processing limit.")
                max_size = max((s["size"] for s in spans), default=0.0)
                lines.append(
                    {"text": text, "max_font_size": max_size, "page_number": page_num}
                )

    return lines


# ---------------------------------------------------------------------------
# Step 2: Section grouping by heading detection
# ---------------------------------------------------------------------------

def _group_into_sections(lines: list[dict], min_heading_size: float) -> list[dict]:
    """
    Split the line stream into sections wherever a heading is detected.

    A line is a heading if its max_font_size >= min_heading_size.

    Returns list of:
        {title: str|None, text: str, page_start: int, page_end: int}
    """
    sections: list[dict] = []
    current_title: str | None = None
    current_lines: list[dict] = []

    def _flush() -> None:
        if not current_lines:
            return
        offset = 0
        spans = []
        for line in current_lines:
            end = offset + len(line["text"])
            spans.append((offset, end, line["page_number"]))
            offset = end + 1  # the joining newline
        sections.append(
            {
                "title": current_title,
                "text": "\n".join(ln["text"] for ln in current_lines),
                "page_start": current_lines[0]["page_number"],
                "page_end": current_lines[-1]["page_number"],
                "_page_spans": spans,
            }
        )

    for line in lines:
        if line["max_font_size"] >= min_heading_size:
            _flush()
            current_title = line["text"]
            current_lines = []
        else:
            current_lines.append(line)

    _flush()
    return sections


# ---------------------------------------------------------------------------
# Step 3: Overflow splitting
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Step 4 (new): Embedded image extraction per page
# ---------------------------------------------------------------------------

def _extract_page_images(
    doc: fitz.Document,
    source_id: str,
    images_dir: Path,
) -> dict[int, list[str]]:
    """
    Extract embedded raster images from every page of *doc*.

    Returns a mapping of ``{1-based page_number: [saved_png_path, ...]}``.
    Images smaller than ``_MIN_IMAGE_DIM`` in either dimension are skipped
    (they are typically decorative rules or icons, not content figures).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    page_images: dict[int, list[str]] = {}
    image_count = 0
    asset_bytes = 0

    for page_num, page in enumerate(doc, start=1):
        paths: list[str] = []
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            width = img_info[2]
            height = img_info[3]
            if width < _MIN_IMAGE_DIM or height < _MIN_IMAGE_DIM:
                continue
            check_image_dimensions(width, height)
            image_count += 1
            if image_count > MAX_EXTRACTED_IMAGES:
                raise ValueError("Document contains too many embedded images.")
            try:
                image_data = doc.extract_image(xref)
            except Exception as exc:
                logger.debug(f"[{source_id}] Could not extract image xref={xref}: {exc}")
                continue
            ext = image_data.get("ext", "png")
            if ext not in {"png", "jpg", "jpeg", "webp"}:
                continue
            asset_bytes += len(image_data["image"])
            if asset_bytes > MAX_DOCUMENT_ASSET_BYTES:
                raise ValueError("Extracted document images exceed the processing limit.")
            out_path = images_dir / f"p{page_num:04d}_img{img_idx}.{ext}"
            out_path.write_bytes(image_data["image"])
            paths.append(str(out_path))
            logger.debug(f"[{source_id}] Extracted image {out_path.name} ({width}×{height})")
        if paths:
            page_images[page_num] = paths

    return page_images


# ---------------------------------------------------------------------------
# Step 5: Chunk assembly
# ---------------------------------------------------------------------------

def _build_chunks(
    source_id: str,
    sections: list[dict],
    page_images: dict[int, list[str]] | None = None,
    source_filename: str | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        text = section["text"].strip()
        if not text:
            continue
        p_start: int = section["page_start"]
        p_end: int = section["page_end"]
        images: list[str] = []
        if page_images:
            for pg in range(p_start, p_end + 1):
                images.extend(page_images.get(pg, []))
        metadata: dict = {
            "page_start": p_start,
            "page_end": p_end,
            "section_title": section["title"],
            "char_count": len(text),
        }
        leading = len(section["text"]) - len(section["text"].lstrip())
        spans = section.get("_page_spans", [])
        if spans:
            metadata["page_spans"] = [
                [max(0, start - leading), min(len(text), end - leading), page]
                for start, end, page in spans
                if end > leading and start < leading + len(text)
            ]
        if images:
            metadata["image_paths"] = images
        if source_filename:
            metadata["original_filename"] = source_filename
        chunks.append(
            Chunk(
                id=f"{source_id}_chunk_{len(chunks)}",
                text=text,
                source_id=source_id,
                modality="pdf",
                metadata=metadata,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str | Path, config: dict) -> list[Chunk]:
    """
    Process a textbook or notes PDF into a list of Chunk objects.

    Args:
        pdf_path: Path to the source PDF file.
        config:   Parsed contents of config/llm.yaml.

    Returns:
        Ordered list of Chunk objects with modality="pdf".
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{pdf_path.suffix}'. "
            f"pdf_processor only handles: {', '.join(_SUPPORTED_EXTENSIONS)}"
        )

    check_pdf_file(pdf_path)
    destination = processed_data_dir(config)
    destination.mkdir(parents=True, exist_ok=True)
    settings = config.get("pdf_processor", {})
    with tempfile.TemporaryDirectory(prefix="pdf-", dir=destination) as temporary:
        # Only scalar parser settings cross this boundary; BYOK and every
        # other configuration section remain in the API process.
        options = {"processed_dir": temporary, "max_chunk_chars": settings.get("max_chunk_chars", 2000),
                   "min_heading_size": settings.get("min_heading_size", 14)}
        chunks = [Chunk.model_validate(item) for item in run_parser("pdf", pdf_path, options=options)]
        images = [image for chunk in chunks for image in chunk.metadata.get("image_paths", [])]
        mapping = publish_parser_assets(images, Path(temporary), destination, max_bytes=MAX_DOCUMENT_ASSET_BYTES)
        for chunk in chunks:
            if "image_paths" in chunk.metadata:
                chunk.metadata["image_paths"] = [mapping[image] for image in chunk.metadata["image_paths"]]
        return chunks


def _process_pdf_local(pdf_path: Path, options: dict) -> list[Chunk]:
    """Worker-only native extraction, including potentially large page trees."""
    check_pdf_file(pdf_path)

    source_id = pdf_path.stem
    max_chunk_chars = options.get("max_chunk_chars", 2000)
    min_heading_size = options.get("min_heading_size", 14)

    with fitz.open(str(pdf_path), filetype="pdf") as doc:
        if not doc.is_pdf:
            raise ValueError("The uploaded file is not a PDF document.")
        check_document_pages(doc)
        lines = _extract_lines(doc)
        if not lines:
            logger.warning(f"[{source_id}] No extractable text found — is this a scanned PDF?")
            return []
        logger.info(f"[{source_id}] {len(lines)} lines extracted, grouping into sections…")
        sections = _group_into_sections(lines, min_heading_size)
        images_dir = Path(options["processed_dir"]) / "pdf_images" / source_id
        page_images = _extract_page_images(doc, source_id, images_dir)

    if page_images:
        total = sum(len(v) for v in page_images.values())
        logger.info(f"[{source_id}] Extracted {total} embedded image(s) from {len(page_images)} page(s)")

    logger.info(f"[{source_id}] {len(sections)} sections found, splitting oversized…")
    split_sections: list[dict] = []
    for section in sections:
        split_sections.extend(_split_section(section, max_chunk_chars))

    chunks = _build_chunks(source_id, split_sections, page_images=page_images, source_filename=pdf_path.name)
    logger.info(f"[{source_id}] Done — {len(chunks)} chunks produced")
    return chunks
