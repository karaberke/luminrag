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
from pathlib import Path

import fitz  # pymupdf

from backend.schemas import Chunk

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf"}


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

    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:   # type 0 = text block
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue
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
    current_page_start: int = lines[0]["page_number"] if lines else 1

    def _flush() -> None:
        if not current_lines:
            return
        sections.append(
            {
                "title": current_title,
                "text": "\n".join(ln["text"] for ln in current_lines),
                "page_start": current_page_start,
                "page_end": current_lines[-1]["page_number"],
            }
        )

    for line in lines:
        if line["max_font_size"] >= min_heading_size:
            _flush()
            current_title = line["text"]
            current_lines = []
            current_page_start = line["page_number"]
        else:
            current_lines.append(line)

    _flush()
    return sections


# ---------------------------------------------------------------------------
# Step 3: Overflow splitting
# ---------------------------------------------------------------------------

def _hard_split(text: str, max_chars: int) -> list[str]:
    """
    Last-resort split for a single paragraph that exceeds max_chars.
    Tries to break at the nearest sentence boundary ('. ') before the limit;
    falls back to a hard character cut.
    """
    parts: list[str] = []
    while len(text) > max_chars:
        split_at = text.rfind(". ", 0, max_chars)
        split_at = (split_at + 1) if split_at != -1 else max_chars
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        parts.append(text)
    return parts


def _split_section(section: dict, max_chunk_chars: int) -> list[dict]:
    """
    If section text exceeds max_chunk_chars, split at paragraph boundaries
    (double newline). Falls back to _hard_split for paragraphs that are
    themselves too large.

    All produced sub-sections inherit title, page_start, and page_end from
    the parent section (so provenance is preserved).
    """
    text = section["text"]
    if len(text) <= max_chunk_chars:
        return [section]

    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        raw_paras = [text]

    # Hard-split any single paragraph that still exceeds the limit
    paragraphs: list[str] = []
    for para in raw_paras:
        if len(para) > max_chunk_chars:
            paragraphs.extend(_hard_split(para, max_chunk_chars))
        else:
            paragraphs.append(para)

    # Greedily pack paragraphs into sub-chunks
    sub_chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chunk_chars and current:
            sub_chunks.append({**section, "text": "\n\n".join(current)})
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        sub_chunks.append({**section, "text": "\n\n".join(current)})

    return sub_chunks


# ---------------------------------------------------------------------------
# Step 4: Chunk assembly
# ---------------------------------------------------------------------------

def _build_chunks(source_id: str, sections: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        text = section["text"].strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                id=f"{source_id}_chunk_{len(chunks)}",
                text=text,
                source_id=source_id,
                modality="pdf",
                metadata={
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "section_title": section["title"],
                    "char_count": len(text),
                },
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

    source_id = pdf_path.stem
    cfg = config.get("pdf_processor", {})
    max_chunk_chars = cfg.get("max_chunk_chars", 2000)
    min_heading_size = cfg.get("min_heading_size", 14)

    doc = fitz.open(str(pdf_path))
    lines = _extract_lines(doc)
    doc.close()

    if not lines:
        logger.warning(f"[{source_id}] No extractable text found — is this a scanned PDF?")
        return []

    logger.info(f"[{source_id}] {len(lines)} lines extracted, grouping into sections…")
    sections = _group_into_sections(lines, min_heading_size)

    logger.info(f"[{source_id}] {len(sections)} sections found, splitting oversized…")
    split_sections: list[dict] = []
    for section in sections:
        split_sections.extend(_split_section(section, max_chunk_chars))

    chunks = _build_chunks(source_id, split_sections)
    logger.info(f"[{source_id}] Done — {len(chunks)} chunks produced")
    return chunks
