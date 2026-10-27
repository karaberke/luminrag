"""Heading-aware Markdown/text chunking, sharing the PDF overflow splitter."""
from __future__ import annotations

from pathlib import Path

from backend.utils.chunking import _split_section
from backend.schemas import Chunk

def _heading_title(line: str) -> str | None:
    # Avoid ambiguous regex repetitions: a malicious line with millions of
    # spaces must take one linear pass, not regex backtracking time.
    prefix = len(line) - len(line.lstrip("#"))
    if not 1 <= prefix <= 6 or len(line) <= prefix or not line[prefix].isspace():
        return None
    title = line[prefix:].strip().rstrip("#").rstrip()
    return title or None


def _slide_number(line: str) -> int | None:
    if not line.startswith("<!--") or not line.endswith("-->"):
        return None
    marker = line[4:-3].strip()
    prefix = "slide number:"
    if not marker.lower().startswith(prefix):
        return None
    value = marker[len(prefix):].strip()
    if value.isascii() and value.isdigit() and len(value) <= 6:
        number = int(value)
        return number if number > 0 else None
    return None


def chunk_markdown(text: str, source_id: str, config: dict, *, modality: str = "markdown") -> list[Chunk]:
    maximum = max(200, min(20_000, int(config.get("markdown_processor", {}).get("max_chunk_chars", config.get("pdf_processor", {}).get("max_chunk_chars", 2000)))))
    sections = []
    title = None
    parts = []
    fenced = False
    slide_number = None
    slide_title = None
    in_comment = False

    def flush() -> None:
        if any(part.strip() for part in parts):
            sections.append({"title": title, "text": "\n".join(parts), "page_start": 1, "page_end": 1, "slide_number": slide_number, "slide_title": slide_title})

    for line in text.splitlines():
        if in_comment:
            if "-->" not in line:
                continue
            line = line.split("-->", 1)[1]
            in_comment = False
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced:
            stripped = line.strip()
            marker = _slide_number(stripped)
            if marker is not None:
                flush()
                slide_number, slide_title, title, parts = marker, None, None, []
                continue
            if stripped.startswith("<!--"):
                if "-->" not in stripped:
                    in_comment = True
                    continue
                line = stripped.split("-->", 1)[1]
                if not line.strip():
                    continue
        heading = _heading_title(line) if not fenced else None
        if heading:
            flush()
            title = heading
            if slide_number is not None and slide_title is None:
                slide_title = heading
            parts = [line]
        else:
            parts.append(line)
    flush()
    chunks = []
    for section in sections:
        for piece in _split_section(section, maximum):
            content = piece["text"].strip()
            if content:
                metadata = {"section_title": section["title"], "char_count": len(content)}
                if section["slide_number"] is not None:
                    metadata.update(slide_number=section["slide_number"], slide_title=section["slide_title"])
                chunks.append(Chunk(id=f"{source_id}_chunk_{len(chunks)}", text=content, source_id=source_id, modality=modality, metadata=metadata))
    return chunks


def process_markdown(path: str | Path, config: dict) -> list[Chunk]:
    path = Path(path)
    chunks = chunk_markdown(path.read_text(encoding="utf-8-sig", errors="replace"), path.stem, config)
    for chunk in chunks:
        chunk.metadata["original_filename"] = path.name
    return chunks
