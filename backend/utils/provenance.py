"""Safe, consistent source locations for query and graph evidence."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from backend.schemas import Chunk, MAX_CHUNK_TEXT_CHARACTERS
from backend.utils.timestamps import format_timestamp

if TYPE_CHECKING:
    from backend.schemas import CitationSpan


def evidence_metadata(chunk: Chunk, citation: CitationSpan | None = None) -> dict:
    metadata = chunk.metadata or {}
    locations = {
        "page": metadata.get("page_number") or metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "slide_number": metadata.get("slide_number") or (metadata.get("page_number") if chunk.modality == "slide" else None),
    }
    for field, value in locations.items():
        try:
            number = int(value) if value is not None and not isinstance(value, bool) else None
            if isinstance(value, float) and value != number:
                number = None
            locations[field] = number if number is not None and 0 < number <= 1_000_000 else None
        except (ValueError, TypeError, OverflowError):
            locations[field] = None
    if locations["page_end"] is not None and locations["page"] is not None and locations["page_end"] < locations["page"]:
        locations["page_end"] = None
    quote_fields = {"quote": None, "quote_start": None, "quote_end": None}
    if citation is not None:
        start, end, quote = citation.quote_start, citation.quote_end, citation.quote
        if citation.chunk_id == chunk.id and isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool) and not isinstance(end, bool) and 0 <= start < end <= min(len(chunk.text), MAX_CHUNK_TEXT_CHARACTERS) and isinstance(quote, str) and quote.strip() and chunk.text[start:end] == quote:
            quote_fields.update(quote=quote, quote_start=start, quote_end=end)
            pages = []
            spans = metadata.get("page_spans")
            if isinstance(spans, list):
                for span in spans:
                    if not isinstance(span, (list, tuple)) or len(span) != 3:
                        continue
                    span_start, span_end, page = span
                    if all(isinstance(value, int) and not isinstance(value, bool) for value in span) and 0 <= span_start < span_end <= len(chunk.text) and 0 < page <= 1_000_000 and span_start < end and span_end > start:
                        pages.append(page)
            if pages:
                locations.update(page=min(pages), page_end=max(pages))
    labels = {
        "source_filename": metadata.get("original_filename") or metadata.get("source_filename"),
        "title": metadata.get("title") or metadata.get("slide_title"),
        "section_title": metadata.get("section_title"),
    }
    for field, value in labels.items():
        labels[field] = value[:2048] if isinstance(value, str) and value.strip() else None
    score = metadata.get("relevancy_score")
    try:
        score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None
        if score is not None and not math.isfinite(score):
            score = None
    except (OverflowError, ValueError):
        score = None
    retrieval_source = metadata.get("retrieval_source")
    if not isinstance(retrieval_source, str) or retrieval_source not in {"dense", "graph", "both"}:
        retrieval_source = None
    return {
        "text": chunk.text[:MAX_CHUNK_TEXT_CHARACTERS],
        "timestamp": format_timestamp(metadata.get("start_time")),
        "retrieval_source": retrieval_source,
        "relevancy_score": score,
        **locations,
        **labels,
        **quote_fields,
    }
