"""Shared linear text splitting with exact PDF character/page provenance."""
from bisect import bisect_left, bisect_right

from backend.schemas import Chunk


def graph_passage_batches(chunks: list[Chunk], *, max_chars: int = 6000, max_passages: int = 8) -> list[list[Chunk]]:
    """Group bounded same-source views; original retrieval chunks stay untouched."""
    max_chars = min(12000, max(2000, int(max_chars)))
    max_passages = min(8, max(1, int(max_passages)))
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    size = 0
    for chunk in chunks:
        # Long custom retrieval chunks receive multiple graph views with the
        # same original ID, so no tail is silently omitted from extraction.
        views = [chunk] if len(chunk.text) <= max_chars else [
            chunk.model_copy(update={"text": chunk.text[offset:offset + max_chars]})
            for offset in range(0, len(chunk.text), max_chars)
        ]
        for view in views:
            if current and (current[0].source_id != view.source_id or len(current) >= max_passages
                            or size + len(view.text) > max_chars or any(item.id == view.id for item in current)):
                batches.append(current)
                current, size = [], 0
            current.append(view)
            size += len(view.text)
    if current:
        batches.append(current)
    return batches

def _hard_split(text: str, max_chars: int) -> list[str]:
    """
    Last-resort split for a single paragraph that exceeds max_chars.
    Tries to break at the nearest sentence boundary ('. ') before the limit;
    falls back to a hard character cut.
    """
    parts: list[str] = []
    offset = 0
    length = len(text)
    while length - offset > max_chars:
        split_at = text.rfind(". ", offset, offset + max_chars)
        split_at = (split_at + 1) if split_at != -1 else offset + max_chars
        piece = text[offset:split_at].strip()
        if piece:
            parts.append(piece)
        offset = split_at
        while offset < length and text[offset].isspace():
            offset += 1
    remainder = text[offset:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def _split_section(section: dict, max_chunk_chars: int) -> list[dict]:
    """
    If section text exceeds max_chunk_chars, split at paragraph boundaries
    (double newline). Falls back to _hard_split for paragraphs that are
    themselves too large.

    When PDF character/page spans are present, each piece receives its actual
    page coverage. Other processors retain their supplied section metadata.
    """
    text = section["text"]
    if len(text) <= max_chunk_chars:
        return [section]

    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        raw_paras = [text]

    # Retain source offsets while splitting. Searches advance monotonically,
    # so repeated paragraphs cannot accidentally resolve to an earlier page.
    paragraphs: list[tuple[str, int, int]] = []
    cursor = 0
    for para in raw_paras:
        start = text.find(para, cursor)
        if start < 0:
            raise ValueError("Could not map section text to its source.")
        cursor = start + len(para)
        piece_cursor = 0
        for piece in _hard_split(para, max_chunk_chars):
            relative = para.find(piece, piece_cursor)
            piece_start = start + relative
            paragraphs.append((piece, piece_start, piece_start + len(piece)))
            piece_cursor = relative + len(piece)

    # Greedily pack paragraphs into sub-chunks
    groups = []
    current: list[tuple[str, int, int]] = []
    current_len = 0

    for para in paragraphs:
        separator = 2 if current else 0
        if current_len + separator + len(para[0]) > max_chunk_chars and current:
            groups.append(current)
            current = [para]
            current_len = len(para[0])
        else:
            current.append(para)
            current_len += separator + len(para[0])

    if current:
        groups.append(current)

    spans = section.get("_page_spans", [])
    starts = [span[0] for span in spans]
    ends = [span[1] for span in spans]
    sub_chunks = []
    for group in groups:
        payload = {**section, "text": "\n\n".join(piece[0] for piece in group)}
        payload.pop("_page_spans", None)
        if spans:
            mapped = []
            output_offset = 0
            for position, piece in enumerate(group):
                if position:
                    output_offset += 2  # paragraph separator in the output
                first = bisect_right(ends, piece[1])
                stop = bisect_left(starts, piece[2])
                for start, end, page in spans[first:stop]:
                    mapped.append((output_offset + max(start, piece[1]) - piece[1], output_offset + min(end, piece[2]) - piece[1], page))
                output_offset += len(piece[0])
            if mapped:
                payload.update(page_start=mapped[0][2], page_end=mapped[-1][2], _page_spans=mapped)
        sub_chunks.append(payload)
    return sub_chunks
