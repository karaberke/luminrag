"""One bounded passage view for answer generation and source verification."""

from __future__ import annotations

from backend.schemas import Chunk, RetrievalResult

DEFAULT_GENERATION_CONTEXT_CHARS = 4000
_SEPARATOR = "\n\n"
_TRUNCATION = "... (truncated — context limit reached)"


def context_chunks(result: RetrievalResult, max_chars: int) -> list[Chunk]:
    """Keep bounded passage prefixes with offsets relative to original text."""
    max_chars = max(0, max_chars)
    full_size = sum(len(f"[{index}] ({chunk.modality}) {chunk.text}")
                    for index, chunk in enumerate(result.chunks, start=1))
    full_size += len(_SEPARATOR) * max(0, len(result.chunks) - 1)
    # Reserve the notice so headers and separators also fit the same budget.
    budget = max_chars if full_size <= max_chars else max(0, max_chars - len(_SEPARATOR) - len(_TRUNCATION))
    included: list[Chunk] = []
    total = 0
    for index, chunk in enumerate(result.chunks, start=1):
        header = f"[{index}] ({chunk.modality}) "
        separator_size = len(_SEPARATOR) if included else 0
        available = budget - total - separator_size - len(header)
        if available <= 0:
            break
        if len(chunk.text) > available:
            included.append(chunk.model_copy(update={"text": chunk.text[:available]}))
            break
        included.append(chunk)
        total += separator_size + len(header) + len(chunk.text)
    return included


def format_context(result: RetrievalResult, max_chars: int) -> str:
    """Format the exact numbered passage prefixes visible within the budget."""
    if not result.chunks:
        return "(no context retrieved)"[:max(0, max_chars)]
    included = context_chunks(result, max_chars)
    lines = [f"[{index}] ({chunk.modality}) {chunk.text}" for index, chunk in enumerate(included, start=1)]
    if len(included) < len(result.chunks) or any(chunk.text != original.text for chunk, original in zip(included, result.chunks)):
        lines.append(_TRUNCATION)
    return _SEPARATOR.join(lines)[:max(0, max_chars)]
