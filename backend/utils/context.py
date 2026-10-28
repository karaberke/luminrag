"""One bounded passage view for answer generation and source verification."""

from __future__ import annotations

from backend.schemas import Chunk, RetrievalResult
from backend.utils.query_intent import multi_document_source_limit

DEFAULT_GENERATION_CONTEXT_CHARS = 4000
DEFAULT_MULTI_DOCUMENT_CONTEXT_CHARS = 12000
_SEPARATOR = "\n\n"
_TRUNCATION = "... (truncated — context limit reached)"


def generation_context_chars(question: str | None, config: dict) -> int:
    """Passage budget shared by generation and answer reflection.

    Explicit multi-document questions use generator.multi_document_max_context_chars;
    all others use generator.max_context_chars. Each configured value wins as is.
    """
    generator = config.get("generator", {})
    if multi_document_source_limit(question):
        return generator.get("multi_document_max_context_chars", DEFAULT_MULTI_DOCUMENT_CONTEXT_CHARS)
    return generator.get("max_context_chars", DEFAULT_GENERATION_CONTEXT_CHARS)


def _header(index: int, chunk: Chunk) -> str:
    return f"[{index}] ({chunk.modality}) "


def _greedy_chunks(chunks: list[Chunk], budget: int) -> list[Chunk]:
    included: list[Chunk] = []
    total = 0
    for index, chunk in enumerate(chunks, start=1):
        header = _header(index, chunk)
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


def _fair_shares(demands: list[int], total: int) -> list[int]:
    """Water-fill *total* characters; short demands release room to longer ones."""
    shares = [0] * len(demands)
    remaining = max(0, total)
    order = sorted(range(len(demands)), key=lambda i: (demands[i], i))
    for position, i in enumerate(order):
        shares[i] = min(demands[i], remaining // (len(order) - position))
        remaining -= shares[i]
    return shares


def _source_balanced_chunks(chunks: list[Chunk], budget: int, source_limit: int) -> list[Chunk] | None:
    """Split the budget across the first represented sources, or None for greedy.

    Returns a contiguous prefix of *chunks* in original order, ending at the
    first passage of the last represented source, so every [N] keeps its
    number. Each source gets a fair share of the text budget, split again over
    its own passages in that prefix; every passage keeps a prefix from offset 0.
    """
    first_index: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        if chunk.source_id not in first_index:
            if len(first_index) >= source_limit:
                break
            first_index[chunk.source_id] = index
    representatives = list(first_index.values())
    for count in range(len(representatives), 1, -1):
        prefix = chunks[:representatives[count - 1] + 1]
        overhead = sum(len(_header(index, chunk)) for index, chunk in enumerate(prefix, start=1))
        text_budget = budget - overhead - len(_SEPARATOR) * (len(prefix) - 1)
        if text_budget < len(prefix):
            continue
        if sum(len(chunk.text) for chunk in prefix) <= text_budget:
            return None  # the ordinary greedy view already shows every representative
        members: dict[str, list[int]] = {}
        for index, chunk in enumerate(prefix):
            members.setdefault(chunk.source_id, []).append(index)
        source_shares = _fair_shares([sum(len(prefix[i].text) for i in group) for group in members.values()],
                                     text_budget)
        limits = [0] * len(prefix)
        for group, share in zip(members.values(), source_shares):
            for i, limit in zip(group, _fair_shares([len(prefix[i].text) for i in group], share)):
                limits[i] = limit
        if any(chunk.text and not limit for chunk, limit in zip(prefix, limits)):
            continue
        return [chunk if len(chunk.text) <= limit else chunk.model_copy(update={"text": chunk.text[:limit]})
                for chunk, limit in zip(prefix, limits)]
    return None


def context_chunks(result: RetrievalResult, max_chars: int, *, question: str | None = None) -> list[Chunk]:
    """Keep bounded passage prefixes with offsets relative to original text.

    With *question* naming an explicit multi-document comparison, the budget is
    shared across the first represented sources instead of letting one long
    passage consume it. Otherwise (and without *question*) selection is greedy.
    """
    max_chars = max(0, max_chars)
    full_size = sum(len(f"[{index}] ({chunk.modality}) {chunk.text}")
                    for index, chunk in enumerate(result.chunks, start=1))
    full_size += len(_SEPARATOR) * max(0, len(result.chunks) - 1)
    # Reserve the notice so headers and separators also fit the same budget.
    budget = max_chars if full_size <= max_chars else max(0, max_chars - len(_SEPARATOR) - len(_TRUNCATION))
    if question is not None and full_size > max_chars:
        source_limit = multi_document_source_limit(question)
        if source_limit:
            balanced = _source_balanced_chunks(result.chunks, budget, source_limit)
            if balanced is not None:
                return balanced
    return _greedy_chunks(result.chunks, budget)


def format_context(result: RetrievalResult, max_chars: int, *, question: str | None = None) -> str:
    """Format the exact numbered passage prefixes visible within the budget."""
    if not result.chunks:
        return "(no context retrieved)"[:max(0, max_chars)]
    included = context_chunks(result, max_chars, question=question)
    lines = [f"[{index}] ({chunk.modality}) {chunk.text}" for index, chunk in enumerate(included, start=1)]
    if len(included) < len(result.chunks) or any(chunk.text != original.text for chunk, original in zip(included, result.chunks)):
        lines.append(_TRUNCATION)
    return _SEPARATOR.join(lines)[:max(0, max_chars)]
