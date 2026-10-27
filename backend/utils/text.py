"""Shared semantic text and conservative citation span matching."""

from __future__ import annotations

import re

_INLINE_MARKUP = re.compile(r"\*\*|__|~~|`{1,3}|\*|_")


def _citation_text(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse spacing and paired inline markup while retaining source offsets."""
    ignored: set[int] = set()
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    opened: dict[str, tuple[int, int, int]] = {}
    line, previous = 0, 0
    for match in _INLINE_MARKUP.finditer(text):
        # Tokenize only fixed delimiters, then pair them in one pass. A regex
        # searching arbitrary content between unmatched delimiters is quadratic.
        start, end = match.span()
        line += text.count("\n", previous, start)
        previous = end
        marker = match.group()
        before, after = text[start - 1:start], text[end:end + 1]
        opening = opened.get(marker)
        if opening and opening[2] == line and before and not before.isspace() and not (after.isalnum() or after == "_"):
            opening_start, opening_end, _ = opened.pop(marker)
            ignored.update(range(opening_start, opening_end))
            ignored.update(range(start, end))
            starts[opening_end] = opening_start
            ends[start - 1] = end
        elif after and not after.isspace() and not (before.isalnum() or before == "_"):
            opened[marker] = (start, end, line)
    characters: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if index in ignored:
            continue
        if character.isspace():
            if characters and characters[-1] == " ":
                offsets[-1] = (offsets[-1][0], index + 1)
            else:
                characters.append(" ")
                offsets.append((starts.get(index, index), ends.get(index, index + 1)))
        else:
            characters.append(character)
            offsets.append((starts.get(index, index), ends.get(index, index + 1)))
    return "".join(characters), offsets


def find_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    """Find an unambiguous real quote; never fuzzy-match or invent source text."""
    quote = quote.strip()
    if not quote:
        return None
    start = text.find(quote)
    if start >= 0:
        return (start, start + len(quote)) if text.find(quote, start + 1) < 0 else None
    normalized, offsets = _citation_text(text)
    needle = _citation_text(quote)[0].strip()
    if not needle:
        return None
    start = normalized.find(needle)
    if start < 0 or normalized.find(needle, start + 1) >= 0:
        return None
    return offsets[start][0], offsets[start + len(needle) - 1][1]


def node_text(attrs: dict, key: str = "") -> str:
    name = attrs.get("name") or attrs.get("title") or key
    summary = (attrs.get("body") or attrs.get("summary") or attrs.get("summary_intermediate")
               or attrs.get("summary_beginner") or "")
    text = f"{name}. {summary}".strip()
    terms = [term for term in attrs.get("key_terms", []) if isinstance(term, str) and term.strip()]
    if terms:
        text += "\nKey terms: " + ", ".join(terms)
    if attrs.get("raw_excerpt"):
        text += "\nExcerpt: " + str(attrs["raw_excerpt"])
    return text
