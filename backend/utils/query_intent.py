"""Narrow, provider-free detection of explicit multi-document questions.

Public API:
    limit = multi_document_source_limit(question)
        0 for ordinary questions, 2 for explicit "both/two papers|documents|
        sources", up to MAX_DOCUMENT_SOURCES for broader explicit comparisons
        or summaries across several documents.

Every pattern is a bounded, non-nested regex over a length-capped question,
so matching stays linear. The intent never calls a provider.
"""

from __future__ import annotations

import re

MAX_DOCUMENT_SOURCES = 4
_MAX_QUESTION_CHARS = 2000

_CONTAINERS = r"(?:papers|documents|docs|articles|files|readings)"
_SINGULAR = r"(?:paper|document|article|file|reading)"
_DETERMINER = r"(?:of\s+)?(?:the\s+|these\s+|those\s+)?(?:\w+\s+)?"

# Documents are explicit containers; a comparison or summary cue is required.
_PAIR_RE = re.compile(r"\b(?:both|two|2|pair\s+of)\s+" + _DETERMINER + _CONTAINERS + r"\b")
_THREE_RE = re.compile(r"\b(?:three|3)\s+" + _DETERMINER + _CONTAINERS + r"\b")
_MANY_RE = re.compile(
    r"\b(?:these|those|the|all|several|multiple|four|4|selected|provided|uploaded|different|various|my)\s+"
    r"(?:\w+\s+)?" + _CONTAINERS + r"\b|\beach\s+" + _SINGULAR + r"\b"
)
# "sources" alone is ambiguous ("sources of noise"); require an explicit
# document quantifier and accept a citation cue as well.
_SOURCES_RE = re.compile(
    r"\b(both|two|2|three|3|four|4|these|those|provided|uploaded|selected|given)\s+"
    r"(?:(two|three|four|provided|uploaded|selected)\s+)?sources\b(?!\s+of\b)"
)
_COMPARE_RE = re.compile(
    r"\b(?:compar\w*|contrast\w*|differ\w*|distinguish\w*|versus|vs|trade-?offs?|similarit\w*"
    r"|in\s+common|summar\w*|overview|synthes\w*)\b"
)
_CITE_RE = re.compile(r"\bcit(?:e|es|ed|ing|ation|ations)\b")

_PAIR_WORDS = frozenset({"both", "two", "2"})
_THREE_WORDS = frozenset({"three", "3"})


def multi_document_source_limit(question: str | None) -> int:
    """Return how many distinct documents an explicit comparison asks for."""
    if not question:
        return 0
    q = question[:_MAX_QUESTION_CHARS].lower()
    compare = _COMPARE_RE.search(q) is not None
    if compare:
        if _PAIR_RE.search(q):
            return 2
        if _THREE_RE.search(q):
            return 3
        if _MANY_RE.search(q):
            return MAX_DOCUMENT_SOURCES
    if compare or _CITE_RE.search(q):
        match = _SOURCES_RE.search(q)
        if match:
            words = {match.group(1), match.group(2)}
            if words & _PAIR_WORDS:
                return 2
            if words & _THREE_WORDS:
                return 3
            return MAX_DOCUMENT_SOURCES
    return 0
