"""
Stage 3 - Query Routing: query_router.py

Decides which retrieval mode to use for a given question:
  "dense"  — single-hop factual lookup (vector DB)
  "graph"  — relational/multi-concept query (knowledge graph traversal)
  "none"   — no retrieval needed (trivial input, greeting, etc.)

Two-pass hybrid approach:
  Pass 1: Heuristic rules — instant, no LLM call, covers obvious cases.
  Pass 2: LLM fallback   — only invoked when heuristics are ambiguous.
                            Can be disabled via config (use_llm_fallback: false).

Public API:
    mode = route_query(question, config)
"""

from __future__ import annotations

import logging
import re

from backend.schemas import RoutingMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic signal tables
# ---------------------------------------------------------------------------

# Common words that don't count toward "content word" length.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "will", "have", "has", "had",
    "be", "been", "being", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "in", "on", "at", "of", "to", "for", "with", "by", "from", "about",
    "into", "that", "this", "these", "those", "and", "or", "but", "not",
    "also", "just", "very", "please",
})

_GREETING_PREFIXES = (
    "hi ", "hello", "hey ", "thanks", "thank you", "ok", "okay",
    "good morning", "good afternoon", "good evening",
)

# Questions that map deterministically to "dense" (single-hop factual).
_DENSE_PREFIXES = (
    "what is ", "what are ", "what was ", "what were ",
    "define ", "definition of ",
    "who is ", "who was ", "who invented ", "who discovered ",
    "when did ", "when was ", "when were ",
    "where is ", "where was ",
    "list ", "name ", "give an example of ",
    "how many ", "how much ",
    "which ",
)

# Questions that map deterministically to "graph" (multi-hop relational).
_GRAPH_PREFIXES = (
    "why ",
    "how does ", "how do ", "how did ", "how would ",
    "explain ", "describe the relationship",
    "compare ", "contrast ",
    "what is the difference", "what are the differences",
    "what is the relationship", "what is the connection",
    "what causes ", "what effect ",
    "in what way",
)

# Phrases anywhere in the question that strongly signal multi-hop reasoning.
_GRAPH_SIGNALS = (
    "compared to", "in comparison",
    "difference between", "relationship between", "connection between",
    "affect", "effect on", "impact on",
    "leads to", "results in", "causes",
    "in terms of", "with respect to",
    "similar to", "as opposed to",
    "contrast between", "trade-off",
    "why does", "why do", "why is",
)

# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

def _content_words(question: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]+\b", question.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _classify_heuristic(question: str) -> RoutingMode | None:
    """
    Return a RoutingMode when heuristics are confident, None when ambiguous.
    Ambiguous questions are escalated to the LLM fallback.

    Order matters: prefix/signal checks run before length check so that
    short but well-formed questions ("What is RAG?") are never mislabelled.
    """
    q = question.strip()
    if not q:
        return "none"

    q_lower = q.lower()

    # Graph signals first — "why does X affect Y?" beats any other pattern
    if any(q_lower.startswith(p) for p in _GRAPH_PREFIXES):
        return "graph"
    if any(signal in q_lower for signal in _GRAPH_SIGNALS):
        return "graph"

    # Dense signals — "What is X?", "Define Y", etc.
    if any(q_lower.startswith(p) for p in _DENSE_PREFIXES):
        return "dense"

    # Greetings / small talk
    if any(q_lower.startswith(g) for g in _GREETING_PREFIXES):
        return "none"

    # Trivially short with no recognised pattern
    if len(_content_words(q)) < 3:
        return "none"

    return None  # ambiguous — defer to LLM


# ---------------------------------------------------------------------------
# LLM fallback classifier
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a query classifier for an educational QA system.

Classify the following question into exactly one of these categories:
- "dense": factual, single-concept questions (What is X? Define Y. Who invented Z?)
- "graph": relational or multi-concept questions that require reasoning across \
multiple topics (Why does X cause Y? Compare A and B. How does X affect Y in context of Z?)
- "none": not a domain question, too vague, or a greeting

Question: {question}

Reply with ONLY one word: dense, graph, or none.\
"""

_VALID_MODES: frozenset[RoutingMode] = frozenset({"dense", "graph", "none"})


def _parse_llm_mode(response: str) -> RoutingMode:
    """Extract mode from LLM response; fall back to 'dense' if unrecognisable."""
    word = response.strip().lower().strip("\"'.,")
    if word in _VALID_MODES:
        return word  # type: ignore[return-value]
    # Search for any valid mode word anywhere in the response
    for mode in ("graph", "none", "dense"):  # preference order
        if mode in response.lower():
            return mode  # type: ignore[return-value]
    logger.warning(f"Unrecognisable LLM routing response: {response!r} — defaulting to 'dense'")
    return "dense"


def _classify_llm_ollama(question: str, cfg: dict) -> RoutingMode:
    import httpx

    prompt = _LLM_PROMPT.format(question=question)
    payload = {
        "model": cfg.get("llm_ollama_model", "llama3.2:3b"),
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": cfg.get("llm_max_tokens", 50)},
    }
    base_url = cfg.get("llm_ollama_base_url", "http://localhost:11434")
    response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=60)
    response.raise_for_status()
    return _parse_llm_mode(response.json()["response"])


def _classify_llm_anthropic(question: str, cfg: dict) -> RoutingMode:
    import anthropic

    client = anthropic.Anthropic()
    prompt = _LLM_PROMPT.format(question=question)
    message = client.messages.create(
        model=cfg.get("llm_anthropic_model", "claude-haiku-4-5-20251001"),
        max_tokens=cfg.get("llm_max_tokens", 50),
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_llm_mode(message.content[0].text)


def _classify_llm(question: str, cfg: dict) -> RoutingMode:
    """Call the configured LLM; fall back to 'dense' on any error."""
    provider = cfg.get("llm_provider", "ollama")
    try:
        if provider == "ollama":
            return _classify_llm_ollama(question, cfg)
        if provider == "anthropic":
            return _classify_llm_anthropic(question, cfg)
        raise ValueError(f"Unknown router LLM provider: '{provider}'")
    except Exception as exc:
        logger.warning(f"LLM routing fallback failed: {exc} — defaulting to 'dense'")
        return "dense"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(question: str, config: dict) -> RoutingMode:
    """
    Determine the retrieval mode for a question.

    Args:
        question: Raw user question string.
        config:   Parsed contents of config/llm.yaml.

    Returns:
        "dense"  — use vector retrieval.
        "graph"  — use knowledge graph traversal.
        "none"   — no retrieval; answer directly or decline.
    """
    if not question or not question.strip():
        return "none"

    cfg = config.get("query_router", {})

    mode = _classify_heuristic(question)
    if mode is not None:
        logger.debug(f"Heuristic routed '{question[:60]}' → {mode}")
        return mode

    if cfg.get("use_llm_fallback", True):
        mode = _classify_llm(question, cfg)
        logger.debug(f"LLM routed '{question[:60]}' → {mode}")
        return mode

    # Heuristic inconclusive + LLM disabled → conservative default
    logger.debug(f"Heuristic inconclusive, LLM disabled → dense")
    return "dense"
