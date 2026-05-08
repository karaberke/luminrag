from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, NamedTuple

from backend.quiz.prompts import GENERATE_QUIZ_SYSTEM, GENERATE_QUIZ_USER
from backend.quiz.schemas import QuizConfig, StoredQuestion, StoredQuiz
from backend.graph._llm import call_llm, parse_json_list
from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval.vector_retriever import VectorIndex, retrieve_dense

logger = logging.getLogger(__name__)

_VALID_Q_TYPES = {"multiple_choice", "short_answer", "true_false"}


def _to_str(val: object) -> str:
    """Coerce an LLM field that should be a string but may come back as a list."""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val) if val is not None else ""


class _Triple(NamedTuple):
    head: str
    relation: str
    tail: str


def generate_quiz(
    config: QuizConfig,
    vector_index: VectorIndex | None,
    graph_builder: GraphBuilder | None,
    store: DocumentStore,
    embedder,  # kept for API compatibility, no longer used
    llm_config: dict,
    progress_cb: Callable[[str], None] | None = None,
) -> StoredQuiz:
    quiz_cfg = llm_config.get("quiz_generator", llm_config.get("generator", {}))
    context_chars = int(quiz_cfg.get("context_chars", 24000))
    max_tokens = int(quiz_cfg.get("max_tokens", 8192))
    batch_size = int(quiz_cfg.get("batch_size", 5))

    query = (
        config.focus_area.strip()
        if config.focus_area
        else "key concepts definitions principles relationships"
    )

    # --- Full-corpus chunk retrieval ------------------------------------------
    all_chunks = store.get_all_chunks()
    if not all_chunks:
        raise ValueError("No content found. Please ingest documents first.")

    # Branch: focus_area → depth (top-N by similarity, no stride skipping)
    #         no focus_area → breadth (even-sample for full-corpus coverage)
    if config.focus_area and vector_index is not None:
        try:
            ranked = retrieve_dense(query, vector_index, store, {
                **llm_config,
                "vector_retriever": {
                    **llm_config.get("vector_retriever", {}),
                    "top_k": len(all_chunks),
                },
            })
            order = {c.id: i for i, c in enumerate(ranked.chunks)}
            all_chunks.sort(key=lambda c: order.get(c.id, len(all_chunks)))
        except Exception as exc:
            logger.warning("Focus-area ranking failed: %s", exc)
        budget, selected = 0, []
        for c in all_chunks:
            if budget + len(c.text) > context_chars:
                break
            selected.append(c)
            budget += len(c.text)
        all_chunks = selected or all_chunks[:1]
    else:
        # No focus area: use vector store to surface concept-dense chunks first,
        # then backfill with even-sampled corpus chunks for breadth.
        vector_chunks: list = []
        if vector_index is not None:
            try:
                ranked = retrieve_dense(query, vector_index, store, {
                    **llm_config,
                    "vector_retriever": {
                        **llm_config.get("vector_retriever", {}),
                        "top_k": min(40, len(all_chunks)),
                    },
                })
                vector_chunks = ranked.chunks
            except Exception as exc:
                logger.warning("Vector retrieval for quiz failed: %s", exc)

        # Even-sample the full corpus for breadth coverage
        if sum(len(c.text) for c in all_chunks) > context_chars:
            step = max(1, len(all_chunks) // (context_chars // 300))
            sampled = all_chunks[::step]
        else:
            sampled = all_chunks[:]

        # Merge: vector results first (highest priority), then sampled backfill
        seen_ids = {c.id for c in vector_chunks}
        all_chunks = vector_chunks + [c for c in sampled if c.id not in seen_ids]

    # --- Degree-ranked graph triples ------------------------------------------
    triples: list[_Triple] = []
    node_keys: set[str] = set()
    if graph_builder is not None:
        g = graph_builder.graph
        node_keys = set(g.nodes())
        top_nodes = {n for n, _ in sorted(g.degree(), key=lambda x: -x[1])[:50]}
        seen: set[tuple] = set()
        for src, dst, data in g.edges(data=True):
            if src not in top_nodes and dst not in top_nodes:
                continue
            head = g.nodes[src].get("name", src)
            tail = g.nodes[dst].get("name", dst)
            key = (head, data.get("relation", ""), tail)
            if key not in seen:
                seen.add(key)
                triples.append(_Triple(head=head, relation=data.get("relation", ""), tail=tail))

    # --- Build context string --------------------------------------------------
    # Use short sequential IDs (C000, C001, ...) instead of raw UUIDs so the LLM
    # can reliably cite them — opaque UUIDs are frequently hallucinated.
    short_id_map: dict[str, str] = {}  # short_id → real chunk.id
    context_parts: list[str] = []
    total_chars = 0
    for idx, chunk in enumerate(all_chunks):
        short_id = f"C{idx:03d}"
        short_id_map[short_id] = chunk.id
        meta = getattr(chunk, "metadata", {}) or {}
        page = meta.get("page_number") or meta.get("page_start", "")
        page_str = f" | page {page}" if page else ""
        entry = f'[Chunk {short_id}{page_str} | {chunk.source_id}]\n"{chunk.text}"'
        if total_chars + len(entry) > context_chars:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    if triples:
        graph_lines = ["Graph Relationships:"]
        for t in triples[:50]:
            line = f'  - "{t.head}" --{t.relation}--> "{t.tail}"'
            if total_chars + len(line) > context_chars:
                break
            graph_lines.append(line)
            total_chars += len(line)
        if len(graph_lines) > 1:
            context_parts.append("\n".join(graph_lines))

    context = "\n\n".join(context_parts)
    if not context.strip():
        raise ValueError("No context available for quiz generation.")

    question_types_str = ", ".join(config.question_types)
    focus_instruction = (
        f"Focus all questions on: {config.focus_area}"
        if config.focus_area
        else "Cover the breadth of the knowledge base."
    )

    # --- Batching loop --------------------------------------------------------
    # Use num_questions * 2 so that even if the LLM returns only 1 question per
    # batch (common for 7B models with large context), we still reach the target.
    max_batches = config.num_questions * 2
    valid_questions: list[StoredQuestion] = []
    generated_questions: list[str] = []  # full question texts to avoid near-duplicates

    for batch_num in range(max_batches):
        if len(valid_questions) >= config.num_questions:
            break

        still_need = config.num_questions - len(valid_questions)
        ask_for = min(batch_size, still_need + 2)

        avoid_instruction = (
            "\n\nDo NOT repeat or rephrase any of these already-generated questions — "
            "each new question must test a different concept:\n"
            + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(generated_questions[-15:]))
            if generated_questions else ""
        )

        if progress_cb:
            batch_start = len(valid_questions) + 1
            batch_end = len(valid_questions) + ask_for
            progress_cb(f"Generating questions {batch_start}–{batch_end}…")

        prompt = GENERATE_QUIZ_SYSTEM + "\n\n" + GENERATE_QUIZ_USER.format(
            num_questions=ask_for,
            num_generate=ask_for,
            difficulty=config.difficulty,
            question_types_str=question_types_str,
            focus_instruction=focus_instruction + avoid_instruction,
            context=context,
        )

        try:
            raw = call_llm(prompt, quiz_cfg, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("Quiz batch %d LLM call failed: %s", batch_num, exc)
            continue

        batch_items = parse_json_list(raw)
        logger.info("Quiz batch %d: LLM returned %d items", batch_num, len(batch_items))

        for item in batch_items:
            q_type = (item.get("question_type") or "").strip()
            if q_type not in _VALID_Q_TYPES or q_type not in config.question_types:
                continue

            q_text = (item.get("question_text") or "").strip()
            correct = (item.get("correct_answer") or "").strip()
            if not q_text or not correct:
                continue

            # Map short IDs (C000, C001, ...) back to real chunk UUIDs.
            # The context uses short IDs so the LLM can cite them reliably.
            raw_chunk_ids = [
                short_id_map[cid]
                for cid in (item.get("source_chunk_ids") or [])
                if cid in short_id_map
            ]
            if not raw_chunk_ids:
                logger.debug("Dropping sourceless question (likely hallucinated): %.60s", q_text)
                continue

            if q_type == "multiple_choice":
                options = item.get("options") or []
                if len(options) < 2:
                    continue
            else:
                options = None

            generated_questions.append(q_text)
            valid_questions.append(
                StoredQuestion(
                    id=str(uuid.uuid4()),
                    question_type=q_type,
                    question_text=q_text,
                    options=options,
                    correct_answer=correct,
                    rubric=_to_str(item.get("rubric")),
                    difficulty=config.difficulty,
                    source_chunk_ids=raw_chunk_ids,
                    source_node_keys=[
                        k for k in (item.get("source_node_keys") or []) if k in node_keys
                    ],
                    hint=_to_str(item.get("hint")),
                )
            )

            if len(valid_questions) >= config.num_questions:
                break

    if not valid_questions:
        raise ValueError(
            "The LLM did not return valid quiz questions. "
            "Try a different focus area or question types."
        )

    return StoredQuiz(
        quiz_id=str(uuid.uuid4()),
        questions=valid_questions,
        config=config,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
