from __future__ import annotations

import logging

from backend.quiz.prompts import (
    DEEP_DIVE_SYSTEM,
    DEEP_DIVE_USER,
    GRADE_SHORT_ANSWER_SYSTEM,
    GRADE_SHORT_ANSWER_USER,
)
from backend.quiz.schemas import (
    DeepDiveResult,
    GradeQuizResult,
    QuizAnswer,
    QuizQuestionResult,
    StoredQuestion,
    StoredQuiz,
)
from backend.graph._llm import call_llm, parse_json_list
from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.retrieval.embedder import Embedder
from backend.retrieval.graph_retriever import retrieve_graph
from backend.retrieval.vector_retriever import VectorIndex, retrieve_dense

logger = logging.getLogger(__name__)


def grade_quiz(
    quiz: StoredQuiz,
    answers: list[QuizAnswer],
    store: DocumentStore,
    graph_builder: GraphBuilder | None,
    llm_config: dict,
) -> GradeQuizResult:
    quiz_cfg = llm_config.get("quiz_generator", llm_config.get("generator", {}))
    answer_map = {a.question_id: a.user_answer.strip() for a in answers}

    results: list[QuizQuestionResult] = []
    for q in quiz.questions:
        user_answer = answer_map.get(q.id, "")
        if q.question_type == "short_answer":
            is_correct, score, explanation = _grade_short_answer(q, user_answer, quiz_cfg)
        else:
            is_correct, score, explanation = _grade_exact(q, user_answer)
        results.append(
            QuizQuestionResult(
                question_id=q.id,
                question_text=q.question_text,
                question_type=q.question_type,
                user_answer=user_answer,
                correct_answer=q.correct_answer,
                is_correct=is_correct,
                score=score,
                explanation=explanation,
                source_chunk_ids=q.source_chunk_ids,
            )
        )

    wrong = [r for r in results if not r.is_correct]
    knowledge_gaps = _find_knowledge_gaps(quiz, wrong, graph_builder)

    total_score = (
        round((sum(r.score for r in results) / len(results)) * 100, 1) if results else 0.0
    )
    num_correct = sum(1 for r in results if r.is_correct)

    return GradeQuizResult(
        quiz_id=quiz.quiz_id,
        total_score=total_score,
        num_correct=num_correct,
        num_total=len(results),
        results=results,
        knowledge_gaps=knowledge_gaps,
        recommended_study_areas=knowledge_gaps,
    )


def _grade_exact(q: StoredQuestion, user_answer: str) -> tuple[bool, float, str]:
    correct = q.correct_answer.strip().lower()
    user = user_answer.strip().lower()

    if q.question_type == "true_false":
        is_correct = user == correct
    else:
        # Accept full option text or single-letter shorthand ("a" matches "a) ...")
        is_correct = user == correct or (
            len(user) == 1 and correct.startswith(f"{user})")
        )

    score = 1.0 if is_correct else 0.0
    explanation = (
        f"Correct! The answer is: {q.correct_answer}"
        if is_correct
        else f"Incorrect. The correct answer is: {q.correct_answer}"
    )
    return is_correct, score, explanation


def _grade_short_answer(
    q: StoredQuestion, user_answer: str, quiz_cfg: dict
) -> tuple[bool, float, str]:
    if not user_answer:
        return False, 0.0, f"No answer provided. The model answer is: {q.correct_answer}"

    prompt = GRADE_SHORT_ANSWER_SYSTEM + "\n\n" + GRADE_SHORT_ANSWER_USER.format(
        question_text=q.question_text,
        correct_answer=q.correct_answer,
        rubric=q.rubric or "Match the key concepts in the model answer.",
        user_answer=user_answer,
    )
    try:
        raw = call_llm(prompt, quiz_cfg, max_tokens=512)
        items = parse_json_list(raw)
        if items:
            item = items[0]
            is_correct = bool(item.get("is_correct", False))
            score = max(0.0, min(1.0, float(item.get("score", 1.0 if is_correct else 0.0))))
            explanation = str(item.get("explanation", ""))
            return is_correct, score, explanation
    except Exception as exc:
        logger.warning("LLM short-answer grading failed: %s", exc)

    # Keyword-overlap fallback when LLM fails
    correct_words = set(q.correct_answer.lower().split())
    user_words = set(user_answer.lower().split())
    score = (
        min(len(correct_words & user_words) / len(correct_words) * 1.5, 1.0)
        if correct_words
        else 0.0
    )
    is_correct = score >= 0.5
    return is_correct, score, f"The model answer is: {q.correct_answer}"


def _find_knowledge_gaps(
    quiz: StoredQuiz,
    wrong_results: list[QuizQuestionResult],
    graph_builder: GraphBuilder | None,
) -> list[str]:
    if not wrong_results or graph_builder is None:
        return []

    graph = graph_builder.graph
    gap_counts: dict[str, int] = {}
    question_map = {q.id: q for q in quiz.questions}

    for result in wrong_results:
        q = question_map.get(result.question_id)
        if not q:
            continue
        for node_key in q.source_node_keys:
            if node_key not in graph:
                continue
            for topic_name in _find_topic_ancestors(node_key, graph):
                gap_counts[topic_name] = gap_counts.get(topic_name, 0) + 1

    return [name for name, _ in sorted(gap_counts.items(), key=lambda x: -x[1])]


def _find_topic_ancestors(node_key: str, graph) -> list[str]:
    """Walk HAS_CONTENT / HAS_SUBTOPIC reverse edges upward to find Topic names."""
    topics: list[str] = []
    queue = [node_key]
    visited = {node_key}

    while queue:
        current = queue.pop()
        for pred in graph.predecessors(current):
            if pred in visited:
                continue
            visited.add(pred)
            edge_data = graph.get_edge_data(pred, current) or {}
            is_structural = any(
                d.get("relation") in ("HAS_SUBTOPIC", "HAS_CONTENT")
                for d in edge_data.values()
            )
            if not is_structural:
                continue
            pred_attrs = graph.nodes.get(pred, {})
            node_type = pred_attrs.get("node_type")
            if node_type == "topic":
                name = pred_attrs.get("name", pred)
                if name not in topics:
                    topics.append(name)
            elif node_type in ("subtopic", "content"):
                queue.append(pred)

    return topics


def get_deep_dive(
    quiz: StoredQuiz,
    question_id: str,
    vector_index: VectorIndex | None,
    graph_builder: GraphBuilder | None,
    store: DocumentStore,
    embedder: Embedder | None,
    llm_config: dict,
) -> DeepDiveResult:
    q = next((q for q in quiz.questions if q.id == question_id), None)
    if q is None:
        raise ValueError(f"Question {question_id!r} not found in quiz {quiz.quiz_id!r}")

    quiz_cfg = llm_config.get("quiz_generator", llm_config.get("generator", {}))
    context_chars = int(quiz_cfg.get("context_chars", 4000))

    chunks_by_id: dict = {}
    if vector_index is not None:
        try:
            dense = retrieve_dense(q.question_text, vector_index, store, llm_config)
            for c in dense.chunks:
                chunks_by_id[c.id] = c
        except Exception:
            pass
    if graph_builder is not None and embedder is not None:
        try:
            gr = retrieve_graph(q.question_text, graph_builder, store, embedder, llm_config)
            for c in gr.chunks:
                chunks_by_id.setdefault(c.id, c)
        except Exception:
            pass

    # Prioritise chunks that sourced this question
    priority_ids = set(q.source_chunk_ids)
    chunks = (
        [c for c in chunks_by_id.values() if c.id in priority_ids]
        + [c for c in chunks_by_id.values() if c.id not in priority_ids]
    )

    context_parts: list[str] = []
    total_chars = 0
    for chunk in chunks:
        if total_chars + len(chunk.text) > context_chars:
            break
        context_parts.append(chunk.text)
        total_chars += len(chunk.text)

    context = "\n\n".join(context_parts) or "No additional context available."

    prompt = DEEP_DIVE_SYSTEM + "\n\n" + DEEP_DIVE_USER.format(
        question_text=q.question_text,
        correct_answer=q.correct_answer,
        context=context,
    )

    concept = q.question_text[:80]
    study_guide = f"**Key Concept**\n\nThe correct answer is: {q.correct_answer}"
    key_takeaways: list[str] = []

    try:
        raw = call_llm(prompt, quiz_cfg, max_tokens=1024)
        items = parse_json_list(raw)
        if items:
            item = items[0]
            concept = item.get("concept") or concept
            study_guide = item.get("study_guide") or study_guide
            key_takeaways = item.get("key_takeaways") or []
    except Exception as exc:
        logger.warning("Deep dive LLM call failed: %s", exc)

    return DeepDiveResult(
        concept=concept,
        study_guide=study_guide,
        key_takeaways=key_takeaways,
        source_chunk_ids=q.source_chunk_ids,
    )
