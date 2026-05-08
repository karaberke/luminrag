from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["multiple_choice", "short_answer", "true_false"]


class QuizConfig(BaseModel):
    num_questions: int = 10
    question_types: list[QuestionType]
    focus_area: str | None = None
    difficulty: Literal["beginner", "intermediate", "advanced"] = "intermediate"


class StoredQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_type: QuestionType
    question_text: str
    options: list[str] | None = None
    correct_answer: str
    rubric: str = ""
    difficulty: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_node_keys: list[str] = Field(default_factory=list)
    hint: str = ""


class StoredQuiz(BaseModel):
    quiz_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    questions: list[StoredQuestion]
    config: QuizConfig
    created_at: str


# ---------------------------------------------------------------------------
# Client-facing (answers stripped)
# ---------------------------------------------------------------------------

class QuizQuestionResponse(BaseModel):
    id: str
    question_type: str
    question_text: str
    options: list[str] | None = None
    difficulty: str


class QuizJobResponse(BaseModel):
    job_id: str
    status: str


class QuizJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: str
    quiz_id: str | None = None
    questions: list[QuizQuestionResponse] | None = None
    error: str | None = None


class QuizAnswer(BaseModel):
    question_id: str
    user_answer: str


class GradeQuizRequest(BaseModel):
    answers: list[QuizAnswer]


class HintResponse(BaseModel):
    hint: str
    related_concepts: list[str]


# ---------------------------------------------------------------------------
# Internal grading models (no HTTP evidence — resolved by main.py)
# ---------------------------------------------------------------------------

class QuizQuestionResult(BaseModel):
    question_id: str
    question_text: str
    question_type: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    score: float
    explanation: str
    source_chunk_ids: list[str]


class GradeQuizResult(BaseModel):
    quiz_id: str
    total_score: float
    num_correct: int
    num_total: int
    results: list[QuizQuestionResult]
    knowledge_gaps: list[str]
    recommended_study_areas: list[str]


class DeepDiveResult(BaseModel):
    concept: str
    study_guide: str
    key_takeaways: list[str]
    source_chunk_ids: list[str]
