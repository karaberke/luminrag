GENERATE_QUIZ_SYSTEM = (
    "You are an expert educational assessment designer creating quiz questions for "
    "university-level students. Every question MUST be directly answerable from the "
    "provided context — do not use outside knowledge. Generate only JSON, no prose."
)

GENERATE_QUIZ_USER = """\
Create {num_questions} quiz questions from the knowledge base below.

REQUIREMENTS:
- Difficulty: {difficulty}
  - beginner: recall definitions, identify basic facts, true/false on stated claims
  - intermediate: apply concepts, explain relationships, short analysis
  - advanced: synthesise across concepts, evaluate trade-offs, edge cases
- Include ONLY these question types (distribute evenly): {question_types_str}
- {focus_instruction}

QUESTION TYPE RULES:
- multiple_choice: exactly 4 options labeled "A) ...", "B) ...", "C) ...", "D) ..."
  correct_answer must be the full option text, e.g. "A) Newton's first law"
- short_answer: correct_answer is a concise model answer (1-3 sentences)
  rubric lists the key points an answer must address
- true_false: question_text is a declarative statement; correct_answer is exactly "True" or "False"

KNOWLEDGE BASE:
{context}

Return a JSON array containing EXACTLY {num_generate} question object(s).
No markdown fences, no prose. Output ONLY the JSON array.

Example format (replace with real content):
[
  {{
    "question_type": "multiple_choice",
    "question_text": "Which of the following best describes X?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_answer": "A) ...",
    "rubric": "",
    "difficulty": "{difficulty}",
    "source_chunk_ids": ["C000"],
    "source_node_keys": [],
    "hint": "Think about the relationship between ..."
  }}
]

IMPORTANT: source_chunk_ids must use the short IDs exactly as they appear in [Chunk Cxxx | ...] headers above (e.g. "C000", "C001"). Include at least one source_chunk_id per question.\
"""

GRADE_SHORT_ANSWER_SYSTEM = (
    "You are a fair university exam grader. Evaluate whether the student's answer "
    "demonstrates understanding of the key concepts. Be lenient about wording — "
    "check for understanding of ideas, not exact phrasing. Award partial credit when appropriate. "
    "Return only JSON, no prose."
)

GRADE_SHORT_ANSWER_USER = """\
Grade this short answer response.

Question: {question_text}
Model Answer: {correct_answer}
Key Points Required: {rubric}
Student's Answer: {user_answer}

Return JSON ONLY (no markdown fences):
{{
  "is_correct": true,
  "score": 1.0,
  "explanation": "2-3 sentences: what the student got right/wrong and why the correct answer is what it is."
}}\
"""

DEEP_DIVE_SYSTEM = (
    "You are a patient tutor creating a focused study guide for a student who answered "
    "a quiz question incorrectly. Use ONLY the provided context — never introduce outside "
    "knowledge. Return only JSON, no prose."
)

DEEP_DIVE_USER = """\
The student answered this question incorrectly and needs a targeted explanation.

Question: {question_text}
Correct Answer: {correct_answer}

Relevant Context:
{context}

Return JSON ONLY (no markdown fences):
{{
  "concept": "The core concept being tested, in 5-10 words",
  "study_guide": "- **Key idea**: ...\\n- **Why it matters**: ...\\n- **Common mistake**: ...\\n- **Remember**: ...",
  "key_takeaways": ["Concise takeaway 1", "Concise takeaway 2", "Concise takeaway 3"]
}}\
"""
