import { useState } from 'react'
import type { GeneratedQuiz, HintData, QuizAnswer } from '../types'
import { useTheme } from '../ThemeContext'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

interface Props {
  quiz: GeneratedQuiz
  answers: Record<string, string>
  onAnswerChange: (questionId: string, value: string) => void
  onSubmit: (answers: QuizAnswer[]) => void
  submitting: boolean
}

export default function QuizView({ quiz, answers, onAnswerChange, onSubmit, submitting }: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const [hints, setHints] = useState<Record<string, HintData>>({})
  const [hintLoading, setHintLoading] = useState<Record<string, boolean>>({})
  const [hintOpen, setHintOpen] = useState<Record<string, boolean>>({})

  const answeredCount = Object.values(answers).filter((v) => v.trim() !== '').length
  const allAnswered = answeredCount === quiz.questions.length

  async function fetchHint(quizId: string, questionId: string) {
    if (hints[questionId]) {
      setHintOpen((prev) => ({ ...prev, [questionId]: !prev[questionId] }))
      return
    }
    setHintLoading((prev) => ({ ...prev, [questionId]: true }))
    try {
      const res = await fetch(`${API_BASE}/api/quiz/${quizId}/hint/${questionId}`)
      if (res.ok) {
        const data: HintData = await res.json()
        setHints((prev) => ({ ...prev, [questionId]: data }))
        setHintOpen((prev) => ({ ...prev, [questionId]: true }))
      }
    } catch {
      // silently ignore
    } finally {
      setHintLoading((prev) => ({ ...prev, [questionId]: false }))
    }
  }

  function handleSubmit() {
    const quizAnswers: QuizAnswer[] = quiz.questions.map((q) => ({
      question_id: q.id,
      user_answer: answers[q.id] ?? '',
    }))
    onSubmit(quizAnswers)
  }

  const borderColor = isDark ? 'border-slate-800' : 'border-[#dde5f5]'
  const cardBg = isDark ? 'bg-slate-800/60' : 'bg-white'
  const cardBorder = isDark ? 'border-slate-700' : 'border-slate-200'
  const labelColor = isDark ? 'text-slate-300' : 'text-slate-700'
  const mutedColor = isDark ? 'text-slate-500' : 'text-slate-400'
  const inputBg = isDark
    ? 'bg-slate-900 border-slate-700 text-slate-200 placeholder-slate-600 focus:border-indigo-500'
    : 'bg-white border-slate-300 text-slate-800 placeholder-slate-400 focus:border-indigo-400'

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Progress bar */}
      <div className={`px-6 py-3 border-b flex items-center gap-4 shrink-0 ${borderColor} ${isDark ? 'bg-slate-900/60' : 'bg-white/80'}`}>
        <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
          {answeredCount} / {quiz.questions.length} answered
        </span>
        <div className={`flex-1 h-1.5 rounded-full overflow-hidden ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`}>
          <div
            className="h-full bg-indigo-500 rounded-full transition-all duration-300"
            style={{ width: `${(answeredCount / quiz.questions.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Questions */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {quiz.questions.map((q, idx) => {
          const hint = hints[q.id]
          const isHintOpen = hintOpen[q.id]

          return (
            <div
              key={q.id}
              className={`rounded-2xl border p-5 ${cardBg} ${cardBorder}`}
            >
              {/* Question header */}
              <div className="flex items-start justify-between gap-3 mb-4">
                <div className="flex items-start gap-3">
                  <span className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                    answers[q.id]?.trim()
                      ? 'bg-indigo-600 text-white'
                      : isDark ? 'bg-slate-700 text-slate-400' : 'bg-slate-200 text-slate-500'
                  }`}>
                    {idx + 1}
                  </span>
                  <div>
                    <p className={`text-sm font-medium leading-relaxed ${labelColor}`}>
                      {q.question_text}
                    </p>
                    <span className={`text-xs mt-1 inline-block ${mutedColor}`}>
                      {q.question_type === 'multiple_choice' && 'Multiple choice'}
                      {q.question_type === 'short_answer' && 'Short answer'}
                      {q.question_type === 'true_false' && 'True / False'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => fetchHint(quiz.quiz_id, q.id)}
                  disabled={hintLoading[q.id]}
                  className={`shrink-0 text-xs px-2.5 py-1 rounded-lg border transition-colors flex items-center gap-1.5 ${
                    isDark
                      ? 'border-amber-800 text-amber-400 hover:bg-amber-900/30 disabled:opacity-40'
                      : 'border-amber-300 text-amber-600 hover:bg-amber-50 disabled:opacity-40'
                  }`}
                  title="Get a hint"
                >
                  {hintLoading[q.id] ? (
                    <div className="w-3 h-3 border border-t-amber-400 rounded-full animate-spin" />
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 16v-4M12 8h.01" strokeLinecap="round" />
                    </svg>
                  )}
                  Hint
                </button>
              </div>

              {/* Hint panel */}
              {isHintOpen && hint && (
                <div className={`mb-4 rounded-xl p-3 text-xs border ${
                  isDark
                    ? 'bg-amber-950/30 border-amber-900/50 text-amber-300'
                    : 'bg-amber-50 border-amber-200 text-amber-800'
                }`}>
                  <p className="font-medium mb-1">Hint</p>
                  <p>{hint.hint}</p>
                  {hint.related_concepts.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {hint.related_concepts.map((c) => (
                        <span key={c} className={`px-2 py-0.5 rounded-full border text-xs ${
                          isDark
                            ? 'bg-amber-900/30 border-amber-800 text-amber-400'
                            : 'bg-amber-100 border-amber-300 text-amber-700'
                        }`}>
                          {c}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Answer input */}
              {q.question_type === 'multiple_choice' && q.options && (
                <fieldset className="space-y-2">
                  {q.options.map((option) => (
                    <label
                      key={option}
                      className={`flex items-center gap-3 rounded-xl border px-4 py-3 cursor-pointer transition-colors ${
                        answers[q.id] === option
                          ? isDark
                            ? 'border-indigo-500 bg-indigo-900/30'
                            : 'border-indigo-400 bg-indigo-50'
                          : isDark
                          ? 'border-slate-700 hover:border-slate-600 hover:bg-slate-700/30'
                          : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                      }`}
                    >
                      <input
                        type="radio"
                        name={q.id}
                        value={option}
                        checked={answers[q.id] === option}
                        onChange={() => onAnswerChange(q.id, option)}
                        className="accent-indigo-500"
                      />
                      <span className={`text-sm ${labelColor}`}>{option}</span>
                    </label>
                  ))}
                </fieldset>
              )}

              {q.question_type === 'true_false' && (
                <div className="flex gap-3">
                  {(['True', 'False'] as const).map((val) => (
                    <label
                      key={val}
                      className={`flex-1 flex items-center justify-center gap-2 rounded-xl border px-4 py-3 cursor-pointer transition-colors ${
                        answers[q.id] === val
                          ? isDark
                            ? 'border-indigo-500 bg-indigo-900/30'
                            : 'border-indigo-400 bg-indigo-50'
                          : isDark
                          ? 'border-slate-700 hover:border-slate-600'
                          : 'border-slate-200 hover:border-slate-300'
                      }`}
                    >
                      <input
                        type="radio"
                        name={q.id}
                        value={val}
                        checked={answers[q.id] === val}
                        onChange={() => onAnswerChange(q.id, val)}
                        className="accent-indigo-500"
                      />
                      <span className={`text-sm font-medium ${labelColor}`}>{val}</span>
                    </label>
                  ))}
                </div>
              )}

              {q.question_type === 'short_answer' && (
                <textarea
                  rows={4}
                  value={answers[q.id] ?? ''}
                  onChange={(e) => onAnswerChange(q.id, e.target.value)}
                  placeholder="Type your answer here…"
                  className={`w-full rounded-xl border px-4 py-3 text-sm resize-none outline-none transition-colors ${inputBg}`}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* Submit */}
      <div className={`px-6 py-4 border-t shrink-0 ${borderColor} ${isDark ? 'bg-slate-900/60' : 'bg-white/80'}`}>
        <button
          onClick={handleSubmit}
          disabled={!allAnswered || submitting}
          className="w-full py-3 rounded-xl text-sm font-semibold bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {submitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Grading…
            </>
          ) : (
            'Submit Quiz'
          )}
        </button>
        {!allAnswered && (
          <p className={`text-xs text-center mt-2 ${mutedColor}`}>
            Answer all {quiz.questions.length} questions to submit
          </p>
        )}
      </div>
    </div>
  )
}
