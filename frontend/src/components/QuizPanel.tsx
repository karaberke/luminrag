import { useEffect, useRef, useState } from 'react'
import type { DeepDiveData, GeneratedQuiz, QuizAnswer, QuizConfig, QuizGradeResult, QuizJobStatus } from '../types'
import { useTheme } from '../ThemeContext'
import QuizView from './QuizView'
import QuizResults from './QuizResults'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

type QuizPhase = 'config' | 'generating' | 'taking' | 'grading' | 'results'

const QUESTION_COUNTS = [5, 10, 20] as const
const DIFFICULTIES = ['beginner', 'intermediate', 'advanced'] as const
const QUESTION_TYPES = [
  { id: 'multiple_choice', label: 'Multiple Choice' },
  { id: 'short_answer', label: 'Short Answer' },
  { id: 'true_false', label: 'True / False' },
] as const

interface Props {
  isDark: boolean
}

export default function QuizPanel({ isDark }: Props) {
  const { theme } = useTheme()
  const _isDark = theme === 'dark' || isDark

  const [phase, setPhase] = useState<QuizPhase>('config')
  const [, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState('Queued')
  const [quiz, setQuiz] = useState<GeneratedQuiz | null>(null)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [results, setResults] = useState<QuizGradeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  // Config form state
  const [numQuestions, setNumQuestions] = useState<5 | 10 | 20>(5)
  const [difficulty, setDifficulty] = useState<QuizConfig['difficulty']>('intermediate')
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set(['multiple_choice', 'true_false']))
  const [focusArea, setFocusArea] = useState('')

  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current)
    }
  }, [])

  // Restore quiz state from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('lumin-quiz-state')
    if (!saved) return
    try {
      const s = JSON.parse(saved)
      if (s.phase === 'taking' && s.quiz) {
        setQuiz(s.quiz)
        setAnswers(s.answers ?? {})
        setPhase('taking')
      }
    } catch {
      localStorage.removeItem('lumin-quiz-state')
    }
  }, [])

  // Persist quiz state while in 'taking' phase
  useEffect(() => {
    if (phase === 'taking' && quiz) {
      localStorage.setItem('lumin-quiz-state', JSON.stringify({ phase, quiz, answers }))
    }
  }, [phase, quiz, answers])

  const pollJob = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/quiz/jobs/${id}`)
      if (!res.ok) throw new Error(`Status ${res.status}`)
      const data: QuizJobStatus = await res.json()
      setProgress(data.progress)
      if (data.status === 'done' && data.quiz_id && data.questions) {
        setQuiz({ quiz_id: data.quiz_id, questions: data.questions })
        setAnswers({})
        setPhase('taking')
      } else if (data.status === 'failed') {
        setError(data.error ?? 'Quiz generation failed.')
        setPhase('config')
      } else {
        pollRef.current = setTimeout(() => pollJob(id), 2000)
      }
    } catch {
      pollRef.current = setTimeout(() => pollJob(id), 3000)
    }
  }

  const handleGenerate = async () => {
    if (selectedTypes.size === 0) {
      setError('Select at least one question type.')
      return
    }
    setError(null)
    setPhase('generating')
    setProgress('Queued')
    try {
      const config: QuizConfig = {
        num_questions: numQuestions,
        question_types: Array.from(selectedTypes) as QuizConfig['question_types'],
        difficulty,
        focus_area: focusArea.trim() || undefined,
      }
      const res = await fetch(`${API_BASE}/api/quiz/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text)
      }
      const { job_id } = await res.json()
      setJobId(job_id)
      pollRef.current = setTimeout(() => pollJob(job_id), 1000)
    } catch (e) {
      setError(`Failed to start quiz: ${e instanceof Error ? e.message : String(e)}`)
      setPhase('config')
    }
  }

  const handleSubmit = async (quizAnswers: QuizAnswer[]) => {
    if (!quiz) return
    setSubmitting(true)
    setPhase('grading')
    try {
      const res = await fetch(`${API_BASE}/api/quiz/${quiz.quiz_id}/grade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: quizAnswers }),
      })
      if (res.status === 404) {
        // Server was restarted — quiz no longer in memory. Clear stale state.
        localStorage.removeItem('lumin-quiz-state')
        setQuiz(null)
        setAnswers({})
        setPhase('config')
        setError('Quiz session expired (server was restarted). Please generate a new quiz.')
        return
      }
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text)
      }
      const data: QuizGradeResult = await res.json()
      localStorage.removeItem('lumin-quiz-state')
      setResults(data)
      setPhase('results')
    } catch (e) {
      setError(`Grading failed: ${e instanceof Error ? e.message : String(e)}`)
      setPhase('taking')
    } finally {
      setSubmitting(false)
    }
  }

  const handleDeepDive = async (questionId: string): Promise<DeepDiveData> => {
    if (!quiz) throw new Error('No quiz loaded')
    const res = await fetch(`${API_BASE}/api/quiz/${quiz.quiz_id}/deep-dive/${questionId}`, {
      method: 'POST',
    })
    if (!res.ok) throw new Error(`Deep dive failed: ${res.status}`)
    return res.json()
  }

  const handleRetake = () => {
    localStorage.removeItem('lumin-quiz-state')
    if (pollRef.current) clearTimeout(pollRef.current)
    setPhase('config')
    setJobId(null)
    setQuiz(null)
    setAnswers({})
    setResults(null)
    setError(null)
  }

  const borderColor = _isDark ? 'border-slate-800' : 'border-[#dde5f5]'
  const cardBg = _isDark ? 'bg-slate-800/60' : 'bg-white'
  const cardBorder = _isDark ? 'border-slate-700' : 'border-slate-200'
  const labelColor = _isDark ? 'text-slate-300' : 'text-slate-700'
  const mutedColor = _isDark ? 'text-slate-500' : 'text-slate-400'
  const inputBg = _isDark
    ? 'bg-slate-900 border-slate-700 text-slate-200 placeholder-slate-600 focus:border-indigo-500'
    : 'bg-white border-slate-300 text-slate-800 placeholder-slate-400 focus:border-indigo-400'

  if (phase === 'taking' && quiz) {
    return (
      <QuizView
        quiz={quiz}
        answers={answers}
        onAnswerChange={(qId, val) => setAnswers((prev) => ({ ...prev, [qId]: val }))}
        onSubmit={handleSubmit}
        submitting={submitting}
      />
    )
  }

  if (phase === 'results' && results && quiz) {
    return (
      <QuizResults
        quizId={quiz.quiz_id}
        results={results}
        onRetake={handleRetake}
        onDeepDive={handleDeepDive}
      />
    )
  }

  if (phase === 'generating' || phase === 'grading') {
    const msg = phase === 'grading' ? 'Grading your answers…' : progress
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className={`w-12 h-12 border-4 border-t-indigo-500 rounded-full animate-spin ${
          _isDark ? 'border-slate-700' : 'border-slate-200'
        }`} />
        <p className={`text-sm font-medium ${labelColor}`}>{msg}</p>
        {phase === 'generating' && (
          <p className={`text-xs ${mutedColor}`}>This may take 30–60 seconds…</p>
        )}
      </div>
    )
  }

  // Config phase
  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className={`px-6 py-5 border-b shrink-0 ${borderColor} ${_isDark ? 'bg-slate-900/60' : 'bg-white/80'}`}>
        <h2 className={`text-lg font-bold ${_isDark ? 'text-white' : 'text-slate-900'}`}>Generate Quiz</h2>
        <p className={`text-xs mt-0.5 ${mutedColor}`}>
          Create a personalised quiz from your course materials.
        </p>
      </div>

      <div className="flex-1 px-6 py-6 space-y-6 max-w-2xl">
        {/* Error banner */}
        {error && (
          <div className={`rounded-xl border px-4 py-3 text-sm ${
            _isDark ? 'border-red-900/50 bg-red-950/20 text-red-400' : 'border-red-200 bg-red-50 text-red-600'
          }`}>
            {error}
          </div>
        )}

        {/* Number of questions */}
        <div>
          <p className={`text-xs font-semibold mb-2 ${labelColor}`}>Number of Questions</p>
          <div className="flex gap-2">
            {QUESTION_COUNTS.map((n) => (
              <button
                key={n}
                onClick={() => setNumQuestions(n)}
                className={`flex-1 py-2 rounded-xl border text-sm font-medium transition-colors ${
                  numQuestions === n
                    ? _isDark ? 'border-indigo-500 bg-indigo-900/30 text-indigo-300' : 'border-indigo-400 bg-indigo-50 text-indigo-700'
                    : _isDark ? `${cardBg} ${cardBorder} ${labelColor} hover:border-slate-600` : `${cardBg} ${cardBorder} ${labelColor} hover:border-slate-300`
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* Question types */}
        <div>
          <p className={`text-xs font-semibold mb-2 ${labelColor}`}>Question Types</p>
          <div className="flex flex-wrap gap-2">
            {QUESTION_TYPES.map(({ id, label }) => {
              const selected = selectedTypes.has(id)
              return (
                <button
                  key={id}
                  onClick={() => {
                    setSelectedTypes((prev) => {
                      const next = new Set(prev)
                      if (selected) next.delete(id)
                      else next.add(id)
                      return next
                    })
                  }}
                  className={`px-3.5 py-1.5 rounded-xl border text-xs font-medium transition-colors ${
                    selected
                      ? _isDark ? 'border-indigo-500 bg-indigo-900/30 text-indigo-300' : 'border-indigo-400 bg-indigo-50 text-indigo-700'
                      : _isDark ? `${cardBorder} text-slate-400 hover:border-slate-600` : `${cardBorder} ${mutedColor} hover:border-slate-300`
                  }`}
                >
                  {label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Difficulty */}
        <div>
          <p className={`text-xs font-semibold mb-2 ${labelColor}`}>Difficulty</p>
          <div className="flex gap-2">
            {DIFFICULTIES.map((d) => (
              <button
                key={d}
                onClick={() => setDifficulty(d)}
                className={`flex-1 py-2 rounded-xl border text-xs font-medium transition-colors capitalize ${
                  difficulty === d
                    ? _isDark ? 'border-indigo-500 bg-indigo-900/30 text-indigo-300' : 'border-indigo-400 bg-indigo-50 text-indigo-700'
                    : _isDark ? `${cardBg} ${cardBorder} ${labelColor} hover:border-slate-600` : `${cardBg} ${cardBorder} ${labelColor} hover:border-slate-300`
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        {/* Focus area */}
        <div>
          <p className={`text-xs font-semibold mb-2 ${labelColor}`}>Focus Area <span className={`font-normal ${mutedColor}`}>(optional)</span></p>
          <input
            type="text"
            value={focusArea}
            onChange={(e) => setFocusArea(e.target.value)}
            placeholder="e.g. neural networks, gradient descent…"
            className={`w-full rounded-xl border px-4 py-2.5 text-sm outline-none transition-colors ${inputBg}`}
          />
        </div>

        {/* Generate button */}
        <button
          onClick={handleGenerate}
          disabled={selectedTypes.size === 0}
          className="w-full py-3 rounded-xl text-sm font-semibold bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Generate Quiz
        </button>
      </div>
    </div>
  )
}
