import { useState } from 'react'
import type { DeepDiveData, QuizGradeResult } from '../types'
import { useTheme } from '../ThemeContext'
import { RichText } from './RichText'
import EvidenceChip from './EvidenceChip'

interface Props {
  quizId: string
  results: QuizGradeResult
  onRetake: () => void
  onDeepDive: (questionId: string) => Promise<DeepDiveData>
}

function ScoreRing({ score, isDark }: { score: number; isDark: boolean }) {
  const r = 40
  const circumference = 2 * Math.PI * r
  const filled = (score / 100) * circumference
  const color =
    score >= 70 ? '#22c55e' : score >= 50 ? '#f59e0b' : '#ef4444'

  return (
    <div className="relative w-28 h-28 flex items-center justify-center">
      <svg className="absolute inset-0 -rotate-90" viewBox="0 0 100 100">
        <circle
          cx="50" cy="50" r={r}
          fill="none"
          stroke={isDark ? '#1e293b' : '#e2e8f0'}
          strokeWidth="10"
        />
        <circle
          cx="50" cy="50" r={r}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.8s ease' }}
        />
      </svg>
      <span className="text-2xl font-bold" style={{ color }}>{Math.round(score)}%</span>
    </div>
  )
}

export default function QuizResults({ results, onRetake, onDeepDive }: Props) {
  void results // quizId kept in Props for future use
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const [deepDives, setDeepDives] = useState<Record<string, DeepDiveData>>({})
  const [deepDiveLoading, setDeepDiveLoading] = useState<Record<string, boolean>>({})
  const [deepDiveOpen, setDeepDiveOpen] = useState<Record<string, boolean>>({})
  const [sourcesOpen, setSourcesOpen] = useState<Record<string, boolean>>({})

  async function handleDeepDive(questionId: string) {
    if (deepDives[questionId]) {
      setDeepDiveOpen((prev) => ({ ...prev, [questionId]: !prev[questionId] }))
      return
    }
    setDeepDiveLoading((prev) => ({ ...prev, [questionId]: true }))
    try {
      const data = await onDeepDive(questionId)
      setDeepDives((prev) => ({ ...prev, [questionId]: data }))
      setDeepDiveOpen((prev) => ({ ...prev, [questionId]: true }))
    } finally {
      setDeepDiveLoading((prev) => ({ ...prev, [questionId]: false }))
    }
  }

  const borderColor = isDark ? 'border-slate-800' : 'border-[#dde5f5]'
  const cardBg = isDark ? 'bg-slate-800/60' : 'bg-white'
  const cardBorder = isDark ? 'border-slate-700' : 'border-slate-200'
  const labelColor = isDark ? 'text-slate-300' : 'text-slate-700'
  const mutedColor = isDark ? 'text-slate-500' : 'text-slate-400'

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Score header */}
      <div className={`px-6 py-5 border-b shrink-0 ${borderColor} ${isDark ? 'bg-slate-900/60' : 'bg-white/80'}`}>
        <div className="flex items-center gap-6">
          <ScoreRing score={results.total_score} isDark={isDark} />
          <div>
            <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>
              Quiz Complete
            </h2>
            <p className={`text-sm mt-0.5 ${mutedColor}`}>
              {results.num_correct} / {results.num_total} correct
            </p>
            {results.knowledge_gaps.length > 0 && (
              <p className={`text-xs mt-1 ${isDark ? 'text-amber-400' : 'text-amber-600'}`}>
                Study areas: {results.knowledge_gaps.join(', ')}
              </p>
            )}
          </div>
          <button
            onClick={onRetake}
            className={`ml-auto text-sm px-4 py-2 rounded-xl border transition-colors ${
              isDark
                ? 'border-slate-700 text-slate-400 hover:text-slate-200 hover:border-indigo-600'
                : 'border-slate-300 text-slate-600 hover:text-slate-800 hover:border-indigo-400 bg-white'
            }`}
          >
            Retake Quiz
          </button>
        </div>

        {/* Knowledge gaps */}
        {results.recommended_study_areas.length > 0 && (
          <div className={`mt-4 rounded-xl border p-3 ${
            isDark ? 'border-amber-900/50 bg-amber-950/20' : 'border-amber-200 bg-amber-50'
          }`}>
            <p className={`text-xs font-semibold mb-2 ${isDark ? 'text-amber-400' : 'text-amber-700'}`}>
              Recommended Study Areas
            </p>
            <div className="flex flex-wrap gap-2">
              {results.recommended_study_areas.map((area) => (
                <span
                  key={area}
                  className={`px-2.5 py-1 rounded-full text-xs border ${
                    isDark
                      ? 'bg-amber-900/30 border-amber-800 text-amber-300'
                      : 'bg-amber-100 border-amber-300 text-amber-700'
                  }`}
                >
                  {area}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Question results */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {results.results.map((result, idx) => {
          const dive = deepDives[result.question_id]
          const isDiveOpen = deepDiveOpen[result.question_id]
          const isSourcesOpen = sourcesOpen[result.question_id]

          return (
            <div key={result.question_id} className={`rounded-2xl border ${cardBg} ${cardBorder} overflow-hidden`}>
              {/* Question header bar */}
              <div className={`flex items-center gap-3 px-5 py-3 border-b ${
                result.is_correct
                  ? isDark ? 'bg-green-950/30 border-green-900/40' : 'bg-green-50 border-green-200'
                  : isDark ? 'bg-red-950/30 border-red-900/40' : 'bg-red-50 border-red-200'
              }`}>
                <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                  result.is_correct
                    ? 'bg-green-500 text-white'
                    : 'bg-red-500 text-white'
                }`}>
                  {result.is_correct ? '✓' : '✗'}
                </span>
                <p className={`text-sm font-medium flex-1 ${labelColor}`}>
                  Q{idx + 1}. {result.question_text}
                </p>
                {result.question_type === 'short_answer' && !result.is_correct && (
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${
                    result.score >= 0.5
                      ? isDark ? 'border-amber-800 text-amber-400 bg-amber-900/20' : 'border-amber-300 text-amber-700 bg-amber-50'
                      : isDark ? 'border-red-800 text-red-400 bg-red-900/20' : 'border-red-300 text-red-600 bg-red-50'
                  }`}>
                    {Math.round(result.score * 100)}%
                  </span>
                )}
              </div>

              <div className="px-5 py-4 space-y-3">
                {/* Answers */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <p className={`font-semibold mb-1 ${mutedColor}`}>Your answer</p>
                    <p className={result.is_correct
                      ? isDark ? 'text-green-400' : 'text-green-700'
                      : isDark ? 'text-red-400' : 'text-red-600'
                    }>
                      {result.user_answer || <span className={mutedColor}>No answer</span>}
                    </p>
                  </div>
                  {!result.is_correct && (
                    <div>
                      <p className={`font-semibold mb-1 ${mutedColor}`}>Correct answer</p>
                      <p className={isDark ? 'text-green-400' : 'text-green-700'}>
                        {result.correct_answer}
                      </p>
                    </div>
                  )}
                </div>

                {/* Explanation */}
                {result.explanation && (
                  <div className={`rounded-xl p-3 text-xs ${
                    isDark ? 'bg-slate-900/60 text-slate-300' : 'bg-slate-50 text-slate-700'
                  }`}>
                    <RichText text={result.explanation} />
                  </div>
                )}

                {/* Sources + deep dive buttons */}
                <div className="flex items-center gap-2 flex-wrap">
                  {(result.evidence ?? []).length > 0 && (
                    <button
                      onClick={() => setSourcesOpen((prev) => ({ ...prev, [result.question_id]: !prev[result.question_id] }))}
                      className={`text-xs px-2.5 py-1 rounded-lg border transition-colors ${
                        isDark
                          ? 'border-slate-700 text-slate-400 hover:text-slate-200'
                          : 'border-slate-300 text-slate-500 hover:text-slate-700'
                      }`}
                    >
                      {isSourcesOpen ? 'Hide sources' : `Sources (${result.evidence.length})`}
                    </button>
                  )}
                  {!result.is_correct && (
                    <button
                      onClick={() => handleDeepDive(result.question_id)}
                      disabled={deepDiveLoading[result.question_id]}
                      className={`text-xs px-2.5 py-1 rounded-lg border transition-colors flex items-center gap-1.5 ${
                        isDark
                          ? 'border-indigo-800 text-indigo-400 hover:bg-indigo-900/30 disabled:opacity-40'
                          : 'border-indigo-300 text-indigo-600 hover:bg-indigo-50 disabled:opacity-40'
                      }`}
                    >
                      {deepDiveLoading[result.question_id] ? (
                        <div className="w-3 h-3 border border-t-indigo-400 rounded-full animate-spin" />
                      ) : (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3">
                          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      )}
                      Explain this concept
                    </button>
                  )}
                </div>

                {/* Sources */}
                {isSourcesOpen && (result.evidence ?? []).length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {result.evidence.map((chunk, chunkIdx) => (
                      <EvidenceChip key={chunk.id} chunk={chunk} index={chunkIdx} />
                    ))}
                  </div>
                )}

                {/* Deep dive */}
                {isDiveOpen && dive && (
                  <div className={`rounded-xl border p-4 space-y-3 ${
                    isDark
                      ? 'border-indigo-900/50 bg-indigo-950/20'
                      : 'border-indigo-200 bg-indigo-50/50'
                  }`}>
                    <p className={`text-xs font-semibold ${isDark ? 'text-indigo-300' : 'text-indigo-700'}`}>
                      Study Guide: {dive.concept}
                    </p>
                    <div className={`text-xs ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                      <RichText text={dive.study_guide} />
                    </div>
                    {dive.key_takeaways.length > 0 && (
                      <div>
                        <p className={`text-xs font-semibold mb-1.5 ${mutedColor}`}>Key takeaways</p>
                        <ul className="space-y-1">
                          {dive.key_takeaways.map((t, i) => (
                            <li key={i} className={`text-xs flex items-start gap-2 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                              <span className="text-indigo-400 shrink-0">•</span>
                              {t}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
