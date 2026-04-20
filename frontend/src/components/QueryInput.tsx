import { useRef, useState, type KeyboardEvent } from 'react'
import { useTheme } from '../ThemeContext'

export type RoutingMode = 'auto' | 'dense' | 'graph' | 'hybrid'

interface Props {
  onSubmit: (query: string, routingMode: RoutingMode, maxSources?: number, minRelevancy?: number) => void
  loading: boolean
  onFirstKeystroke?: () => void
  suggestions?: string[]
}

const MODES: { value: RoutingMode; label: string; title: string }[] = [
  { value: 'auto',   label: 'Auto',   title: 'Let the system decide (heuristic + LLM router)' },
  { value: 'dense',  label: 'Dense',  title: 'Force vector/dense retrieval' },
  { value: 'graph',  label: 'Graph',  title: 'Force knowledge-graph retrieval' },
  { value: 'hybrid', label: 'Hybrid', title: 'Combine dense vector and knowledge-graph retrieval' },
]

export default function QueryInput({ onSubmit, loading, onFirstKeystroke, suggestions = [] }: Props) {
  const [value, setValue] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [routingMode, setRoutingMode] = useState<RoutingMode>('auto')
  const [showOptions, setShowOptions] = useState(false)
  const [minRelevancy, setMinRelevancy] = useState(0)
  const [maxSources, setMaxSources] = useState<number | undefined>(undefined)
  const wasEmptyRef = useRef(true)
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || loading) return
    onSubmit(trimmed, routingMode, maxSources, minRelevancy > 0 ? minRelevancy : undefined)
    setValue('')
    wasEmptyRef.current = true
    setShowSuggestions(false)
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const next = e.target.value
    if (wasEmptyRef.current && next.length > 0) {
      wasEmptyRef.current = false
      onFirstKeystroke?.()
    }
    if (next.length === 0) wasEmptyRef.current = true
    setValue(next)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const modeActiveClass = (m: RoutingMode) => {
    if (routingMode !== m) return isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'
    if (m === 'dense')  return 'bg-teal-600 text-white'
    if (m === 'graph')  return 'bg-indigo-600 text-white'
    if (m === 'hybrid') return 'bg-violet-600 text-white'
    return isDark ? 'bg-slate-600 text-slate-100' : 'bg-slate-300 text-slate-800'
  }

  const relevancyLabel = minRelevancy === 0 ? 'No filter' : `${Math.round(minRelevancy * 100)}%`

  return (
    <div className="relative">
      {/* Suggestions */}
      {showSuggestions && !value && suggestions.length > 0 && (
        <div className={`absolute bottom-full left-0 right-0 mb-2 border rounded-xl overflow-hidden shadow-xl z-10 ${
          isDark ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'
        }`}>
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => { setValue(s); setShowSuggestions(false) }}
              className={`w-full text-left px-4 py-2.5 text-sm transition-colors border-b last:border-0 ${
                isDark
                  ? 'text-slate-300 hover:bg-slate-700 hover:text-white border-slate-700/50'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 border-slate-100'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Options panel */}
      {showOptions && (
        <div className={`mb-2 px-3 py-2.5 rounded-xl border text-xs ${
          isDark ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200 shadow-sm'
        }`}>
          <div className="flex flex-wrap gap-x-6 gap-y-2 items-center">
            {/* Min relevancy slider */}
            <label className={`flex flex-col gap-1 min-w-[160px] ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              <span className="flex justify-between">
                <span>Min relevancy</span>
                <span className={`font-medium tabular-nums ${minRelevancy > 0 ? (isDark ? 'text-indigo-400' : 'text-indigo-600') : ''}`}>
                  {relevancyLabel}
                </span>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={minRelevancy}
                onChange={(e) => setMinRelevancy(parseFloat(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </label>

            {/* Max sources number input */}
            <label className={`flex flex-col gap-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              <span>Max sources</span>
              <input
                type="number"
                min={1}
                max={50}
                placeholder="No limit"
                value={maxSources ?? ''}
                onChange={(e) => {
                  const v = e.target.value
                  setMaxSources(v === '' ? undefined : Math.max(1, parseInt(v, 10)))
                }}
                className={`w-24 px-2 py-0.5 rounded-lg border text-xs outline-none ${
                  isDark
                    ? 'bg-slate-900 border-slate-600 text-slate-200 placeholder-slate-600 focus:border-indigo-500'
                    : 'bg-slate-50 border-slate-300 text-slate-800 placeholder-slate-400 focus:border-indigo-400'
                }`}
              />
            </label>
          </div>
        </div>
      )}

      <div className={`flex items-end gap-2 border rounded-xl px-3 py-2 transition-colors ${
        isDark
          ? 'bg-slate-800 border-slate-700 focus-within:border-indigo-500'
          : 'bg-white border-slate-300 focus-within:border-indigo-400 shadow-sm'
      }`}>
        {suggestions.length > 0 && (
          <button
            onClick={() => setShowSuggestions((v) => !v)}
            title="Show example questions"
            className={`mb-1 transition-colors text-lg leading-none ${
              isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-indigo-500'
            }`}
          >
            ✦
          </button>
        )}

        {/* Routing mode selector */}
        <div className={`mb-1 flex items-center rounded-lg border overflow-hidden shrink-0 self-end ${
          isDark ? 'bg-slate-900 border-slate-700' : 'bg-slate-100 border-slate-200'
        }`}>
          {MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => setRoutingMode(m.value)}
              title={m.title}
              className={`px-2 py-1 text-xs font-medium transition-colors ${modeActiveClass(m.value)}`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Options toggle */}
        <button
          onClick={() => setShowOptions((v) => !v)}
          title="Source filtering options"
          className={`mb-1 w-6 h-6 flex items-center justify-center rounded-lg text-xs transition-colors shrink-0 ${
            showOptions || minRelevancy > 0 || maxSources !== undefined
              ? (isDark ? 'bg-indigo-700 text-white' : 'bg-indigo-100 text-indigo-600')
              : (isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600')
          }`}
        >
          ⚙
        </button>

        <textarea
          value={value}
          onChange={handleChange}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about your course material…"
          rows={1}
          className={`flex-1 resize-none bg-transparent text-sm outline-none leading-relaxed py-1 ${
            isDark ? 'text-slate-200 placeholder-slate-500' : 'text-slate-800 placeholder-slate-400'
          }`}
          style={{ minHeight: '36px', maxHeight: '120px' }}
          onInput={(e) => {
            const t = e.currentTarget
            t.style.height = 'auto'
            t.style.height = Math.min(t.scrollHeight, 120) + 'px'
          }}
        />

        <button
          onClick={submit}
          disabled={!value.trim() || loading}
          className={`mb-1 w-8 h-8 rounded-lg text-white flex items-center justify-center transition-colors shrink-0 ${
            isDark
              ? 'bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500'
              : 'bg-indigo-500 hover:bg-indigo-400 disabled:bg-slate-200 disabled:text-slate-400'
          }`}
        >
          {loading ? (
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2.5">
              <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
      </div>
      <p className={`text-xs mt-1.5 text-center ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  )
}
