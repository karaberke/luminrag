import BrandMark from './BrandMark'
import { useEffect, useRef, useState } from 'react'
import type { Message } from '../types'
import EvidenceChip from './EvidenceChip'
import ReflectionBadge from './ReflectionBadge'
import RichText from './LazyRichText'
import { useTheme } from '../theme'
import { citationIndex, citedEvidenceIndices, injectCitationLinks } from '../utils/citations'
import { openSourceEvidence, sourceDisplayName } from '../utils/source'

const modeLabel: Record<string, string> = {
  dense:  'Dense RAG',
  graph:  'Graph RAG',
  hybrid: 'Hybrid RAG',
  none:   'No Retrieval',
}

const darkModeBg: Record<string, string> = {
  dense:  'bg-blue-900/50 text-blue-300 border-blue-700',
  graph:  'bg-indigo-900/50 text-indigo-300 border-indigo-700',
  hybrid: 'bg-violet-900/50 text-violet-300 border-violet-700',
  none:   'bg-slate-800 text-slate-400 border-slate-600',
}

const lightModeBg: Record<string, string> = {
  dense:  'bg-blue-50 text-blue-700 border-blue-200',
  graph:  'bg-indigo-50 text-indigo-700 border-indigo-200',
  hybrid: 'bg-violet-50 text-violet-700 border-violet-200',
  none:   'bg-slate-100 text-slate-500 border-slate-300',
}

interface Props {
  message: Message
}

export default function MessageBubble({ message }: Props) {
  const [showEvidence, setShowEvidence] = useState(false)
  const [showReflection, setShowReflection] = useState(false)
  const [citation, setCitation] = useState<{ index: number } | null>(null)
  const [expandedChips, setExpandedChips] = useState<Set<number>>(new Set())
  const chipRefs = useRef<(HTMLDivElement | null)[]>([])
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const cited = citedEvidenceIndices(message.content, message.evidence?.length ?? 0)

  const handleCitationClick = (n: number) => {
    const index = citationIndex(n, message.evidence?.length ?? 0)
    if (index === null) return
    openSourceEvidence(message.evidence![index])
    setShowEvidence(true)
    setExpandedChips((previous) => new Set([...previous, index]))
    setCitation({ index })
  }
  useEffect(() => {
    if (!citation || !showEvidence) return
    const element = chipRefs.current[citation.index]
    element?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    element?.focus({ preventScroll: true })
    const timer = setTimeout(() => setCitation(null), 2000)
    return () => clearTimeout(timer)
  }, [citation, showEvidence])

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="lumin-message-user max-w-[75%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-3 text-sm text-white shadow">
          {message.content}
        </div>
      </div>
    )
  }

  const modeBg = isDark ? darkModeBg : lightModeBg

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-start gap-3">
        {/* Avatar */}
        <BrandMark size="small" />

        <div className="flex-1 min-w-0">
          {/* Mode badge */}
          {message.routing_mode && (
            <span
              className={`inline-flex items-center text-xs px-2 py-0.5 rounded-full border font-mono mb-2 ${modeBg[message.routing_mode]}`}
            >
              {modeLabel[message.routing_mode]}
              {message.hops && message.hops.length > 0 && (
                <span className="ml-1 opacity-70">· {message.hops.length} graph nodes</span>
              )}
            </span>
          )}

          {/* Answer text — RichText so LaTeX + markdown render. */}
          <div className={`lumin-message-assistant rounded-2xl rounded-tl-sm border px-4 py-3 text-sm leading-relaxed ${
            isDark
              ? 'bg-slate-800 border-slate-700 text-slate-200'
              : 'bg-white border-slate-200 text-slate-800 shadow-sm'
          }`}>
            <RichText
              text={injectCitationLinks(message.content)}
              onCitationClick={message.evidence?.length ? handleCitationClick : undefined}
            />
          </div>

          {/* Toggles */}
          <div className="flex gap-3 mt-2">
            {message.evidence && message.evidence.length > 0 && (
              <button
                onClick={() => setShowEvidence((v) => !v)}
                className={`text-xs transition-colors flex items-center gap-1 ${
                  isDark ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-500 hover:text-indigo-700'
                }`}
              >
                <span>{showEvidence ? '▾' : '▸'}</span>
                {message.evidence.length} retrieved passage{message.evidence.length !== 1 ? 's' : ''} · {cited.size} cited
              </button>
            )}
            {message.reflection && (
              <button
                onClick={() => setShowReflection((v) => !v)}
                className={`text-xs transition-colors flex items-center gap-1 ${
                  isDark ? 'text-teal-400 hover:text-teal-300' : 'text-teal-600 hover:text-teal-800'
                }`}
              >
                <span>{showReflection ? '▾' : '▸'}</span>
                reflection
              </button>
            )}
          </div>

          {/* Evidence */}
          {showEvidence && message.evidence && (
            <div className="mt-2 flex flex-col gap-2">
              {message.evidence.map((chunk, i) => (
                <div key={chunk.id} ref={(el) => { chipRefs.current[i] = el }} tabIndex={-1} aria-label={`Source ${i + 1}: ${sourceDisplayName(chunk)}`}>
                  <EvidenceChip chunk={chunk} index={i} cited={cited.has(i)} highlighted={citation?.index === i} expanded={expandedChips.has(i)} onExpandedChange={(expanded) => setExpandedChips((previous) => { const next = new Set(previous); if (expanded) next.add(i); else next.delete(i); return next })} />
                </div>
              ))}
            </div>
          )}

          {/* Reflection */}
          {showReflection && message.reflection && (
            <ReflectionBadge verdict={message.reflection} />
          )}
        </div>
      </div>
    </div>
  )
}
