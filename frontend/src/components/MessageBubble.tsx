import { useRef, useState } from 'react'
import type { Message } from '../types'
import EvidenceChip from './EvidenceChip'
import ReflectionBadge from './ReflectionBadge'
import { RichText } from './RichText'
import { useTheme } from '../ThemeContext'

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

// Convert standalone [N] citation markers into markdown links (#citation-N).
// Skips [N](url) and [N][ref] patterns that are already part of a link.
function injectCitationLinks(text: string): string {
  return text.replace(/\[(\d+)\](?!\(|\[)/g, '[$1](#citation-$1)')
}

export default function MessageBubble({ message }: Props) {
  const [showEvidence, setShowEvidence] = useState(false)
  const [showReflection, setShowReflection] = useState(false)
  const [highlightedChip, setHighlightedChip] = useState<number | null>(null)
  const chipRefs = useRef<(HTMLDivElement | null)[]>([])
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const handleCitationClick = (n: number) => {
    setShowEvidence(true)
    const idx = n - 1
    setTimeout(() => {
      chipRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      setHighlightedChip(idx)
      setTimeout(() => setHighlightedChip(null), 1500)
    }, 60)
  }

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-3 text-sm text-white shadow">
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
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5">
          L
        </div>

        <div className="flex-1 min-w-0">
          {/* Mode badge */}
          {message.routing_mode && (
            <span
              className={`inline-flex items-center text-xs px-2 py-0.5 rounded-full border font-mono mb-2 ${modeBg[message.routing_mode]}`}
            >
              {modeLabel[message.routing_mode]}
              {message.hops && message.hops.length > 0 && (
                <span className="ml-1 opacity-70">· {message.hops.length} hops</span>
              )}
            </span>
          )}

          {/* Answer text — RichText so LaTeX + markdown render. */}
          <div className={`rounded-2xl rounded-tl-sm border px-4 py-3 text-sm leading-relaxed ${
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
                {message.evidence.length} source{message.evidence.length !== 1 ? 's' : ''}
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
                <div key={chunk.id} ref={(el) => { chipRefs.current[i] = el }}>
                  <EvidenceChip chunk={chunk} index={i} highlighted={highlightedChip === i} />
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
