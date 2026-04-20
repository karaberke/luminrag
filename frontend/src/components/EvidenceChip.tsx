import type { EvidenceChunk } from '../types'
import { useTheme } from '../ThemeContext'
import { RichText } from './RichText'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

function parseTimestamp(ts: string | undefined): number | null {
  if (!ts) return null
  const parts = ts.split(':').map((p) => parseInt(p, 10))
  if (parts.some(Number.isNaN)) return null
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  return null
}

function sourceUrl(chunk: EvidenceChunk): string {
  const base = `${API_BASE}/api/source/${encodeURIComponent(chunk.source)}`
  if ((chunk.modality === 'pdf' || chunk.modality === 'slide') && chunk.page != null) {
    return `${base}#page=${chunk.page}`
  }
  if (chunk.modality === 'video' || chunk.modality === 'audio') {
    const secs = parseTimestamp(chunk.timestamp)
    if (secs != null) return `${base}#t=${secs}`
  }
  return base
}

const modalityIcon: Record<EvidenceChunk['modality'], string> = {
  video: '▶',
  slide: '⊞',
  pdf: '⊟',
  image: '◻',
  audio: '♪',
}

const darkColor: Record<EvidenceChunk['modality'], string> = {
  video: 'bg-purple-900/60 border-purple-700 text-purple-300',
  slide: 'bg-blue-900/60 border-blue-700 text-blue-300',
  pdf: 'bg-emerald-900/60 border-emerald-700 text-emerald-300',
  image: 'bg-orange-900/60 border-orange-700 text-orange-300',
  audio: 'bg-pink-900/60 border-pink-700 text-pink-300',
}

const lightColor: Record<EvidenceChunk['modality'], string> = {
  video: 'bg-purple-50 border-purple-200 text-purple-700',
  slide: 'bg-blue-50 border-blue-200 text-blue-700',
  pdf: 'bg-emerald-50 border-emerald-200 text-emerald-700',
  image: 'bg-orange-50 border-orange-200 text-orange-700',
  audio: 'bg-pink-50 border-pink-200 text-pink-700',
}

const retrievalSourceLabel: Record<string, string> = {
  dense: 'Vector',
  graph: 'Graph',
  both:  'Both',
}

const retrievalSourceStyle: Record<string, { dark: string; light: string }> = {
  dense: { dark: 'bg-teal-900/60 text-teal-300 border-teal-700',   light: 'bg-teal-50 text-teal-700 border-teal-200' },
  graph: { dark: 'bg-indigo-900/60 text-indigo-300 border-indigo-700', light: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
  both:  { dark: 'bg-violet-900/60 text-violet-300 border-violet-700', light: 'bg-violet-50 text-violet-700 border-violet-200' },
}

interface Props {
  chunk: EvidenceChunk
  index: number
  highlighted?: boolean
}

export default function EvidenceChip({ chunk, index, highlighted = false }: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const color = isDark ? darkColor[chunk.modality] : lightColor[chunk.modality]

  return (
    <div className={`rounded-lg border p-3 text-xs transition-shadow duration-300 ${color} ${
      highlighted ? (isDark ? 'ring-2 ring-indigo-400' : 'ring-2 ring-indigo-400') : ''
    }`}>
      <div className="flex items-center gap-2 mb-1.5">
        <span className="font-mono text-base leading-none">{modalityIcon[chunk.modality]}</span>
        <span className="font-semibold truncate flex-1">{chunk.source}</span>
        <span className="shrink-0 opacity-60">
          {chunk.page ? `p.${chunk.page}` : chunk.timestamp ? `@${chunk.timestamp}` : ''}
        </span>
        <a
          href={sourceUrl(chunk)}
          target="_blank"
          rel="noreferrer"
          className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
            isDark ? 'border-current opacity-60 hover:opacity-100' : 'border-current opacity-50 hover:opacity-100'
          }`}
          title="Open source in a new tab"
        >
          ↗
        </a>
        {chunk.retrieval_source && retrievalSourceStyle[chunk.retrieval_source] && (
          <span
            className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border font-medium ${
              isDark
                ? retrievalSourceStyle[chunk.retrieval_source].dark
                : retrievalSourceStyle[chunk.retrieval_source].light
            }`}
            title={
              chunk.retrieval_source === 'dense' ? 'Found via vector search' :
              chunk.retrieval_source === 'graph' ? 'Found via knowledge graph' :
              'Found by both vector search and knowledge graph'
            }
          >
            {retrievalSourceLabel[chunk.retrieval_source]}
          </span>
        )}
        <span className={`shrink-0 rounded px-1.5 py-0.5 font-mono ${isDark ? 'bg-white/10' : 'bg-black/5'}`}>[{index + 1}]</span>
      </div>
      <RichText
        text={chunk.text}
        className={`leading-relaxed line-clamp-3 ${isDark ? 'text-slate-300' : 'text-slate-600'}`}
      />
    </div>
  )
}
