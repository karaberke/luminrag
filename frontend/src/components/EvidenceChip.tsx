import { useId, useState } from 'react'
import type { EvidenceChunk } from '../types'
import { useTheme } from '../theme'
import RichText from './LazyRichText'
import { chunkSourceUrl, evidenceWithContext, openSourceEvidence, sourceDisplayName, sourceLocation } from '../utils/source'

const modalityIcon: Record<EvidenceChunk['modality'], string> = {
  video: '▶',
  slide: '⊞',
  pdf: '⊟',
  image: '◻',
  audio: '♪',
  note: '▣', document: '▤', markdown: '▤', office: '▤',
}

const darkColor: Record<EvidenceChunk['modality'], string> = {
  video: 'bg-purple-900/60 border-purple-700 text-purple-300',
  slide: 'bg-blue-900/60 border-blue-700 text-blue-300',
  pdf: 'bg-emerald-900/60 border-emerald-700 text-emerald-300',
  image: 'bg-orange-900/60 border-orange-700 text-orange-300',
  audio: 'bg-pink-900/60 border-pink-700 text-pink-300',
  note: 'bg-amber-900/60 border-amber-700 text-amber-300', document: 'bg-blue-900/60 border-blue-700 text-blue-300', markdown: 'bg-blue-900/60 border-blue-700 text-blue-300', office: 'bg-blue-900/60 border-blue-700 text-blue-300',
}

const lightColor: Record<EvidenceChunk['modality'], string> = {
  video: 'bg-purple-50 border-purple-200 text-purple-700',
  slide: 'bg-blue-50 border-blue-200 text-blue-700',
  pdf: 'bg-emerald-50 border-emerald-200 text-emerald-700',
  image: 'bg-orange-50 border-orange-200 text-orange-700',
  audio: 'bg-pink-50 border-pink-200 text-pink-700',
  note: 'bg-amber-50 border-amber-200 text-amber-700', document: 'bg-blue-50 border-blue-200 text-blue-700', markdown: 'bg-blue-50 border-blue-200 text-blue-700', office: 'bg-blue-50 border-blue-200 text-blue-700',
}

const retrievalSourceLabel: Record<string, string> = {
  dense: 'Dense',
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
  cited?: boolean
  expanded?: boolean
  onExpandedChange?: (expanded: boolean) => void
}

export default function EvidenceChip({ chunk: originalChunk, index, highlighted = false, cited = false, expanded, onExpandedChange }: Props) {
  const { theme } = useTheme()
  const chunk = evidenceWithContext(originalChunk)
  const [locallyExpanded, setLocallyExpanded] = useState(false)
  const isExpanded = expanded ?? locallyExpanded
  const passageId = useId()
  const filename = sourceDisplayName(chunk)
  const location = sourceLocation(chunk)
  const isDark = theme === 'dark'
  const color = isDark ? darkColor[chunk.modality] : lightColor[chunk.modality]

  return (
    <article aria-label={`Source ${index + 1}: ${filename}`} className={`rounded-lg border p-3 text-xs transition-shadow duration-300 ${color} ${
      highlighted ? 'ring-2 ring-indigo-400' : ''
    }`}>
      <div className="flex items-center gap-2 mb-1.5">
        <span className="font-mono text-base leading-none">{modalityIcon[chunk.modality]}</span>
        <span className="font-semibold truncate flex-1" title={filename}>{filename}</span>
        {cited && <span className="shrink-0 rounded border border-current px-1.5 py-0.5 text-[10px] font-semibold">Cited</span>}
        {chunk.modality !== 'note' && <a
          href={chunkSourceUrl(chunk)}
          target="_blank"
          rel="noreferrer"
          className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
            isDark ? 'border-current opacity-60 hover:opacity-100' : 'border-current opacity-50 hover:opacity-100'
          }`}
          title={`Open original ${filename}${location ? ` at ${location}` : ''}`}
          aria-label={`Open original ${filename}`}
        >
          ↗
        </a>}
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
      {location && <p className="mb-2 font-medium">{location}</p>}
      <div id={passageId} role={isExpanded ? 'region' : undefined} aria-label={isExpanded ? `Cited passage from ${filename}` : undefined}><RichText
        text={chunk.text}
        className={`leading-relaxed whitespace-pre-wrap ${isExpanded ? '' : 'line-clamp-3'} ${isDark ? 'text-slate-300' : 'text-slate-600'}`}
      /></div>
      <button type="button" aria-expanded={isExpanded} aria-controls={passageId} onClick={() => { const next = !isExpanded; setLocallyExpanded(next); onExpandedChange?.(next) }} className="mt-2 font-semibold underline underline-offset-2">{isExpanded ? 'Collapse passage' : 'Show full passage'}</button>
      <button type="button" onClick={() => openSourceEvidence(chunk)} className="mt-2 ml-4 font-semibold underline underline-offset-2">View in source</button>
    </article>
  )
}
