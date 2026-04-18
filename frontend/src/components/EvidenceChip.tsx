import type { EvidenceChunk } from '../types'
import { useTheme } from '../ThemeContext'

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

interface Props {
  chunk: EvidenceChunk
  index: number
}

export default function EvidenceChip({ chunk, index }: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const color = isDark ? darkColor[chunk.modality] : lightColor[chunk.modality]

  return (
    <div className={`rounded-lg border p-3 text-xs ${color}`}>
      <div className="flex items-center gap-2 mb-1.5">
        <span className="font-mono text-base leading-none">{modalityIcon[chunk.modality]}</span>
        <span className="font-semibold truncate flex-1">{chunk.source}</span>
        <span className="shrink-0 opacity-60">
          {chunk.page ? `p.${chunk.page}` : chunk.timestamp ? `@${chunk.timestamp}` : ''}
        </span>
        <span className={`shrink-0 rounded px-1.5 py-0.5 font-mono ${isDark ? 'bg-white/10' : 'bg-black/5'}`}>[{index + 1}]</span>
      </div>
      <p className={`leading-relaxed line-clamp-3 ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>{chunk.text}</p>
    </div>
  )
}
