import type { EvidenceChunk } from '../types'
import { API_BASE, apiUrl } from './api'
import { cachedAssetUrl, cachedPassage, cachedSourceName, cachedSourceUrl, isCachedBlobUrl } from '../storage/db'

export function openSourceEvidence(chunk: EvidenceChunk): void {
  window.dispatchEvent(new CustomEvent('lumin-open-source', { detail: evidenceWithContext(chunk) }))
}

/** Offsets from Python are Unicode code points; quotes are never inferred by the viewer. */
export function validatedEvidenceQuote(chunk: EvidenceChunk): string | null {
  if (!chunk.quote || !chunk.quote.trim()) return null
  if (chunk.quote_start != null || chunk.quote_end != null) {
    const characters = Array.from(chunk.text)
    return Number.isSafeInteger(chunk.quote_start) && Number.isSafeInteger(chunk.quote_end) && chunk.quote_start! >= 0 && chunk.quote_end! > chunk.quote_start! && chunk.quote_end! <= characters.length && characters.slice(chunk.quote_start!, chunk.quote_end!).join('') === chunk.quote ? chunk.quote : null
  }
  return chunk.text.includes(chunk.quote) ? chunk.quote : null
}

/** Untrusted markdown must not make automatic requests to external image hosts. */
export function trustedMarkdownImageUrl(source: string | undefined): string | null {
  if (!source) return null
  const cached = cachedAssetUrl(source)
  if (cached) return cached
  if (source.startsWith('blob:')) return isCachedBlobUrl(source) ? source : null
  const origin = typeof window === 'undefined' ? 'http://localhost' : window.location.origin
  try {
    const backend = new URL(`${API_BASE}/`, origin)
    const candidate = new URL(source, backend)
    if (candidate.origin !== backend.origin || !['http:', 'https:'].includes(candidate.protocol)) return null
    const allowedPaths = ['/static/illustrations/', '/api/source/'].map((path) => new URL(apiUrl(path), origin).pathname)
    return allowedPaths.some((path) => candidate.pathname.startsWith(path)) ? candidate.href : null
  } catch {
    return null
  }
}

/** Parse "MM:SS" or "HH:MM:SS" → seconds. Returns null if unrecognised. */
export function parseTimestamp(ts: string | null | undefined): number | null {
  if (!ts) return null
  const parts = ts.split(':').map((p) => parseInt(p, 10))
  if (parts.some(Number.isNaN)) return null
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  return null
}

/**
 * Build a deep-linking URL for a source file: #page=N for PDFs, #t=SSs for AV.
 * Opened as a plain navigation, so it is authenticated by the session cookie.
 */
export function sourceUrl(
  source: string,
  modality: string,
  page: number | null | undefined,
  timestamp: string | null | undefined,
): string {
  const base = cachedSourceUrl(source) ?? apiUrl(`/api/source/${encodeURIComponent(source)}`)
  if ((modality === 'pdf' || modality === 'slide') && page != null) {
    return `${base}#page=${page}`
  }
  if (modality === 'video' || modality === 'audio') {
    const secs = parseTimestamp(timestamp)
    if (secs != null) return `${base}#t=${secs}`
  }
  return base
}

/** Convenience wrapper that unpacks an EvidenceChunk. */
export function chunkSourceUrl(chunk: EvidenceChunk): string {
  return sourceUrl(chunk.source, chunk.modality, chunk.page, chunk.timestamp)
}

/** Restore a complete passage and location from the browser bundle for older saved answers. */
export function evidenceWithContext(chunk: EvidenceChunk): EvidenceChunk {
  const cached = cachedPassage(chunk.id)
  if (!cached || cached.source_id !== chunk.source) return chunk
  // Current responses already carry full text and authoritative provenance.
  // Only repair legacy responses, and never replace a changed passage by ID alone.
  if ('source_filename' in chunk || 'section_title' in chunk || !cached.text.startsWith(chunk.text)) return chunk
  const metadata = cached.metadata ?? {}
  const text = chunk.text.length === 500 && cached.text.length > 500 ? cached.text.slice(0, 100_000) : chunk.text
  return { ...chunk, text,
    source_filename: chunk.source_filename ?? (typeof metadata.original_filename === 'string' ? metadata.original_filename : undefined),
    title: chunk.title ?? (typeof metadata.title === 'string' ? metadata.title : undefined),
    section_title: chunk.section_title ?? (typeof metadata.section_title === 'string' ? metadata.section_title : undefined),
    page: chunk.page ?? positiveLocation(metadata.page_number ?? metadata.page_start),
    page_end: chunk.page_end ?? positiveLocation(metadata.page_end),
    slide_number: chunk.slide_number ?? positiveLocation(metadata.slide_number),
  }
}

function positiveLocation(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

export function sourceDisplayName(chunk: EvidenceChunk): string {
  return chunk.source_filename || cachedSourceName(chunk.source) || chunk.title || chunk.source
}

export function sourceLocation(chunk: EvidenceChunk): string {
  const parts: string[] = []
  if (chunk.slide_number) parts.push(`Slide ${chunk.slide_number}`)
  else if (chunk.page) parts.push(chunk.page_end && chunk.page_end > chunk.page ? `pp. ${chunk.page}–${chunk.page_end}` : `${chunk.modality === 'slide' ? 'Slide' : 'p.'} ${chunk.page}`)
  if (chunk.timestamp) parts.push(`@ ${chunk.timestamp}`)
  if (chunk.section_title) parts.push(chunk.section_title)
  return parts.join(' · ')
}
