import type { EvidenceChunk } from '../types'

/** Parse "MM:SS" or "HH:MM:SS" → seconds. Returns null if unrecognised. */
export function parseTimestamp(ts: string | undefined): number | null {
  if (!ts) return null
  const parts = ts.split(':').map((p) => parseInt(p, 10))
  if (parts.some(Number.isNaN)) return null
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  return null
}

/** Build a deep-linking URL for a source file: #page=N for PDFs, #t=SSs for AV. */
export function sourceUrl(
  apiBase: string,
  source: string,
  modality: string,
  page: number | undefined,
  timestamp: string | undefined,
): string {
  const base = `${apiBase}/api/source/${encodeURIComponent(source)}`
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
export function chunkSourceUrl(apiBase: string, chunk: EvidenceChunk): string {
  return sourceUrl(apiBase, chunk.source, chunk.modality, chunk.page, chunk.timestamp)
}
