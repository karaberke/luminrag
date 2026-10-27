import { zipSync } from 'fflate'
import { readBoundedArchive, type ArchiveLimits } from './archive'

interface NormalizedText { text: string; starts: number[]; ends: number[] }

/** Match extraction whitespace, ligatures and typographic punctuation without altering the source DOM. */
function normalizeWithOffsets(input: string): NormalizedText {
  let text = '', offset = 0
  const starts: number[] = [], ends: number[] = []
  const formatting = new Set<number>()
  // Word extraction can carry Markdown emphasis that the original styled runs do not.
  // Strip only paired markers; underscores inside identifiers remain ordinary text.
  const opened = new Map<string, { start: number; end: number; line: number }>()
  let line = 0, previous = 0
  for (const match of input.matchAll(/\*\*|__|~~|`{1,3}|\*|_/g)) {
    const start = match.index, end = start + match[0].length
    for (let index = previous; index < start; index++) if (input[index] === '\n') line++
    previous = end
    const before = input[start - 1] ?? '', after = input[end] ?? '', opening = opened.get(match[0])
    if (opening && opening.line === line && before && !/\s/u.test(before) && !/[\p{L}\p{N}_]/u.test(after)) {
      opened.delete(match[0])
      for (let index = opening.start; index < opening.end; index++) formatting.add(index)
      for (let index = start; index < end; index++) formatting.add(index)
    } else if (after && !/\s/u.test(after) && !/[\p{L}\p{N}_]/u.test(before)) opened.set(match[0], { start, end, line })
  }
  for (const character of input) {
    const end = offset + character.length
    if (formatting.has(offset)) { offset = end; continue }
    const normalized = character.normalize('NFKC').toLowerCase().replace(/[\s\u00ad\u200b-\u200d\ufeff]/gu, '').replace(/[‘’]/gu, "'").replace(/[“”]/gu, '"').replace(/[‐‑‒–—−]/gu, '-')
    for (let index = 0; index < normalized.length; index++) { text += normalized[index]; starts.push(offset); ends.push(end) }
    offset = end
  }
  return { text, starts, ends }
}

export function findSourceTextRange(text: string, quote: string, context?: string): { start: number; end: number } | null {
  const source = normalizeWithOffsets(text), needle = normalizeWithOffsets(quote).text
  if (!needle) return null
  let start = source.text.indexOf(needle)
  if (start >= 0 && source.text.indexOf(needle, start + 1) >= 0) {
    const passage = context ? normalizeWithOffsets(context).text : ''
    const passageStart = passage ? source.text.indexOf(passage) : -1
    if (passageStart < 0 || source.text.indexOf(passage, passageStart + 1) >= 0) return null
    const within = passage.indexOf(needle)
    if (within < 0 || passage.indexOf(needle, within + 1) >= 0) return null
    start = passageStart + within
  }
  return start < 0 ? null : { start: source.starts[start], end: source.ends[start + needle.length - 1] }
}

/** Older page ranges must not turn the first repeated sentence into an invented exact page. */
export function resolveSourceQuotePage(pages: Array<{ page: number; text: string }>, quote: string, passage: string): number | null {
  const candidates = pages.filter(({ text }) => findSourceTextRange(text, quote, passage))
  if (candidates.length === 1) return candidates[0].page
  const contextual = candidates.filter(({ text }) => findSourceTextRange(text, passage))
  return contextual.length === 1 ? contextual[0].page : null
}

/** Mark only text nodes, never HTML strings. Supports PDF text spans and styled Word runs. */
export function highlightSourceText(root: HTMLElement, quote: string, transparentText = false, context?: string): HTMLElement[] {
  for (const mark of root.querySelectorAll('mark[data-lumin-source-match]')) mark.replaceWith(...mark.childNodes)
  root.normalize()
  const walker = root.ownerDocument.createTreeWalker(root, 4)
  const segments: Array<{ node: Text; start: number; end: number }> = []
  let text = '', current: Node | null
  while ((current = walker.nextNode())) {
    if (current.parentElement?.closest('script, style, [aria-hidden="true"]')) continue
    const value = current.textContent ?? ''
    segments.push({ node: current as Text, start: text.length, end: text.length + value.length })
    text += value
  }
  const match = findSourceTextRange(text, quote, context)
  if (!match) return []
  const marks: HTMLElement[] = []
  for (const segment of segments) {
    const start = Math.max(match.start, segment.start), end = Math.min(match.end, segment.end)
    if (end <= start) continue
    const range = root.ownerDocument.createRange()
    range.setStart(segment.node, start - segment.start)
    range.setEnd(segment.node, end - segment.start)
    const mark = root.ownerDocument.createElement('mark')
    mark.dataset.luminSourceMatch = 'true'
    mark.style.backgroundColor = 'rgba(245, 196, 65, 0.6)'
    mark.style.color = transparentText ? 'transparent' : 'inherit'
    mark.style.borderRadius = '2px'
    mark.style.padding = '0'
    range.surroundContents(mark)
    marks.push(mark)
  }
  return marks
}

export const MAX_PREVIEW_BYTES = 50 * 1024 * 1024
const WORD_ARCHIVE_LIMITS: ArchiveLimits = {
  maxArchiveBytes: MAX_PREVIEW_BYTES, maxExpandedBytes: 100 * 1024 * 1024,
  maxEntryBytes: 100 * 1024 * 1024, maxEntries: 10_000,
  entryLimit: (name) => /\.(xml|rels)$/i.test(name) ? 16 * 1024 * 1024 : 100 * 1024 * 1024,
  limitMessage: 'This Word file exceeds the safe preview size limit.',
  pathMessage: 'This Word file contains an unsafe archive path.',
}
/** Return a normalized archive containing only the actually validated, bounded Word entries. */
export function validateWordPreview(bytes: Uint8Array): Uint8Array {
  if (bytes.byteLength > MAX_PREVIEW_BYTES) throw new Error('This Word file is too large to preview. Open the downloaded original instead.')
  const entries = readBoundedArchive(bytes, WORD_ARCHIVE_LIMITS)
  if (!entries['word/document.xml']) throw new Error('This file does not contain a Word document.')
  return zipSync(entries, { level: 0 })
}
