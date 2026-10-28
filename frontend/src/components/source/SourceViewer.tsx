import { lazy, Suspense, useEffect, useId, useRef, useState } from 'react'
import type { EvidenceChunk } from '../../types'
import { originalForEvidence, cachedSourceUrl, type RawFileRecord } from '../../storage/db'
import { evidenceWithContext, sourceDisplayName, sourceLocation, sourceUrl, validatedEvidenceQuote } from '../../utils/source'
import { errorMessage } from '../../utils/api'
import RichText from '../LazyRichText'

const PdfSourceView = lazy(() => import('./PdfSourceView'))
const WordSourceView = lazy(() => import('./WordSourceView'))

export default function SourceViewer({ chunk: originalChunk, onClose }: { chunk: EvidenceChunk; onClose: () => void }) {
  const chunk = evidenceWithContext(originalChunk)
  const titleId = useId()
  const dialog = useRef<HTMLDialogElement>(null)
  // A record read for another source must never be shown (or previewed) for this one.
  const sourceKey = `${chunk.source}\u0000${chunk.source_filename ?? ''}`
  const [opened, setOpened] = useState<{ key: string; record: RawFileRecord | null; error: string | null }>()
  const original = opened?.key === sourceKey ? opened.record : undefined
  const error = opened?.key === sourceKey ? opened.error : null
  const filename = sourceDisplayName(chunk)
  const location = sourceLocation(chunk)
  const quote = validatedEvidenceQuote(chunk)
  const extension = (original?.name ?? filename).split('.').at(-1)?.toLowerCase()
  const originalURL = original && (cachedSourceUrl(chunk.source) || cachedSourceUrl(original.name))

  useEffect(() => {
    const element = dialog.current
    if (!element) return
    const previous = document.activeElement as HTMLElement | null
    if (!element.open) element.showModal()
    element.querySelector<HTMLButtonElement>('button')?.focus()
    return () => { element.close(); previous?.focus?.({ preventScroll: true }) }
  }, [])

  useEffect(() => {
    let cancelled = false
    originalForEvidence(chunk.source, chunk.source_filename).then((record) => { if (!cancelled) setOpened({ key: sourceKey, record: record ?? null, error: null }) })
      .catch((cause) => { if (!cancelled) setOpened({ key: sourceKey, record: null, error: errorMessage(cause) }) })
    return () => { cancelled = true }
  }, [chunk.source, chunk.source_filename, sourceKey])

  return <dialog ref={dialog} aria-modal="true" aria-labelledby={titleId} onCancel={(event) => { event.preventDefault(); onClose() }} className="lumin-modal-card m-auto p-0 w-[min(1160px,calc(100vw-24px))] max-w-none h-[min(92dvh,1000px)] max-h-none overflow-hidden text-slate-800 backdrop:bg-black/60">
    <div className="h-full flex flex-col text-[var(--lumin-ink)]">
      <header className="flex items-center justify-between gap-3 px-5 py-3 border-b border-[var(--lumin-border)] shrink-0">
        <div className="min-w-0"><h2 id={titleId} className="font-bold text-base truncate">{filename}</h2><p className="text-xs text-[var(--lumin-muted)]">{location || 'Original source and supporting passage'}</p></div>
        <div className="flex items-center gap-3 shrink-0">{originalURL && <a className="text-xs underline" href={sourceUrl(original?.name ?? chunk.source, chunk.modality, chunk.page, chunk.timestamp)} target="_blank" rel="noreferrer">Open original ↗</a>}<button type="button" aria-label="Close source viewer" onClick={onClose} className="rounded-lg border border-[var(--lumin-border)] px-2 py-1 text-sm">Close ×</button></div>
      </header>
      <div className="min-h-0 flex-1 flex flex-col lg:flex-row overflow-hidden">
        <section aria-label="Original document" className="min-h-0 min-w-0 flex-1 overflow-auto [scrollbar-gutter:stable] bg-[var(--lumin-paper-soft)] p-3">
          {original === undefined ? <p role="status" className="p-5 text-sm">Opening the original saved on this device…</p> : original && extension === 'pdf' ? <Suspense fallback={<p role="status" className="p-5 text-sm">Loading PDF viewer…</p>}><PdfSourceView key={`${sourceKey}\u0000${chunk.id}`} original={original.blob} chunk={chunk} quote={quote} /></Suspense> : original && extension === 'docx' ? <Suspense fallback={<p role="status" className="p-5 text-sm">Loading Word preview…</p>}><WordSourceView original={original.blob} quote={quote} passage={chunk.text} /></Suspense> : <div className="rounded-xl border border-[var(--lumin-border)] bg-[var(--lumin-paper)] p-5">
            <h3 className="text-sm font-bold">{chunk.modality === 'note' ? 'Your note' : chunk.slide_number ? `Extracted text from slide ${chunk.slide_number}` : 'Extracted source passage'}</h3>
            {!original && chunk.modality !== 'note' && <p role="status" className="text-xs text-[var(--lumin-muted)] my-3">The original file is not saved on this device. Import a backup with original files, or upload it again. The cited passage is available below.</p>}
            {original && <p className="text-xs text-[var(--lumin-muted)] my-3">This format is shown as extracted text. Open the original to see its formatting.</p>}
            {error && <p role="alert" className="text-xs text-red-500 my-2">{error}</p>}
            <RichText text={chunk.text} className="text-sm leading-relaxed whitespace-pre-wrap mt-4" />
          </div>}
        </section>
        <aside aria-label="Supporting passage" className="shrink-0 lg:w-80 max-h-[35%] lg:max-h-none overflow-auto border-t lg:border-t-0 lg:border-l border-[var(--lumin-border)] p-4 bg-[var(--lumin-paper)]">
          <h3 className="font-bold text-sm mb-2">{quote ? 'Supporting quote' : 'Cited passage'}</h3>
          {quote ? <blockquote className="border-l-4 border-amber-400 pl-3 text-sm leading-relaxed whitespace-pre-wrap">{quote}</blockquote> : <p className="text-xs text-[var(--lumin-muted)] mb-3">This saved answer includes a passage rather than a specific supporting quote.</p>}
          <details className="mt-4" open={!quote}><summary className="cursor-pointer text-xs font-bold">Full cited passage</summary><RichText text={chunk.text} className="mt-3 text-xs leading-relaxed whitespace-pre-wrap" /></details>
        </aside>
      </div>
    </div>
  </dialog>
}
