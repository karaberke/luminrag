import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import type { EvidenceChunk } from '../../types'
import { highlightSourceText, MAX_PREVIEW_BYTES, resolveSourceQuotePage } from '../../utils/sourceHighlight'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
const PDF_OPTIONS = { enableXfa: false, isEvalSupported: false, useSystemFonts: true, useWasm: false, disableAutoFetch: true }

type PageOutcome = 'text' | 'no-text' | 'failed'
type PageTarget = { page: number; width: number }
type SettlePage = (page: number, width: number, outcome: PageOutcome, layer: HTMLElement | null) => void

/** One rendered page. It settles once both its canvas and text layer finish, so the swap never shows a blank bitmap. */
function PdfPageSlot({ page, width, current, onSettled }: PageTarget & { current: boolean; onSettled: SettlePage }) {
  const element = useRef<HTMLDivElement>(null)
  const progress = useRef<{ canvas?: boolean; text?: PageOutcome }>({})
  // react-pdf leaves a failed canvas hidden, so a failure swaps the page for a visible card instead of a blank one.
  const [broken, setBroken] = useState(false)
  const report = useCallback((update: { canvas?: boolean; text?: PageOutcome }) => {
    // A late callback from an unmounted slot for the same page (2 → 1 → 2) must not settle its replacement; failure is final.
    if (!element.current?.isConnected || progress.current.text === 'failed') return
    const state = Object.assign(progress.current, update)
    const outcome = state.text === 'failed' ? 'failed' : state.canvas ? state.text : undefined
    if (outcome === 'failed') setBroken(true)
    if (outcome) onSettled(page, width, outcome, outcome === 'failed' ? null : element.current.querySelector<HTMLElement>('.react-pdf__Page__textContent'))
  }, [onSettled, page, width])
  // react-pdf rebuilds the text layer (clearing highlights) whenever these callbacks change identity.
  const rendered = useCallback((proxy: { pageNumber: number }) => { if (proxy.pageNumber === page) report({ canvas: true }) }, [page, report])
  const textRendered = useCallback(() => report({ text: 'text' }), [report])
  const textFailed = useCallback(() => report({ text: 'no-text' }), [report])
  const failed = useCallback(() => report({ text: 'failed' }), [report])
  return <div ref={element} data-pdf-current={current || undefined} className={current ? 'shadow-lg' : 'invisible absolute left-0 top-0'}>
    {broken ? <p data-pdf-failed-page={page} style={{ width }} className="bg-[var(--lumin-paper)] p-6 text-sm">PDF page {page} could not be rendered. Open the original to read it.</p> : <Page pageNumber={page} width={width} devicePixelRatio={Math.min(window.devicePixelRatio || 1, 2)} renderAnnotationLayer={false} renderTextLayer onRenderSuccess={rendered} onRenderError={failed} onLoadError={failed} onGetTextError={textFailed} onRenderTextLayerSuccess={textRendered} onRenderTextLayerError={textFailed} loading={<p className="p-6">Rendering original page…</p>} error={<p className="p-6">This page could not be rendered.</p>} />}
  </div>
}

export default function PdfSourceView({ original, chunk, quote }: { original: Blob; chunk: EvidenceChunk; quote: string | null }) {
  const passage = chunk.text
  const [data, setData] = useState<{ data: Uint8Array }>()
  const [page, setPage] = useState(Math.max(1, chunk.page ?? 1))
  const [citedPage, setCitedPage] = useState(page)
  const [shown, setShown] = useState<PageTarget | null>(null)
  const [total, setTotal] = useState(0)
  const [width, setWidth] = useState(700)
  const [status, setStatus] = useState('Opening the cited PDF page…')
  const [location, setLocation] = useState<'locating' | 'ready' | 'ambiguous'>(quote && (chunk.page_end ?? 1) > (chunk.page ?? 1) ? 'locating' : 'ready')
  const [error, setError] = useState<string | null>(null)
  const host = useRef<HTMLDivElement>(null)
  const pageHost = useRef<HTMLDivElement>(null)
  const alive = useRef(true)
  // The bytes handed to <Document>; quote-location work for an older original must not apply to a newer one.
  const opened = useRef<{ data: Uint8Array } | null>(null)
  const requested = useRef<PageTarget>({ page, width })
  // Only opening the source or "Back to cited page" may scroll; ordinary paging keeps the reader's position.
  const scrollToCitation = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])

  useEffect(() => {
    let active = true
    Promise.resolve().then(() => {
      if (original.size > MAX_PREVIEW_BYTES) throw new Error('This PDF is too large to preview. Open the original in a new tab.')
      return original.arrayBuffer()
    }).then((buffer) => {
      if (buffer.byteLength > MAX_PREVIEW_BYTES) throw new Error('This PDF is too large to preview. Open the original in a new tab.')
      if (!active) return
      opened.current = { data: new Uint8Array(buffer) }
      setData(opened.current)
    }).catch((cause) => { if (active) setError(cause instanceof Error ? cause.message : 'Could not open this PDF.') })
    return () => { active = false; opened.current = null }
  }, [original])

  useEffect(() => {
    if (!host.current) return
    const observer = new ResizeObserver(([entry]) => {
      const available = Math.floor(entry.contentRect.width)
      // A hidden pane reports zero width; keep the last real width instead of rendering a tiny page.
      if (available > 0) setWidth(Math.max(200, Math.min(900, available - 8)))
    })
    observer.observe(host.current)
    return () => observer.disconnect()
  }, [])

  useLayoutEffect(() => { requested.current = { page, width } }, [page, width])

  useLayoutEffect(() => {
    if (!scrollToCitation.current || shown?.page !== page || page !== citedPage) return
    // Highlights are applied before promotion, so a missing mark now will not appear later.
    pageHost.current?.querySelector<HTMLElement>('[data-pdf-current] mark[data-lumin-source-match]')?.scrollIntoView({ block: 'center', inline: 'nearest' })
    scrollToCitation.current = false
  }, [shown, page, citedPage])

  const loaded = async (pdf: PDFDocumentProxy) => {
    const file = opened.current
    const start = Math.min(pdf.numPages, Math.max(1, chunk.page ?? 1))
    const reportedEnd = Math.min(pdf.numPages, Math.max(start, chunk.page_end ?? start))
    const end = Math.min(reportedEnd, start + 15)
    let target = start
    let unresolved: string | null = null
    try {
      // Search only the backend-reported source span for older bundles without exact quote-page maps.
      if (quote && reportedEnd > start) {
        const pages: Array<{ page: number; text: string }> = []
        if (reportedEnd <= end) for (let candidate = start; candidate <= end; candidate++) {
          const sourcePage = await pdf.getPage(candidate)
          const content = await sourcePage.getTextContent()
          pages.push({ page: candidate, text: content.items.map((item) => 'str' in item ? item.str : '').join(' ') })
        }
        const resolved = resolveSourceQuotePage(pages, quote, passage)
        if (resolved === null) unresolved = 'The quote could not be uniquely located within the reported source pages. Read the supporting quote beside the original and use the page controls.'
        target = resolved ?? start
      }
    } catch { unresolved = 'The cited page is open. Use the page controls to read the source.' }
    if (!alive.current || !file || opened.current !== file) return
    setLocation(unresolved ? 'ambiguous' : 'ready')
    if (unresolved) setStatus(unresolved)
    setTotal(pdf.numPages); setPage(target); setCitedPage(target)
  }

  // Page slots mount only after `loaded`, so these dependencies stay fixed while pages render.
  const settle = useCallback<SettlePage>((settledPage, settledWidth, outcome, layer) => {
    // A superseded page may finish late; it must not replace or relabel the requested one.
    if (requested.current.page !== settledPage || requested.current.width !== settledWidth) return
    setShown((value) => value?.page === settledPage && value.width === settledWidth ? value : { page: settledPage, width: settledWidth })
    if (outcome === 'failed') setStatus(`PDF page ${settledPage} could not be rendered. Open the original to read it.`)
    else if (location !== 'ready') return
    else if (outcome === 'no-text' || !layer) setStatus(`PDF page ${settledPage} is visible, but selectable text is unavailable.`)
    else if (settledPage !== citedPage) setStatus(`PDF page ${settledPage}. The ${quote ? 'supporting quote' : 'cited passage'} is on page ${citedPage}.`)
    else if (highlightSourceText(layer, quote || passage, true, passage).length) setStatus(`${quote ? 'Supporting quote' : 'Cited passage'} highlighted on PDF page ${settledPage}.`)
    else setStatus(quote ? `PDF page ${settledPage}. An exact selectable-text match was not found on this page; the supporting quote remains visible beside it.` : `PDF page ${settledPage}. Read the cited passage beside the original page.`)
  }, [citedPage, location, quote, passage])

  const navigate = (next: (value: number) => number, citation = false) => {
    scrollToCitation.current = citation
    setPage(next)
  }

  // Keep the displayed page until the requested page (or width) has fully rendered: at most two Page instances.
  const pending = shown !== null && (shown.page !== page || shown.width !== width)
  // Until `loaded` has set the total and cited page, a click would be silently overwritten.
  const navigable = total > 0 && location !== 'locating'
  const slots: PageTarget[] = shown && pending ? [shown, { page, width }] : [{ page, width }]

  return <div ref={host} className="min-w-0">
    <div className="sticky top-0 z-10 mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-[var(--lumin-border)] bg-[var(--lumin-paper)] p-2 text-xs">
      <button disabled={!navigable || page <= 1} onClick={() => navigate((value) => value - 1)} className="rounded-lg border border-[var(--lumin-border)] px-2 py-1 disabled:opacity-40">Previous page</button>
      <output aria-label="PDF page number" aria-live="off">Page {page}{total ? ` of ${total}` : ''}</output>
      <button disabled={!navigable || page >= total} onClick={() => navigate((value) => value + 1)} className="rounded-lg border border-[var(--lumin-border)] px-2 py-1 disabled:opacity-40">Next page</button>
      {page !== citedPage && <button onClick={() => { navigate(() => citedPage, true); pageHost.current?.focus({ preventScroll: true }) }} className="underline ml-auto">Back to cited page</button>}
    </div>
    <p role="status" className="mb-3 text-xs text-[var(--lumin-muted)]">{pending && shown.page !== page ? `Rendering PDF page ${page}…` : status}</p>
    {error ? <p role="alert" className="text-sm text-red-500">{error}</p> : data ? <Document file={data} options={PDF_OPTIONS} suspense={false} onLoadSuccess={(pdf) => { loaded(pdf) }} onLoadError={(cause) => setError(cause.message)} onPassword={() => setError('This PDF is password-protected. Open the original with your PDF reader.')} loading={<p role="status">Loading original PDF…</p>} error={<p className="text-sm">Could not open this PDF.</p>}>
      <div ref={pageHost} tabIndex={-1} role="region" aria-label={`Original PDF page ${shown?.page ?? page}`} aria-busy={!shown || pending} className={`relative mx-auto w-fit ${shown ? '' : 'min-h-96'}`}>
        {total ? slots.map((slot) => <PdfPageSlot key={`${slot.page}@${slot.width}`} page={slot.page} width={slot.width} current={!shown || (slot.page === shown.page && slot.width === shown.width)} onSettled={settle} />) : <p className="p-6 text-sm">Locating the cited page…</p>}
      </div>
    </Document> : <p role="status">Reading the original PDF from this device…</p>}
  </div>
}
