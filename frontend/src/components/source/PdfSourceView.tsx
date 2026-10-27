import { useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import type { EvidenceChunk } from '../../types'
import { highlightSourceText, MAX_PREVIEW_BYTES, resolveSourceQuotePage } from '../../utils/sourceHighlight'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
const PDF_OPTIONS = { enableXfa: false, isEvalSupported: false, useSystemFonts: true, useWasm: false, disableAutoFetch: true }

export default function PdfSourceView({ original, chunk, quote }: { original: Blob; chunk: EvidenceChunk; quote: string | null }) {
  const [data, setData] = useState<{ data: Uint8Array }>()
  const [page, setPage] = useState(Math.max(1, chunk.page ?? 1))
  const [citedPage, setCitedPage] = useState(page)
  const [total, setTotal] = useState(0)
  const [width, setWidth] = useState(700)
  const [status, setStatus] = useState('Opening the cited PDF page…')
  const [location, setLocation] = useState<'locating' | 'ready' | 'ambiguous'>(quote && (chunk.page_end ?? 1) > (chunk.page ?? 1) ? 'locating' : 'ready')
  const [error, setError] = useState<string | null>(null)
  const host = useRef<HTMLDivElement>(null)
  const pageHost = useRef<HTMLDivElement>(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    Promise.resolve().then(() => {
      if (original.size > MAX_PREVIEW_BYTES) throw new Error('This PDF is too large to preview. Open the original in a new tab.')
      return original.arrayBuffer()
    }).then((buffer) => {
      if (buffer.byteLength > MAX_PREVIEW_BYTES) throw new Error('This PDF is too large to preview. Open the original in a new tab.')
      if (alive.current) setData({ data: new Uint8Array(buffer) })
    }).catch((cause) => { if (alive.current) setError(cause instanceof Error ? cause.message : 'Could not open this PDF.') })
    return () => { alive.current = false }
  }, [original])

  useEffect(() => {
    if (!host.current) return
    const observer = new ResizeObserver((entries) => setWidth(Math.max(200, Math.min(900, Math.floor(entries[0].contentRect.width) - 8))))
    observer.observe(host.current)
    return () => observer.disconnect()
  }, [])

  const loaded = async (pdf: PDFDocumentProxy) => {
    const start = Math.min(pdf.numPages, Math.max(1, chunk.page ?? 1))
    const reportedEnd = Math.min(pdf.numPages, Math.max(start, chunk.page_end ?? start))
    const end = Math.min(reportedEnd, start + 15)
    let target = start
    // Search only the backend-reported source span for older bundles without exact quote-page maps.
    let ambiguous = false
    if (quote && reportedEnd > start) {
      const pages: Array<{ page: number; text: string }> = []
      if (reportedEnd <= end) for (let candidate = start; candidate <= end; candidate++) {
        const sourcePage = await pdf.getPage(candidate)
        const content = await sourcePage.getTextContent()
        pages.push({ page: candidate, text: content.items.map((item) => 'str' in item ? item.str : '').join(' ') })
      }
      const resolved = resolveSourceQuotePage(pages, quote, chunk.text)
      ambiguous = resolved === null
      target = resolved ?? start
    }
    if (!alive.current) return
    setLocation(ambiguous ? 'ambiguous' : 'ready')
    if (ambiguous) setStatus('The quote could not be uniquely located within the reported source pages. Read the supporting quote beside the original and use the page controls.')
    setTotal(pdf.numPages); setPage(target); setCitedPage(target)
  }

  const highlight = () => {
    if (location !== 'ready') return
    const layer = pageHost.current?.querySelector<HTMLElement>('.react-pdf__Page__textContent')
    if (!layer) return
    const marks = highlightSourceText(layer, quote || chunk.text, true, chunk.text)
    if (marks.length) {
      marks[0].scrollIntoView({ block: 'center', inline: 'nearest' })
      setStatus(`${quote ? 'Supporting quote' : 'Cited passage'} highlighted on PDF page ${page}.`)
    } else setStatus(quote ? `PDF page ${page}. An exact selectable-text match was not found on this page; the supporting quote remains visible beside it.` : `PDF page ${page}. Read the cited passage beside the original page.`)
  }

  return <div ref={host} className="min-w-0">
    <div className="sticky top-0 z-10 mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-[var(--lumin-border)] bg-[var(--lumin-paper)] p-2 text-xs">
      <button disabled={page <= 1} onClick={() => setPage((value) => value - 1)} className="rounded-lg border border-[var(--lumin-border)] px-2 py-1 disabled:opacity-40">Previous page</button>
      <output aria-label="PDF page number">Page {page}{total ? ` of ${total}` : ''}</output>
      <button disabled={!total || page >= total} onClick={() => setPage((value) => value + 1)} className="rounded-lg border border-[var(--lumin-border)] px-2 py-1 disabled:opacity-40">Next page</button>
      {page !== citedPage && <button onClick={() => setPage(citedPage)} className="underline ml-auto">Back to cited page</button>}
    </div>
    <p role="status" className="mb-3 text-xs text-[var(--lumin-muted)]">{status}</p>
    {error ? <p role="alert" className="text-sm text-red-500">{error}</p> : data ? <Document file={data} options={PDF_OPTIONS} onLoadSuccess={(pdf) => { loaded(pdf).catch(() => setStatus('The cited page is open. Use the page controls to read the source.')) }} onLoadError={(cause) => setError(cause.message)} onPassword={() => setError('This PDF is password-protected. Open the original with your PDF reader.')} loading={<p role="status">Loading original PDF…</p>}>
      <div ref={pageHost} role="region" aria-label={`Original PDF page ${page}`} className="w-fit mx-auto shadow-lg"><Page key={`${page}:${location}`} pageNumber={page} width={width} devicePixelRatio={Math.min(window.devicePixelRatio || 1, 2)} renderAnnotationLayer={false} renderTextLayer onRenderTextLayerSuccess={highlight} onRenderTextLayerError={() => setStatus(`PDF page ${page} is visible, but selectable text is unavailable.`)} loading={<p role="status" className="p-6">Rendering original page…</p>} /></div>
    </Document> : <p role="status">Reading the original PDF from this device…</p>}
  </div>
}
