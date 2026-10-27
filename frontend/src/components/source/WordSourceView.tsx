import { useEffect, useRef, useState } from 'react'
import { renderAsync } from 'docx-preview'
import { highlightSourceText, MAX_PREVIEW_BYTES, validateWordPreview } from '../../utils/sourceHighlight'

export const WORD_PREVIEW_DOCUMENT = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src blob: data:; font-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"><style>html{background:#ebe9e0}body{margin:0;color:#111;font-family:Arial,sans-serif}#document>.lumin-docx-wrapper{background:#ebe9e0;padding:16px}section.lumin-docx{box-shadow:0 2px 10px #0002;margin-bottom:16px}mark[data-lumin-source-match]{background:#f4cb64}a{color:inherit;pointer-events:none}</style></head><body><div id="styles"></div><div id="document"></div></body></html>`

export default function WordSourceView({ original, quote, passage }: { original: Blob; quote: string | null; passage: string }) {
  const frame = useRef<HTMLIFrameElement>(null)
  const renderedDocument = useRef<Document | null>(null)
  const resizeObserver = useRef<ResizeObserver | null>(null)
  const resize = useRef<() => void>(() => {})
  const fit = useRef(true)
  const [fitWidth, setFitWidth] = useState(true)
  const [status, setStatus] = useState('Rendering the original Word document…')
  const [error, setError] = useState<string | null>(null)
  useEffect(() => () => resizeObserver.current?.disconnect(), [])

  const render = async () => {
    const document = frame.current?.contentDocument
    if (!document || document === renderedDocument.current) return
    renderedDocument.current = document
    try {
      if (original.size > MAX_PREVIEW_BYTES) throw new Error('This Word file is too large to preview. Open the downloaded original instead.')
      const buffer = await original.arrayBuffer()
      const validatedArchive = validateWordPreview(new Uint8Array(buffer))
      const body = document.getElementById('document'), styles = document.getElementById('styles')
      if (!body || !styles) throw new Error('The secure Word preview could not start.')
      body.addEventListener('click', (event) => { if ((event.target as Element)?.closest('a')) event.preventDefault() }, true)
      await renderAsync(validatedArchive.buffer, body, styles, { className: 'lumin-docx', inWrapper: true, ignoreFonts: true,
        breakPages: true, ignoreLastRenderedPageBreak: false, renderAltChunks: false, renderChanges: false,
        renderComments: false, renderFootnotes: true, renderEndnotes: true, useBase64URL: true })
      // Defence in depth: the script-disabled frame also blocks remote resources and navigation.
      for (const element of body.querySelectorAll('script, iframe, object, embed, form, input, button')) element.remove()
      for (const element of body.querySelectorAll('*')) for (const attribute of [...element.attributes]) {
        if (/^on/i.test(attribute.name) || ['href', 'xlink:href', 'srcset'].includes(attribute.name)) element.removeAttribute(attribute.name)
        if (attribute.name === 'src' && !/^(?:data:image\/|blob:)/i.test(attribute.value)) element.removeAttribute(attribute.name)
      }
      const pages = [...body.querySelectorAll('section.lumin-docx')]
      pages.forEach((page, index) => { page.setAttribute('role', 'region'); page.setAttribute('aria-label', `Word preview page ${index + 1}`) })
      const wrapper = body.querySelector<HTMLElement>('.lumin-docx-wrapper')
      if (wrapper && frame.current) {
        const originalWidth = Math.max(...pages.map((page) => (page as HTMLElement).offsetWidth), 300) + 32
        resize.current = () => {
          wrapper.style.width = `${originalWidth}px`; wrapper.style.maxWidth = 'none'; wrapper.style.boxSizing = 'border-box'
          wrapper.style.zoom = String(fit.current ? Math.min(1, Math.max(0.1, ((frame.current?.clientWidth ?? originalWidth) - 8) / originalWidth)) : 1)
        }
        resize.current()
        resizeObserver.current?.disconnect()
        resizeObserver.current = new ResizeObserver(() => resize.current())
        resizeObserver.current.observe(frame.current)
      }
      const marks = highlightSourceText(body, quote || passage, false, passage)
      const page = marks[0]?.closest('section.lumin-docx')
      const pageIndex = page ? pages.indexOf(page) + 1 : 0
      if (frame.current?.contentDocument !== document) return
      if (marks.length) { marks[0].scrollIntoView({ block: 'center', inline: 'start' }); setStatus(`${quote ? 'Supporting quote' : 'Cited passage'} highlighted${pageIndex ? ` on preview page ${pageIndex} of ${pages.length}` : ' in the document'}.`) }
      else setStatus(`Original Word preview${pages.length ? ` · ${pages.length} preview pages` : ''}. An exact text match was not found; read the supporting passage beside it.`)
    } catch (cause) { if (frame.current?.contentDocument === document) setError(cause instanceof Error ? cause.message : 'Could not render this Word document.') }
  }

  return <div className="h-full flex flex-col min-h-80">
    <p role="status" className="text-xs mb-2 text-[var(--lumin-muted)]">{status}</p>
    <p className="text-[11px] mb-3 text-[var(--lumin-muted)]">Word preview pages may differ from Microsoft Word. Explicit page breaks are preserved.</p>
    <button type="button" onClick={() => { fit.current = !fit.current; setFitWidth(fit.current); resize.current() }} aria-pressed={fitWidth} className="self-start mb-2 rounded-lg border border-[var(--lumin-border)] px-2 py-1 text-xs">{fitWidth ? 'Show actual size' : 'Fit page to width'}</button>
    {error ? <p role="alert" className="text-sm text-red-500">{error}</p> : <iframe ref={frame} title="Original Word document preview" sandbox="allow-same-origin" referrerPolicy="no-referrer" srcDoc={WORD_PREVIEW_DOCUMENT} onLoad={() => { render() }} className="w-full flex-1 min-h-[55vh] rounded-xl border border-[var(--lumin-border)] bg-white" />}
  </div>
}
