import { useRef, useState } from 'react'
import { useTheme } from '../ThemeContext'

interface IngestResult {
  files_processed: number
  chunks_produced: number
  triples_extracted: number
  graph_nodes: number
  graph_edges: number
  failed: string[]
}

interface UploadModalProps {
  onClose: () => void
  onSuccess: () => void
}

const ACCEPTED = '.pdf,.mp4,.mkv,.mov,.avi,.mp3,.wav,.m4a,.ogg,.flac,.aac,.wma,.jpg,.jpeg,.png,.webp'
const API_BASE = import.meta.env.VITE_API_URL ?? ''

export default function UploadModal({ onClose, onSuccess }: UploadModalProps) {
  const [files, setFiles] = useState<File[]>([])
  const [slideFiles, setSlideFiles] = useState<Set<string>>(new Set())
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const addFiles = (list: FileList | null) => {
    if (!list) return
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name))
      return [...prev, ...Array.from(list).filter((f) => !existing.has(f.name))]
    })
  }

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name))
    setSlideFiles((prev) => { const n = new Set(prev); n.delete(name); return n })
  }

  const toggleSlide = (name: string) => {
    setSlideFiles((prev) => {
      const n = new Set(prev)
      n.has(name) ? n.delete(name) : n.add(name)
      return n
    })
  }

  const handleUpload = async () => {
    if (!files.length || uploading) return
    setUploading(true)
    setError(null)

    const form = new FormData()
    for (const f of files) form.append('files', f)
    form.append('slides', Array.from(slideFiles).join(','))

    try {
      const res = await fetch(`${API_BASE}/api/ingest`, { method: 'POST', body: form })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Server error ${res.status}: ${text}`)
      }
      const data: IngestResult = await res.json()
      setResult(data)
      onSuccess()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  const stats: [string, number][] = result
    ? [
        ['Files processed', result.files_processed],
        ['Chunks produced', result.chunks_produced],
        ['Triples extracted', result.triples_extracted],
        ['Graph nodes', result.graph_nodes],
        ['Graph edges', result.graph_edges],
      ]
    : []

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm ${
      isDark ? 'bg-black/60' : 'bg-black/30'
    }`}>
      <div className={`border rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden ${
        isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'
      }`}>
        {/* Header */}
        <div className={`flex items-center justify-between px-5 py-4 border-b ${
          isDark ? 'border-slate-800' : 'border-slate-100'
        }`}>
          <h2 className={`text-sm font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
            Upload Course Content
          </h2>
          <button
            onClick={onClose}
            className={`transition-colors ${isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'}`}
            aria-label="Close"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
              <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-4">
          {result ? (
            /* ── Result view ── */
            <div className="space-y-3">
              <div className={`flex items-center gap-2 ${isDark ? 'text-green-400' : 'text-green-600'}`}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5 shrink-0">
                  <path d="M22 11.08V12a10 10 0 11-5.93-9.14" strokeLinecap="round" />
                  <path d="M22 4L12 14.01l-3-3" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className="text-sm font-medium">Ingestion complete — graph updated</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {stats.map(([label, value]) => (
                  <div key={label} className={`rounded-lg px-3 py-2 ${isDark ? 'bg-slate-800' : 'bg-slate-50'}`}>
                    <p className={isDark ? 'text-slate-500' : 'text-slate-400'}>{label}</p>
                    <p className={`font-semibold text-base ${isDark ? 'text-white' : 'text-slate-900'}`}>{value}</p>
                  </div>
                ))}
              </div>
              {result.failed.length > 0 && (
                <p className={`text-xs rounded-lg px-3 py-2 border ${
                  isDark
                    ? 'text-amber-400 bg-amber-950/30 border-amber-800'
                    : 'text-amber-700 bg-amber-50 border-amber-300'
                }`}>
                  Failed: {result.failed.join(', ')}
                </p>
              )}
            </div>
          ) : (
            /* ── Upload form ── */
            <>
              {/* Drop zone */}
              <div
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
                  dragging
                    ? 'border-indigo-500 bg-indigo-500/5'
                    : isDark
                      ? 'border-slate-700 hover:border-indigo-600'
                      : 'border-slate-300 hover:border-indigo-400 hover:bg-indigo-50/50'
                }`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  accept={ACCEPTED}
                  className="hidden"
                  onChange={(e) => addFiles(e.target.files)}
                />
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                  className={`w-8 h-8 mx-auto mb-2 ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <p className={`text-sm font-medium mb-1 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                  Drop files here or click to browse
                </p>
                <p className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  PDF · MP4 · MKV · MOV · MP3 · WAV · JPG · PNG · WEBP
                </p>
              </div>

              {/* File list */}
              {files.length > 0 && (
                <ul className="space-y-1.5 max-h-52 overflow-y-auto">
                  {files.map((f) => (
                    <li key={f.name} className={`flex items-center gap-3 rounded-lg px-3 py-2 ${
                      isDark ? 'bg-slate-800' : 'bg-slate-50 border border-slate-200'
                    }`}>
                      <span className={`text-xs flex-1 truncate ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                        {f.name}
                      </span>
                      {f.name.toLowerCase().endsWith('.pdf') && (
                        <label className={`flex items-center gap-1.5 text-xs cursor-pointer shrink-0 ${
                          isDark ? 'text-slate-400' : 'text-slate-500'
                        }`}>
                          <input
                            type="checkbox"
                            className="accent-indigo-500"
                            checked={slideFiles.has(f.name)}
                            onChange={() => toggleSlide(f.name)}
                          />
                          Slides
                        </label>
                      )}
                      <button
                        onClick={() => removeFile(f.name)}
                        className={`transition-colors shrink-0 ${
                          isDark ? 'text-slate-600 hover:text-slate-400' : 'text-slate-400 hover:text-slate-600'
                        }`}
                        aria-label={`Remove ${f.name}`}
                      >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                          <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
                        </svg>
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              {error && (
                <p className={`text-xs rounded-lg px-3 py-2 border ${
                  isDark
                    ? 'text-red-400 bg-red-950/40 border-red-800'
                    : 'text-red-600 bg-red-50 border-red-300'
                }`}>
                  {error}
                </p>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className={`flex items-center justify-end gap-2 px-5 py-3 border-t ${
          isDark ? 'border-slate-800 bg-slate-950/40' : 'border-slate-100 bg-slate-50/60'
        }`}>
          {result ? (
            <button
              onClick={onClose}
              className="text-sm bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-1.5 rounded-lg transition-colors"
            >
              Done
            </button>
          ) : (
            <>
              <button
                onClick={onClose}
                className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${
                  isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={!files.length || uploading}
                className="text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-1.5 rounded-lg transition-colors flex items-center gap-2"
              >
                {uploading ? (
                  <>
                    <div className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin" />
                    Processing…
                  </>
                ) : (
                  'Upload & Ingest'
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
