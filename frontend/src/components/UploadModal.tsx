import { useEffect, useRef, useState } from 'react'
import { useTheme } from '../theme'
import { ApiError, apiJson, errorMessage, isAbortError } from '../utils/api'
import { activeWorkspaceId, bindOriginalsToJob, discardStagedOriginals, stageOriginals, withOriginalsLock } from '../storage/db'

interface UploadModalProps {
  onClose: () => void
  onJobStarted: (jobId: string, workspaceId: string) => void
}

type UploadStage = 'idle' | 'uploading' | 'starting' | 'done' | 'error'

interface FileEntry {
  id: number
  file: File
  slides: boolean
}

const DEFAULT_ACCEPTED = '.pdf,.jpg,.jpeg,.png,.webp,.docx,.pptx,.xlsx,.html,.htm,.epub,.md,.markdown,.txt'

let nextEntryId = 0

const isPdf = (file: File) => file.name.toLowerCase().endsWith('.pdf')

function describeUploadError(e: unknown): string {
  if (!(e instanceof ApiError)) return errorMessage(e)
  const detail = e.detail ? `: ${e.detail}` : ''
  switch (e.status) {
    case 413: return `Upload rejected — file too large${detail}`
    case 415: return `Upload rejected — unsupported file type${detail}`
    case 400:
    case 422: return `Upload rejected${detail}`
    default:  return `Upload failed (HTTP ${e.status})${detail}`
  }
}

/** Multipart POST straight to the backend (default). */
async function ingestDirect(entries: FileEntry[], signal: AbortSignal, workspaceId: string): Promise<string> {
  const form = new FormData()
  for (const { file } of entries) form.append('files', file, file.name)
  form.append('slides', JSON.stringify(entries.filter((e) => e.slides).map((e) => e.file.name)))
  const { job_id } = await apiJson<{ job_id: string }>('/api/ingest', { method: 'POST', body: form, signal, workspaceId })
  return job_id
}

export default function UploadModal({ onClose, onJobStarted }: UploadModalProps) {
  const [entries, setEntries] = useState<FileEntry[]>([])
  const [stage, setStage] = useState<UploadStage>('idle')
  const [accepted, setAccepted] = useState(DEFAULT_ACCEPTED)
  const [maxBytes, setMaxBytes] = useState(50 * 1024 * 1024)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const uploadAbortRef = useRef<AbortController | null>(null)
  const lifetimeRef = useRef<AbortController | null>(null)
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const busy = stage === 'uploading' || stage === 'starting'
  const cancellable = stage === 'uploading'
  useEffect(() => {
    const controller = new AbortController()
    apiJson<{ features?: { accepted_extensions?: string[]; max_file_bytes?: number } }>('/api/health', { signal: controller.signal }).then((health) => {
      if (health.features?.accepted_extensions?.length) setAccepted(health.features.accepted_extensions.join(','))
      if (health.features?.max_file_bytes) setMaxBytes(health.features.max_file_bytes)
    }).catch(() => {})
    return () => controller.abort()
  }, [])

  // Abort any in-flight upload when the modal unmounts
  useEffect(() => {
    const controller = new AbortController()
    lifetimeRef.current = controller
    return () => controller.abort()
  }, [])

  // Auto-close shortly after the job has been handed off
  useEffect(() => {
    if (stage !== 'done') return
    const t = setTimeout(onClose, 1500)
    return () => clearTimeout(t)
  }, [stage, onClose])

  const addFiles = (list: FileList | null) => {
    if (!list || busy) return
    const incoming = Array.from(list)
    setEntries((prev) => {
      const existing = new Set(prev.map((e) => e.file.name))
      const added = incoming
        .filter((f) => !existing.has(f.name))
        .map((file) => ({ id: nextEntryId++, file, slides: false }))
      return [...prev, ...added]
    })
  }

  const removeEntry = (id: number) => {
    setEntries((prev) => prev.filter((e) => e.id !== id))
  }

  const toggleSlide = (id: number) => {
    setEntries((prev) => prev.map((e) => (e.id === id ? { ...e, slides: !e.slides } : e)))
  }

  const handleUpload = async () => {
    if (!entries.length || busy) return
    const controller = new AbortController()
    uploadAbortRef.current = controller
    lifetimeRef.current?.signal.addEventListener('abort', () => controller.abort(), { once: true })
    // Snapshot so ordering/flags can't change mid-upload (controls are disabled too)
    const snapshot = entries.map((e) => ({ ...e, slides: e.slides && isPdf(e.file) }))
    setError(null)
    let stagedOriginals: string | null = null

    try {
      const workspaceId = activeWorkspaceId()
      const unsupported = snapshot.find(({ file }) => !accepted.split(',').some((extension) => file.name.toLowerCase().endsWith(extension)))
      if (unsupported) throw new Error(`Unsupported file: ${unsupported.file.name}`)
      if (snapshot.some(({ file }) => file.size > maxBytes)) throw new Error(`Files must be no larger than ${Math.round(maxBytes / 1024 / 1024)} MB.`)
      setStage('uploading')
      await withOriginalsLock(workspaceId, async () => {
        if (controller.signal.aborted) return
        stagedOriginals = await stageOriginals(snapshot.map(({ file }) => file), workspaceId)
        if (controller.signal.aborted) { await discardStagedOriginals(stagedOriginals); stagedOriginals = null; return }
        setStage('starting')
        const jobId = await ingestDirect(snapshot, controller.signal, workspaceId)
        await bindOriginalsToJob(stagedOriginals, jobId)
        stagedOriginals = null
        // Hand off to App immediately — App owns all polling from here.
        onJobStarted(jobId, workspaceId)
        setStage('done')
      }, true)
    } catch (e) {
      if (stagedOriginals) await discardStagedOriginals(stagedOriginals).catch(() => {})
      if (controller.signal.aborted || isAbortError(e)) return
      setError(describeUploadError(e))
      setStage('error')
    } finally {
      if (uploadAbortRef.current === controller) uploadAbortRef.current = null
    }
  }

  const handleCancel = () => {
    if (busy) {
      if (!cancellable) return
      uploadAbortRef.current?.abort()
      setStage('idle')
      setError(null)
      return
    }
    onClose()
  }

  const handleClose = () => {
    if (busy && !cancellable) return
    uploadAbortRef.current?.abort()
    onClose()
  }

  const stageLabel = (): string => {
    switch (stage) {
      case 'uploading': return `Uploading ${entries.length} file${entries.length !== 1 ? 's' : ''}…`
      case 'starting':   return 'Starting ingestion…'
      default:           return 'Upload & Ingest'
    }
  }

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm ${
      isDark ? 'bg-black/60' : 'bg-black/30'
    }`}>
      <div className={`lumin-modal-card border rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden ${
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
            onClick={handleClose}
            disabled={busy && !cancellable}
            className={`transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'}`}
            aria-label="Close"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
              <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-4">
          {stage === 'done' ? (
            <div className={`flex items-center gap-2 ${isDark ? 'text-green-400' : 'text-green-600'}`}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5 shrink-0">
                <path d="M22 11.08V12a10 10 0 11-5.93-9.14" strokeLinecap="round" />
                <path d="M22 4L12 14.01l-3-3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span className="text-sm font-medium">Upload complete — ingestion running in background</span>
            </div>
          ) : (
            <>
              {/* Drop zone */}
              <div
                aria-disabled={busy}
                className={`lumin-dropzone border-2 border-dashed rounded-xl p-6 text-center transition-colors ${
                  busy
                    ? `opacity-50 cursor-not-allowed ${isDark ? 'border-slate-700' : 'border-slate-300'}`
                    : dragging
                      ? 'cursor-pointer border-indigo-500 bg-indigo-500/5'
                      : isDark
                        ? 'cursor-pointer border-slate-700 hover:border-indigo-600'
                        : 'cursor-pointer border-slate-300 hover:border-indigo-400 hover:bg-indigo-50/50'
                }`}
                onClick={() => { if (!busy) inputRef.current?.click() }}
                onDragOver={(e) => { e.preventDefault(); if (!busy) setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  multiple
                  accept={accepted}
                  className="hidden"
                  disabled={busy}
                  onChange={(e) => { addFiles(e.target.files); e.target.value = '' }}
                />
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                  className={`w-8 h-8 mx-auto mb-2 ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <p className={`text-sm font-medium mb-1 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                  Drop files here or click to browse
                </p>
                <p className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  {accepted.replace(/\./g, '').split(',').join(' · ').toUpperCase()} · {Math.round(maxBytes / 1024 / 1024)} MB per file
                </p>
              </div>

              {/* File list */}
              {entries.length > 0 && (
                <ul className="space-y-1.5 max-h-52 overflow-y-auto">
                  {entries.map(({ id, file, slides }) => (
                    <li key={id} className={`flex items-center gap-3 rounded-lg px-3 py-2 ${
                      isDark ? 'bg-slate-800' : 'bg-slate-50 border border-slate-200'
                    }`}>
                      <span className={`text-xs flex-1 truncate ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                        {file.name}
                      </span>
                      {isPdf(file) && (
                        <label className={`flex items-center gap-1.5 text-xs shrink-0 ${
                          busy ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
                        } ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                          <input
                            type="checkbox"
                            className="accent-indigo-500"
                            checked={slides}
                            disabled={busy}
                            onChange={() => toggleSlide(id)}
                          />
                          Slides
                        </label>
                      )}
                      <button
                        onClick={() => removeEntry(id)}
                        disabled={busy}
                        className={`transition-colors shrink-0 disabled:opacity-40 disabled:cursor-not-allowed ${
                          isDark ? 'text-slate-600 hover:text-slate-400' : 'text-slate-400 hover:text-slate-600'
                        }`}
                        aria-label={`Remove ${file.name}`}
                      >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                          <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
                        </svg>
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              {stage === 'error' && error && (
                <p role="alert" className={`text-xs rounded-lg px-3 py-2 border break-words ${
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
          {stage === 'done' ? (
            <button
              onClick={onClose}
              className="text-sm bg-indigo-600 hover:bg-indigo-600 text-white px-4 py-1.5 rounded-lg transition-colors"
            >
              Done
            </button>
          ) : (
            <>
              <button
                onClick={handleCancel}
                disabled={busy && !cancellable}
                className={`text-xs px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={!entries.length || busy}
                className="text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-1.5 rounded-lg transition-colors flex items-center gap-2"
              >
                {busy ? (
                  <>
                    <div className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin" />
                    {stageLabel()}
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
