import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ApiError, assertSecureCredentialTransport, errorMessage, isAbortError } from '../utils/api'
import { createSession, fetchAuthStatus } from '../utils/auth'
import BrandMark from './BrandMark'

interface Props {
  isDark: boolean
  /** Set when the auth status check itself failed (network / server error). */
  statusError: string | null
  onRetryStatus: () => void
  onAuthenticated: () => void
}

/**
 * Full-screen access-key prompt shown in token mode before any protected data
 * loads. The key lives only in this component's state and is exchanged for an
 * HttpOnly session cookie; it is cleared as soon as the exchange succeeds.
 */
export default function AccessGate({ isDark, statusError, onRetryStatus, onAuthenticated }: Props) {
  const [accessKey, setAccessKey] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const lifetimeRef = useRef<AbortController | null>(null)
  let transportError: string | null = null
  try { assertSecureCredentialTransport('/api/auth/session') } catch (cause) { transportError = errorMessage(cause) }

  useEffect(() => {
    const controller = new AbortController()
    lifetimeRef.current = controller
    return () => controller.abort()
  }, [])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const key = accessKey.trim()
    if (!key || submitting || transportError) return
    const signal = lifetimeRef.current?.signal
    setSubmitting(true)
    setError(null)
    try {
      await createSession(key, signal)
      setAccessKey('')
      // Confirm the browser actually kept the cookie before unlocking
      const status = await fetchAuthStatus(signal)
      if (!status.authenticated) {
        setError('The server accepted the key, but the session cookie was not stored. Make sure the app and API are served from the same site.')
        return
      }
      onAuthenticated()
    } catch (err) {
      if (signal?.aborted || isAbortError(err)) return
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        setError('Invalid access key.')
      } else if (err instanceof ApiError && err.status === 429) {
        setError('Too many attempts. Wait a moment and try again.')
      } else {
        setError(`Could not connect: ${errorMessage(err)}`)
      }
    } finally {
      if (!signal?.aborted) setSubmitting(false)
    }
  }

  const cardCls = isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'
  const errorCls = isDark
    ? 'text-red-400 bg-red-950/40 border-red-800'
    : 'text-red-600 bg-red-50 border-red-300'

  return (
    <div className={`fixed inset-0 z-[60] flex items-center justify-center backdrop-blur ${
      isDark ? 'bg-slate-950/95' : 'bg-[var(--lumin-background)]/95'
    }`}>
      <div className={`w-full max-w-sm mx-4 border rounded-2xl shadow-2xl p-6 ${cardCls}`}>
        <div className="flex items-center gap-2 mb-4">
          <BrandMark />
          <h1 className={`text-sm font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>Connect to LuminRAG</h1>
        </div>

        {statusError ? (
          <div className="space-y-3">
            <p role="alert" className={`text-xs rounded-lg px-3 py-2 border break-words ${errorCls}`}>
              {statusError}
            </p>
            <button
              onClick={onRetryStatus}
              className="w-full text-sm bg-indigo-600 hover:bg-indigo-600 text-white py-1.5 rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-3">
            <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              This server requires an access key.
            </p>
            {transportError && <p role="alert" className={`text-xs rounded-lg px-3 py-2 border ${errorCls}`}>{transportError}</p>}
            <input
              type="password"
              name="access-key"
              autoComplete="off"
              autoFocus
              value={accessKey}
              onChange={(e) => setAccessKey(e.target.value)}
              disabled={submitting || !!transportError}
              placeholder="Access key"
              aria-label="Access key"
              className={`w-full text-sm border rounded-lg px-3 py-2 outline-none focus:border-indigo-500 ${
                isDark ? 'bg-slate-800 border-slate-600 text-slate-200' : 'bg-slate-50 border-slate-300 text-slate-800'
              }`}
            />
            {error && (
              <p role="alert" className={`text-xs rounded-lg px-3 py-2 border break-words ${errorCls}`}>
                {error}
              </p>
            )}
            <button
              type="submit"
              disabled={!accessKey.trim() || submitting || !!transportError}
              className="w-full text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white py-1.5 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {submitting && <div className="w-3.5 h-3.5 border border-white/30 border-t-white rounded-full animate-spin" />}
              {submitting ? 'Connecting…' : 'Connect'}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}
