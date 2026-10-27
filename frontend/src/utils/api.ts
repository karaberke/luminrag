import { activeWorkspaceId, cachedAssetUrl, initializeStorage } from '../storage/db'
import { llmHeaders, loadLlmSettings } from './settings'

/**
 * Centralised backend access.
 *
 * Every request to the LuminRAG backend goes through `apiFetch` so that the
 * base URL, workspace capability, provider credentials, cache hydration and
 * 401/403 handling live in one place. Provider OAuth uses its own public API.
 */

/** Backend origin (empty string = same origin, served via the Vite/nginx proxy). */
export const API_BASE: string = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

/** Join a backend path onto API_BASE exactly once. */
export function apiUrl(path: string): string {
  return `${API_BASE}/${path.replace(/^\/+/, '')}`
}

/**
 * Resolve an asset path returned by the backend (e.g. an illustration).
 * Browser assets resolve to known Blob URLs. Remote illustrations are limited
 * to the configured backend so imported content cannot create tracking images.
 */
export function resolveAssetUrl(path: string | null | undefined): string | null {
  const trimmed = path?.trim()
  if (!trimmed) return null
  const cached = cachedAssetUrl(trimmed)
  if (cached) return cached
  if (/^https?:\/\//i.test(trimmed)) {
    const origin = typeof window === 'undefined' ? 'http://localhost' : window.location.origin
    const backend = new URL(`${API_BASE}/`, origin)
    const candidate = new URL(trimmed)
    if (candidate.origin !== backend.origin || !candidate.pathname.startsWith('/static/')) return null
    return trimmed
  }
  return apiUrl(trimmed)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail ? `${detail} (HTTP ${status})` : `Request failed (HTTP ${status})`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError'
}

export function isAuthError(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 401 || e.status === 403)
}

/** Network failures, timeouts, rate limits and 5xx are worth retrying; other HTTP errors are not. */
export function isRetryableError(e: unknown): boolean {
  if (!(e instanceof ApiError)) return !isAbortError(e)
  return e.status === 408 || e.status === 429 || e.status >= 500
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/** Extract a readable message from an error response (FastAPI `detail`, plain text, or status text). */
export async function readErrorDetail(res: Response): Promise<string> {
  let text = ''
  try {
    text = await res.text()
  } catch {
    return res.statusText
  }
  try {
    const body: unknown = JSON.parse(text)
    if (body && typeof body === 'object' && 'detail' in body) {
      const { detail } = body as { detail: unknown }
      if (typeof detail === 'string') return detail
      if (Array.isArray(detail)) {
        return detail
          .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d)))
          .join('; ')
      }
    }
  } catch {
    // not JSON — fall through to raw text
  }
  return text.trim() || res.statusText
}

// ---------------------------------------------------------------------------
// Auth-required notifications
// ---------------------------------------------------------------------------

type AuthListener = () => void
const authListeners = new Set<AuthListener>()
const LLM_ROUTES = new Set(['/api/query', '/api/ingest'])
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])
export function isLlmRoute(path: string): boolean {
  const route = path.split('?')[0]
  return LLM_ROUTES.has(route)
}

/** Subscribe to 401/403 responses from protected endpoints. Returns an unsubscribe function. */
export function onAuthRequired(listener: AuthListener): () => void {
  authListeners.add(listener)
  return () => { authListeners.delete(listener) }
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

export interface ApiFetchOptions extends Omit<RequestInit, 'credentials'> {
  /** JSON request body; sets Content-Type automatically. */
  json?: unknown
  /** Don't broadcast 401/403 to auth listeners (used by the auth endpoints themselves). */
  skipAuthCheck?: boolean
  /** Internal: avoid recursive hydrate during status, import and export. */
  skipHydrate?: boolean
  workspaceId?: string
}

type Hydrator = (workspaceId: string) => Promise<void>
let hydrateWorkspace: Hydrator | undefined
export function registerHydrator(hydrator: Hydrator): void { hydrateWorkspace = hydrator }

/** Every secret uses the same transport policy, including the shared server login token. */
export function assertSecureCredentialTransport(path: string): void {
  if (typeof window === 'undefined') return
  if (!window.isSecureContext && !LOOPBACK_HOSTS.has(window.location.hostname)) throw new Error('Open LuminRAG over HTTPS before sending credentials.')
  const target = new URL(apiUrl(path), window.location.origin)
  if (target.protocol !== 'https:' && !(target.protocol === 'http:' && LOOPBACK_HOSTS.has(target.hostname))) throw new Error('The server must use HTTPS before it can receive credentials.')
  if (target.username || target.password) throw new Error('The server address must not contain credentials.')
}

/** fetch() against the backend with cookie credentials. Resolves for any HTTP status. */
export async function apiFetch(path: string, options: ApiFetchOptions = {}): Promise<Response> {
  const requestedWorkspace = options.workspaceId ?? activeWorkspaceId()
  if (typeof indexedDB !== 'undefined') await Promise.all([initializeStorage(), loadLlmSettings()])
  const { json, skipAuthCheck = false, skipHydrate = false, workspaceId = requestedWorkspace || activeWorkspaceId(), headers, ...init } = options
  const finalHeaders = new Headers(headers)
  if (workspaceId) finalHeaders.set('X-Workspace-Id', workspaceId)
  for (const name of ['X-LLM-Provider', 'X-LLM-Key', 'X-LLM-Model-Fast', 'X-LLM-Model-Smart']) finalHeaders.delete(name)
  if (isLlmRoute(path)) for (const [key, value] of Object.entries(llmHeaders())) finalHeaders.set(key, value)
  const sessionLogin = path.split('?')[0] === '/api/auth/session' && init.method?.toUpperCase() === 'POST'
  if (finalHeaders.has('X-LLM-Key') || sessionLogin) {
    assertSecureCredentialTransport(path)
    init.redirect = 'error'
  }
  if (json !== undefined) {
    finalHeaders.set('Content-Type', 'application/json')
    init.body = JSON.stringify(json)
  }
  let res = await fetch(apiUrl(path), { ...init, headers: finalHeaders, credentials: 'include' })
  if (res.status === 409 && !skipHydrate && hydrateWorkspace) {
    const data = await res.clone().json().catch(() => null)
    if (data?.code === 'workspace_not_loaded' || data?.detail?.code === 'workspace_not_loaded') {
      await hydrateWorkspace(workspaceId)
      if (init.signal?.aborted) throw new DOMException('Cancelled', 'AbortError')
      res = await fetch(apiUrl(path), { ...init, headers: finalHeaders, credentials: 'include' })
    }
  }
  if (res.ok && ['POST', 'PATCH', 'DELETE', 'PUT'].includes(init.method ?? '') && !skipHydrate && typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('lumin-workspace-mutated', { detail: workspaceId }))
  }
  if (!skipAuthCheck && (res.status === 401 || res.status === 403)) {
    authListeners.forEach((listener) => listener())
  }
  return res
}

/** apiFetch + throw ApiError on non-2xx. */
export async function apiRequest(path: string, options: ApiFetchOptions = {}): Promise<Response> {
  const res = await apiFetch(path, options)
  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res))
  return res
}

/** apiFetch + throw ApiError on non-2xx + parse JSON body. */
export async function apiJson<T>(path: string, options: ApiFetchOptions = {}): Promise<T> {
  const res = await apiRequest(path, options)
  return res.json() as Promise<T>
}
