import { ApiError, apiFetch, apiJson, isAbortError, onAuthRequired, readErrorDetail } from './api'

export interface AuthStatus {
  authenticated: boolean
  token_required: boolean
}

/** Public endpoint — reports whether the current cookie session is valid. */
export function fetchAuthStatus(signal?: AbortSignal): Promise<AuthStatus> {
  return apiJson<AuthStatus>('/api/auth/status', { signal, cache: 'no-store', skipAuthCheck: true })
}

/** Recheck rejected requests once at a time without accepting replies after disposal. */
export function observeSessionStatus(
  onStatus: (status: AuthStatus) => void,
  onError: (error: unknown) => void,
): () => void {
  const controller = new AbortController()
  let checking = false
  const unsubscribe = onAuthRequired(() => {
    if (checking || controller.signal.aborted) return
    checking = true
    fetchAuthStatus(controller.signal)
      .then((status) => {
        if (!controller.signal.aborted) onStatus(status)
      })
      .catch((error) => {
        if (!controller.signal.aborted && !isAbortError(error)) onError(error)
      })
      .finally(() => { checking = false })
  })
  return () => {
    unsubscribe()
    controller.abort()
  }
}

/**
 * Exchange an access key for an HttpOnly session cookie. The key is only ever
 * sent in the request body — never stored, logged, or put in a URL.
 */
export async function createSession(token: string, signal?: AbortSignal): Promise<void> {
  const res = await apiFetch('/api/auth/session', {
    method: 'POST',
    json: { token },
    signal,
    skipAuthCheck: true,
  })
  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res))
}

export async function endSession(): Promise<void> {
  const res = await apiFetch('/api/auth/session', { method: 'DELETE', skipAuthCheck: true })
  if (!res.ok) throw new ApiError(res.status, await readErrorDetail(res))
}
