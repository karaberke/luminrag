/** A CSP source must be an exact configured origin, never a hosting-provider wildcard. */
export function backendConnectOrigin(apiBase: string): string | null {
  const configured = apiBase.trim()
  if (!configured) return null
  if (/[\\\s]/u.test(configured)) throw new Error('VITE_API_URL must be a valid HTTPS origin or same-origin path.')
  if (configured.startsWith('/') && !configured.startsWith('//')) return null
  const target = new URL(configured)
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(target.hostname)
  if (target.protocol !== 'https:' && !(target.protocol === 'http:' && loopback)) throw new Error('VITE_API_URL must use HTTPS for a remote server.')
  if (target.username || target.password || !/^(?:[a-z0-9.-]+|\[[a-f0-9:]+\])$/i.test(target.hostname)) throw new Error('VITE_API_URL must name one exact server without credentials or wildcards.')
  return target.origin
}

export function deploymentHeaders(template: string, apiBase: string): string {
  const origin = backendConnectOrigin(apiBase)
  const connect = [...new Set(["'self'", 'https://openrouter.ai', ...(origin ? [origin] : [])])].join(' ')
  if (!/connect-src[^;]*;/u.test(template)) throw new Error('The security header template needs a connect-src directive.')
  return template.replace(/connect-src[^;]*;/u, `connect-src ${connect};`)
}
