import assert from 'node:assert/strict'
import { after, afterEach, before, test } from 'node:test'
import { createTestServer } from './helpers/vite.mjs'
import { createServer as createHttpServer } from 'node:http'
import { once } from 'node:events'

let server
let api
let auth
const originalFetch = globalThis.fetch
const subscriptions = new Set()
const settle = () => new Promise((resolve) => setImmediate(resolve))

before(async () => {
  server = await createTestServer()
  api = await server.ssrLoadModule('/src/utils/api.ts')
  auth = await server.ssrLoadModule('/src/utils/auth.ts')
})

afterEach(() => {
  for (const unsubscribe of subscriptions) unsubscribe()
  subscriptions.clear()
  globalThis.fetch = originalFetch
})

after(async () => { await server?.close() })

function observe(onStatus, onError = (error) => { throw error }) {
  const unsubscribe = auth.observeSessionStatus(onStatus, onError)
  subscriptions.add(unsubscribe)
  return unsubscribe
}

test('concurrent rejected data requests trigger one expiry cleanup and no auth-status loop', async () => {
  let completeStatus
  let statusRequests = 0
  let invalidations = 0
  globalThis.fetch = async (url) => {
    if (url.endsWith('/api/auth/status')) {
      statusRequests++
      return new Promise((resolve) => { completeStatus = resolve })
    }
    return new Response(null, { status: 401 })
  }
  observe((status) => { if (!status.authenticated) invalidations++ })

  await Promise.all(['/api/graph', '/api/query', '/api/suggestions'].map((path) => api.apiFetch(path)))
  assert.equal(statusRequests, 1)
  completeStatus(Response.json({ authenticated: false, token_required: true }))
  await settle()
  assert.equal(invalidations, 1)
  assert.equal(statusRequests, 1)
})

test('a forbidden operation with a valid session does not invalidate loaded state', async () => {
  let invalidations = 0
  globalThis.fetch = async (url) => url.endsWith('/api/auth/status')
    ? Response.json({ authenticated: true, token_required: true })
    : new Response(null, { status: 403 })
  observe((status) => { if (!status.authenticated) invalidations++ })

  await api.apiFetch('/api/graph/node')
  await settle()
  assert.equal(invalidations, 0)
})

test('disposing a session observer cancels its request and ignores a late status reply', async () => {
  let completeStatus
  let requestSignal
  const statuses = []
  globalThis.fetch = async (url, options) => {
    if (url.endsWith('/api/auth/status')) {
      requestSignal = options.signal
      // Deliberately ignore cancellation, as a reply may already have arrived.
      return new Promise((resolve) => { completeStatus = resolve })
    }
    return new Response(null, { status: 401 })
  }
  const unsubscribe = observe((status) => statuses.push(status))
  await api.apiFetch('/api/graph')
  unsubscribe()
  assert.equal(requestSignal.aborted, true)
  completeStatus(Response.json({ authenticated: false, token_required: true }))
  await settle()
  assert.deepEqual(statuses, [])
})

test('failed session rechecks report once and allow a later request to retry', async () => {
  let statusRequests = 0
  let invalidations = 0
  const errors = []
  globalThis.fetch = async (url) => {
    if (url.endsWith('/api/auth/status')) {
      statusRequests++
      return statusRequests === 1
        ? new Response(null, { status: 401 })
        : Response.json({ authenticated: false, token_required: true })
    }
    return new Response(null, { status: 401 })
  }
  observe((status) => { if (!status.authenticated) invalidations++ }, (error) => errors.push(error))
  await api.apiFetch('/api/graph')
  await settle()
  assert.equal(statusRequests, 1)
  assert.equal(errors.length, 1)
  assert.equal(errors[0].status, 401)
  assert.equal(invalidations, 0)

  await api.apiFetch('/api/suggestions')
  await settle()
  assert.equal(statusRequests, 2)
  assert.equal(invalidations, 1)
})

test('hosted connection needs no app login and a provider 401 keeps the access gate closed', async () => {
  const workspaceId = 'fbe84c15-c76e-4284-9e40-a197b384a1ca'
  let statusRequests = 0
  const backend = createHttpServer((request, response) => {
    response.setHeader('Content-Type', 'application/json')
    if (request.url === '/api/health') {
      response.end(JSON.stringify({ status: 'ok', features: { hosted: true, accepted_extensions: ['.pdf', '.md'] } }))
    } else if (request.url === '/api/auth/status') {
      statusRequests++
      // Public status has no workspace dependency or application login cookie.
      assert.equal(request.headers['x-workspace-id'], undefined)
      assert.equal(request.headers.cookie, undefined)
      response.end(JSON.stringify({ authenticated: true, token_required: false }))
    } else if (request.url === '/api/query') {
      assert.equal(request.headers['x-workspace-id'], workspaceId)
      response.statusCode = 401
      response.end(JSON.stringify({ detail: 'Connect an OpenRouter or Anthropic account in Settings.' }))
    } else {
      response.statusCode = 404
      response.end('{}')
    }
  })
  backend.listen(0, '127.0.0.1')
  await once(backend, 'listening')
  const origin = `http://127.0.0.1:${backend.address().port}`
  globalThis.fetch = (url, options) => originalFetch(new URL(url, origin), options)
  try {
    const health = await api.apiJson('/api/health', { skipAuthCheck: true })
    const initialStatus = await auth.fetchAuthStatus()
    assert.equal(health.features.hosted, true)
    assert.deepEqual(initialStatus, { authenticated: true, token_required: false })

    let accessGateVisible = !initialStatus.authenticated
    const rechecked = new Promise((resolve, reject) => {
      observe((status) => {
        accessGateVisible = !status.authenticated
        resolve(status)
      }, reject)
    })
    await assert.rejects(api.apiJson('/api/query', { method: 'POST', workspaceId, json: { query: 'A question' } }), (error) => {
      assert.equal(error.status, 401)
      assert.match(error.detail, /Settings/)
      return true
    })
    assert.deepEqual(await rechecked, { authenticated: true, token_required: false })
    assert.equal(accessGateVisible, false)
    assert.equal(statusRequests, 2)
  } finally {
    for (const unsubscribe of subscriptions) unsubscribe()
    subscriptions.clear()
    backend.closeAllConnections()
    await new Promise((resolve) => backend.close(resolve))
  }
})
