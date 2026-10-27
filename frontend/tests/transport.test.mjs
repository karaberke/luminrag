import assert from 'node:assert/strict'
import { after, afterEach, test } from 'node:test'
import { createTestServer } from './helpers/vite.mjs'
import { createServer as createHttpServer } from 'node:http'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

const originalFetch = globalThis.fetch
const servers = []
afterEach(() => { globalThis.fetch = originalFetch; delete globalThis.window })
after(async () => { await Promise.all(servers.map((server) => server.close())) })

async function fixture(target, page = 'https://lumin.example/', secure = true) {
  globalThis.window = Object.assign(new EventTarget(), { location: new URL(page), isSecureContext: secure })
  const server = await createTestServer({ define: { 'import.meta.env.VITE_API_URL': JSON.stringify(target) } })
  servers.push(server)
  const api = await server.ssrLoadModule('/src/utils/api.ts')
  const auth = await server.ssrLoadModule('/src/utils/auth.ts')
  const settings = await server.ssrLoadModule('/src/utils/settings.ts')
  settings.getLlmSettings().key = 'test-only-provider-key'
  return { api, auth, server, settings }
}

test('session body tokens and provider keys both reject insecure remote pages before any request', async () => {
  const { api, auth, settings, server } = await fixture('https://api.example.test', 'http://insecure.example/', false)
  let calls = 0
  globalThis.fetch = async () => { calls++; return Response.json({}) }
  await assert.rejects(api.apiFetch('/api/query', { method: 'POST', json: { query: 'Hello' } }), /HTTPS/)
  settings.getLlmSettings().key = ''
  await assert.rejects(auth.createSession('test-only-login-token'), /HTTPS/)
  assert.equal(calls, 0)
  const { default: AccessGate } = await server.ssrLoadModule('/src/components/AccessGate.tsx')
  const markup = renderToStaticMarkup(createElement(AccessGate, { isDark: false, statusError: null, onRetryStatus: () => {}, onAuthenticated: () => {} }))
  assert.match(markup, /HTTPS before sending credentials/)
  assert.match(markup, /<input[^>]*disabled=""/)
})

test('both secret types reject insecure remote API targets and URL user credentials', async () => {
  for (const target of ['http://api.example.test', 'https://user:password@api.example.test']) {
    const { api, auth, settings } = await fixture(target)
    let calls = 0
    globalThis.fetch = async () => { calls++; return Response.json({}) }
    await assert.rejects(api.apiFetch('/api/ingest', { method: 'POST' }), /HTTPS|must not contain credentials/)
    settings.getLlmSettings().key = ''
    await assert.rejects(auth.createSession('test-only-login-token'), /HTTPS|must not contain credentials/)
    assert.equal(calls, 0)
  }
})

test('loopback development permits both secret types while overriding caller redirect policies', async () => {
  const { api, auth, settings } = await fixture('http://127.0.0.1:8000', 'http://localhost:5173/', false)
  const requests = []
  globalThis.fetch = async (url, options) => { requests.push({ url, options }); return Response.json({}) }
  await api.apiFetch('/api/query', { method: 'POST', redirect: 'follow', json: { query: 'Question' } })
  settings.getLlmSettings().key = ''
  await auth.createSession('test-only-login-token')
  assert.equal(requests.length, 2)
  for (const request of requests) assert.equal(request.options.redirect, 'error')
  assert.equal(requests[0].options.headers.get('X-LLM-Key'), 'test-only-provider-key')
  assert.equal(requests[1].options.headers.get('X-LLM-Key'), null)
  assert.equal(JSON.parse(requests[1].options.body).token, 'test-only-login-token')
  assert.ok(requests.every(({ url }) => !url.includes('test-only')))
})

test('native fetch refuses cross-origin 307 redirects before either credential reaches the redirected server', async () => {
  let destinationRequests = 0, initialRequests = 0
  const destination = createHttpServer((_request, response) => { destinationRequests++; response.end('Unexpected secret delivery') })
  await new Promise((resolve) => destination.listen(0, '127.0.0.1', resolve))
  const destinationURL = `http://127.0.0.1:${destination.address().port}/capture`
  const initial = createHttpServer((request, response) => { initialRequests++; request.resume(); response.writeHead(307, { Location: destinationURL }); response.end() })
  await new Promise((resolve) => initial.listen(0, '127.0.0.1', resolve))
  try {
    const { api, auth, settings } = await fixture(`http://127.0.0.1:${initial.address().port}`, 'http://localhost:5173/', false)
    await assert.rejects(api.apiFetch('/api/query', { method: 'POST', json: { query: 'Local test' } }), /fetch failed/i)
    settings.getLlmSettings().key = ''
    await assert.rejects(auth.createSession('test-only-login-token'), /fetch failed/i)
    assert.equal(initialRequests, 2)
    assert.equal(destinationRequests, 0)
  } finally {
    await Promise.all([new Promise((resolve) => initial.close(resolve)), new Promise((resolve) => destination.close(resolve))])
  }
})
