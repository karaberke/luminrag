import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { readFile } from 'node:fs/promises'
import { createTestServer } from './helpers/vite.mjs'

let server, headers, template
before(async () => {
  server = await createTestServer()
  headers = await server.ssrLoadModule('/src/utils/deploymentHeaders.ts')
  template = await readFile('public/_headers', 'utf8')
})
after(async () => { await server?.close() })

test('deployed CSP retains the exact configured HTTPS backend origin without hosting wildcards', () => {
  const generated = headers.deploymentHeaders(template, 'https://my-project-api-123.europe-west1.run.app/backend/')
  assert.match(generated, /connect-src 'self' https:\/\/openrouter\.ai https:\/\/my-project-api-123\.europe-west1\.run\.app;/)
  assert.doesNotMatch(generated, /\*\.(?:hf\.space|run\.app)|\.run\.app\/backend/)
  assert.match(generated, /worker-src 'self' blob:/)
  assert.match(generated, /Content-Type: application\/javascript/)
})

test('same-origin and loopback deployments keep their intended connection policy', () => {
  assert.match(headers.deploymentHeaders(template, ''), /connect-src 'self' https:\/\/openrouter\.ai;/)
  assert.equal(headers.backendConnectOrigin('/proxy'), null)
  assert.equal(headers.backendConnectOrigin('http://localhost:8000/api'), 'http://localhost:8000')
  assert.equal(headers.backendConnectOrigin('http://[::1]:8000'), 'http://[::1]:8000')
})

test('CSP configuration rejects remote HTTP, wildcard hosts, credentials and directive injection', () => {
  for (const invalid of ['http://remote.example', 'https://*.hf.space', 'https://user:password@api.example',
    'https://api.example;script-src', '//api.example', '/\\api.example', 'https://api.example/\nscript-src']) {
    assert.throws(() => headers.backendConnectOrigin(invalid), /VITE_API_URL|Invalid URL/)
  }
})
