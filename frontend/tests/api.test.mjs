import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createTestServer } from './helpers/vite.mjs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

let server
let api
let auth
let sources
let graphUtils
let queries
let citations
const originalFetch = globalThis.fetch

before(async () => {
  server = await createTestServer({
    define: { 'import.meta.env.VITE_API_URL': JSON.stringify('https://api.example.test///') },
  })
  api = await server.ssrLoadModule('/src/utils/api.ts')
  auth = await server.ssrLoadModule('/src/utils/auth.ts')
  sources = await server.ssrLoadModule('/src/utils/source.ts')
  graphUtils = await server.ssrLoadModule('/src/utils/graph.ts')
  queries = await server.ssrLoadModule('/src/utils/query.ts')
  citations = await server.ssrLoadModule('/src/utils/citations.ts')
})

after(async () => {
  globalThis.fetch = originalFetch
  await server?.close()
})

test('backend and illustration URLs normalize base slashes without corrupting absolute assets', () => {
  assert.equal(api.apiUrl('/api/graph'), 'https://api.example.test/api/graph')
  assert.equal(api.resolveAssetUrl('/static/illustrations/chart.png'), 'https://api.example.test/static/illustrations/chart.png')
  assert.equal(api.resolveAssetUrl('https://images.example.test/chart.png'), null)
  assert.equal(api.resolveAssetUrl(undefined), null)
})

test('successive persisted edits retain earlier fields when the graph refresh fails', () => {
  const original = { nodes: [{ id: 'node-1', name: 'Original', summary: 'Old summary', node_type: 'topic', source_ids: ['source-1'] }], edges: [] }
  const firstSave = graphUtils.applyNodeUpdates(original, 'node-1', { summary: 'Updated summary' })
  const secondSave = graphUtils.applyNodeUpdates(firstSave, 'node-1', { name: 'Renamed' })
  assert.equal(secondSave.nodes[0].summary, 'Updated summary')
  assert.equal(secondSave.nodes[0].name, 'Renamed')
  assert.equal(original.nodes[0].summary, 'Old summary')
  assert.deepEqual(secondSave.nodes[0].source_ids, ['source-1'])
})

test('reupload invalidates selected-node evidence even when the node id stays the same', () => {
  const oldEvidence = [{ id: 'old-chunk', text: 'Rubisco fixes carbon dioxide.', source: 'review-course', modality: 'pdf' }]
  const cache = { nodeId: 'photosynthesis', version: 1, evidence: oldEvidence }
  assert.deepEqual(graphUtils.evidenceForNode(cache, 'photosynthesis', 2), [])
  assert.deepEqual(graphUtils.evidenceForNode(cache, 'another-node', 1), [])
  const replacement = [{ ...oldEvidence[0], id: 'replacement-chunk', text: 'The Calvin cycle occurs in the stroma.' }]
  assert.deepEqual(graphUtils.evidenceForNode({ ...cache, version: 2, evidence: replacement }, 'photosynthesis', 2), replacement)
})

test('source links preserve encoded document names and PDF/audio deep links', () => {
  assert.equal(sources.sourceUrl('lecture #1', 'pdf', 4), 'https://api.example.test/api/source/lecture%20%231#page=4')
  assert.equal(sources.sourceUrl('lecture', 'audio', undefined, '1:02:03'), 'https://api.example.test/api/source/lecture#t=3723')
  assert.equal(sources.sourceUrl('lecture', 'audio', undefined, 'invalid'), 'https://api.example.test/api/source/lecture')
})

test('markdown images cannot automatically send document text to external hosts', () => {
  for (const source of ['https://attacker.test/image?private=context', '//attacker.test/image', 'data:image/svg+xml,<svg/>', '/static/illustrations/../../other/image.png']) {
    assert.equal(sources.trustedMarkdownImageUrl(source), null)
  }
  assert.equal(sources.trustedMarkdownImageUrl('/static/illustrations/chart.png'), 'https://api.example.test/static/illustrations/chart.png')
  assert.equal(sources.trustedMarkdownImageUrl('https://api.example.test/api/source/lecture'), 'https://api.example.test/api/source/lecture')
})

test('backend requests include session cookies and preserve cancellation signals', async () => {
  const controller = new AbortController()
  let request
  globalThis.fetch = async (url, options) => {
    request = { url, options }
    return new Response('{}', { status: 200 })
  }
  await api.apiFetch('/api/query', { method: 'POST', json: { query: 'What is attention?' }, signal: controller.signal })
  assert.equal(request.url, 'https://api.example.test/api/query')
  assert.equal(request.options.credentials, 'include')
  assert.equal(request.options.signal, controller.signal)
  assert.equal(request.options.headers.get('Content-Type'), 'application/json')
  assert.deepEqual(JSON.parse(request.options.body), { query: 'What is attention?' })
})

test('multipart uploads keep their browser-generated content type', async () => {
  const body = new FormData()
  body.append('slides', 'lecture.pdf')
  let request
  globalThis.fetch = async (_url, options) => {
    request = options
    return new Response('{}', { status: 202 })
  }
  await api.apiFetch('/api/ingest', { method: 'POST', body })
  assert.equal(request.body, body)
  assert.equal(request.headers.has('Content-Type'), false)
})

test('expired sessions notify access screens, but login failures and removed listeners do not', async () => {
  let notifications = 0
  const unsubscribe = api.onAuthRequired(() => { notifications++ })
  globalThis.fetch = async () => new Response('{"detail":"Session expired"}', { status: 401 })
  try {
    await api.apiFetch('/api/graph')
    await api.apiFetch('/api/auth/session', { skipAuthCheck: true })
    assert.equal(notifications, 1)
  } finally {
    unsubscribe()
  }
  await api.apiFetch('/api/graph')
  assert.equal(notifications, 1)
})

test('permanent missing resources and validation errors are not retryable', () => {
  for (const status of [400, 401, 403, 404, 422]) {
    assert.equal(api.isRetryableError(new api.ApiError(status, 'Permanent failure')), false)
  }
  for (const status of [408, 429, 500, 503]) {
    assert.equal(api.isRetryableError(new api.ApiError(status, 'Temporary failure')), true)
  }
  assert.equal(api.isRetryableError(new DOMException('Cancelled', 'AbortError')), false)
})

test('backend validation errors become readable typed API errors', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: [{ msg: 'File is too large' }] }), { status: 422 })
  await assert.rejects(api.apiJson('/api/upload/presign'), (error) => {
    assert.equal(error.status, 422)
    assert.match(error.message, /File is too large/)
    return true
  })
})

test('login exchanges the key in a JSON body rather than a URL', async () => {
  let request
  globalThis.fetch = async (url, options) => {
    request = { url, options }
    return new Response(null, { status: 204 })
  }
  await auth.createSession('test-only-access-key')
  assert.equal(request.url, 'https://api.example.test/api/auth/session')
  assert.deepEqual(JSON.parse(request.options.body), { token: 'test-only-access-key' })
  assert.equal(request.options.credentials, 'include')
})

test('patched KaTeX renders equations while keeping untrusted HTML and links disabled', () => {
  const html = renderToStaticMarkup(createElement(ReactMarkdown, {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: false }]],
    children: 'Inline $x^2$ and a fraction $\\frac{1}{2}$.',
  }))
  assert.equal((html.match(/class="katex"/g) ?? []).length, 2)
  assert.doesNotMatch(html, /katex-error/)
  const untrusted = renderToStaticMarkup(createElement(ReactMarkdown, {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: false }]],
    children: '$\\href{javascript:alert(1)}{Click}$\n<script>steal()</script>',
  }))
  assert.doesNotMatch(untrusted, /href="javascript:/i)
  assert.doesNotMatch(untrusted, /<script/i)
})

test('Hybrid requests preserve Fast/Thorough choice and every dense/graph/both evidence tag', async () => {
  const evidence = ['dense', 'graph', 'both'].map((retrieval_source, index) => ({ id: `chunk-${index}`, source: 'course', text: `Passage ${index}`, modality: 'pdf', page: index + 1, retrieval_source }))
  const payload = { role: 'assistant', content: 'A supported answer [1] [2].', routing_mode: 'hybrid', evidence, hops: ['biology', 'enzyme'] }
  const requests = []
  globalThis.fetch = async (_url, options) => { requests.push(JSON.parse(options.body)); return Response.json(payload) }
  for (const queryMode of ['fast', 'thorough']) {
    const result = await queries.queryWorkspace('Explain enzymes', { routingMode: 'hybrid', queryMode, maxSources: 8, minRelevancy: 0.3 })
    assert.deepEqual(result, payload)
    assert.deepEqual(result.evidence.map((chunk) => chunk.retrieval_source), ['dense', 'graph', 'both'])
  }
  assert.deepEqual(requests.map((body) => [body.routing_mode, body.query_mode, body.max_sources, body.min_relevancy]), [['hybrid', 'fast', 8, 0.3], ['hybrid', 'thorough', 8, 0.3]])
})

test('citation numbering maps to the exact evidence item and invalid numbers never select another source', () => {
  assert.equal(citations.citationIndex(1, 3), 0)
  assert.equal(citations.citationIndex(3, 3), 2)
  for (const number of [0, -1, 4, 1.5]) assert.equal(citations.citationIndex(number, 3), null)
  assert.equal(citations.injectCitationLinks('Answer [2] and [1].'), 'Answer [2](#citation-2) and [1](#citation-1).')
  assert.equal(citations.injectCitationLinks('[1](https://example.test)'), '[1](https://example.test)')
})

test('expanded source cards expose the complete passage with filename, location and retrieval provenance', async () => {
  const { default: EvidenceChip } = await server.ssrLoadModule('/src/components/EvidenceChip.tsx')
  const text = 'Supported context. '.repeat(50) + 'EXACT END OF THE CITED PASSAGE'
  const chunk = { id: 'citation1', source: 'course_pdf_privatehash', source_filename: 'Biology notes.pdf', modality: 'pdf', text, page: 2, page_end: 3, section_title: 'Enzyme catalysis', retrieval_source: 'both' }
  const expanded = renderToStaticMarkup(createElement(EvidenceChip, { chunk, index: 0, expanded: true }))
  assert.match(expanded, /Biology notes\.pdf/)
  assert.match(expanded, /pp\. 2–3 · Enzyme catalysis/)
  assert.match(expanded, /EXACT END OF THE CITED PASSAGE/)
  assert.match(expanded, /aria-expanded="true"/)
  assert.doesNotMatch(expanded, /line-clamp-3/)
  assert.match(expanded, />Both<\/span>/)
  for (const [retrieval_source, label] of [['dense', 'Dense'], ['graph', 'Graph']]) {
    const markup = renderToStaticMarkup(createElement(EvidenceChip, { chunk: { ...chunk, retrieval_source }, index: 0 }))
    assert.ok(markup.includes(`>${label}</span>`))
    assert.match(markup, /aria-expanded="false"/)
  }
  assert.equal(sources.sourceLocation({ ...chunk, modality: 'office', slide_number: 4, page: null, page_end: null }), 'Slide 4 · Enzyme catalysis')
})
