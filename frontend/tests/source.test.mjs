import 'fake-indexeddb/auto'
import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createTestServer } from './helpers/vite.mjs'
import { Window } from 'happy-dom'
import { zipSync, unzipSync, strFromU8, strToU8 } from 'fflate'
import React, { act } from 'react'

const browser = new Window({ url: 'https://lumin.example/' })
browser.document.write('<!doctype html><html><head></head><body></body></html>')
Object.assign(globalThis, { window: browser, document: browser.document, localStorage: browser.localStorage,
  HTMLElement: browser.HTMLElement, Event: browser.Event, CustomEvent: browser.CustomEvent,
  IS_REACT_ACT_ENVIRONMENT: true })
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: browser.navigator })
let server, highlight, source, storage, SourceViewer, citations, createRoot
before(async () => {
  ;({ createRoot } = await import('react-dom/client'))
  server = await createTestServer()
  highlight = await server.ssrLoadModule('/src/utils/sourceHighlight.ts')
  source = await server.ssrLoadModule('/src/utils/source.ts')
  storage = await server.ssrLoadModule('/src/storage/db.ts')
  citations = await server.ssrLoadModule('/src/utils/citations.ts')
  await server.ssrLoadModule('/src/components/RichText.tsx')
  SourceViewer = (await server.ssrLoadModule('/src/components/source/SourceViewer.tsx')).default
  await storage.initializeStorage()
})
after(async () => { storage?.db.close(); await storage?.db.delete(); await server?.close(); await browser.happyDOM.close() })

test('original text matching handles Word styled runs, PDF whitespace and ligatures without changing text', () => {
  const element = document.createElement('div')
  element.innerHTML = '<p>Intro. The <strong>efficient enzyme</strong> catalyzes <em>reactions.</em> End.</p>'
  const original = element.textContent
  const marks = highlight.highlightSourceText(element, 'The **efficient enzyme** catalyzes _reactions_.')
  assert.equal(marks.map((mark) => mark.textContent).join(''), 'The efficient enzyme catalyzes reactions.')
  assert.equal(element.textContent, original)
  assert.ok(element.querySelector('strong mark'))
  assert.deepEqual(highlight.findSourceTextRange('The eﬃcient\n enzyme “works”.', 'The efficient enzyme "works".'), { start: 0, end: 28 })
  assert.equal(highlight.highlightSourceText(element, 'not present').length, 0)
  assert.equal(element.querySelectorAll('mark').length, 0, 'A replacement target clears the previous marks')
})

test('repeated quotes require unique full-passage context instead of selecting the first document occurrence', () => {
  const element = document.createElement('div')
  element.innerHTML = '<section>First subject. Repeat this sentence. Earlier detail.</section><section>Target subject. Repeat this sentence. Specific conclusion.</section>'
  assert.equal(highlight.findSourceTextRange(element.textContent, 'Repeat this sentence.'), null)
  const marks = highlight.highlightSourceText(element, 'Repeat this sentence.', false, 'Target subject. Repeat this sentence. Specific conclusion.')
  assert.equal(marks.length, 1)
  assert.equal(marks[0].closest('section'), element.lastElementChild)
  assert.equal(highlight.highlightSourceText(element, 'Repeat this sentence.', false, 'Repeat this sentence.').length, 0)
})

test('legacy multi-page citations resolve unique passage context and never invent the first repeated page', () => {
  const pages = [{ page: 1, text: 'Earlier context. Repeat this sentence. Old detail.' },
    { page: 2, text: 'Target context. Repeat this sentence. Specific conclusion.' }]
  assert.equal(highlight.resolveSourceQuotePage(pages, 'Repeat this sentence.', 'Target context. Repeat this sentence. Specific conclusion.'), 2)
  assert.equal(highlight.resolveSourceQuotePage(pages, 'Repeat this sentence.', 'Repeat this sentence.'), null)
  assert.equal(highlight.resolveSourceQuotePage(pages, 'Specific conclusion.', 'Specific conclusion.'), 2)
})

test('malformed 100k-character Markdown markers remain bounded and identifiers keep literal underscores', () => {
  const started = performance.now()
  for (const marker of ['*a ', '`a ', '__a ']) {
    const text = marker.repeat(Math.ceil(100_000 / marker.length)).slice(0, 100_000)
    assert.equal(highlight.findSourceTextRange(text, 'No actual source match.'), null)
  }
  assert.ok(performance.now() - started < 2000, 'Fixed delimiter scans should remain linear on malformed input')
  assert.equal(highlight.findSourceTextRange('alpha_beta_gamma', 'alpha beta gamma'), null)
})

test('highlighting hostile source text creates escaped DOM text and transparent PDF marks only', () => {
  const element = document.createElement('div')
  const quote = '<img src=x onerror=steal()> & ordinary text'
  element.textContent = `Prefix 😀 ${quote} suffix`
  const marks = highlight.highlightSourceText(element, quote, true)
  assert.equal(marks.length, 1)
  assert.equal(marks[0].textContent, quote)
  assert.equal(marks[0].style.color, 'transparent')
  assert.equal(element.querySelector('img'), null)
  assert.equal(element.textContent, `Prefix 😀 ${quote} suffix`)
})

test('validated citation offsets use Python Unicode code points and never highlight an invalid provider quote', () => {
  const text = '😀 Intro. Exact supporting sentence. Tail.'
  const quote = 'Exact supporting sentence.'
  const quote_start = Array.from(text.slice(0, text.indexOf(quote))).length
  const evidence = { id: 'quote', source: 'doc', modality: 'pdf', text, quote, quote_start, quote_end: quote_start + Array.from(quote).length }
  assert.equal(source.validatedEvidenceQuote(evidence), quote)
  assert.equal(source.validatedEvidenceQuote({ ...evidence, quote_start: quote_start + 1 }), null)
  assert.equal(source.validatedEvidenceQuote({ ...evidence, quote: 'Invented sentence.' }), null)
  assert.equal(source.validatedEvidenceQuote({ ...evidence, quote_start: null, quote_end: null }), quote)
  assert.equal(source.validatedEvidenceQuote({ ...evidence, quote: null }), null)
})

test('Word archives are inspected before rendering, including unsafe paths and expansion limits', () => {
  const valid = zipSync({ 'word/document.xml': strToU8('<document/>') })
  assert.doesNotThrow(() => highlight.validateWordPreview(valid))
  assert.equal(strFromU8(unzipSync(highlight.validateWordPreview(valid))['word/document.xml']), '<document/>', 'Renderer input is rebuilt from the actual bounded entries')
  assert.throws(() => highlight.validateWordPreview(zipSync({ 'other.xml': strToU8('x') })), /does not contain/)
  assert.throws(() => highlight.validateWordPreview(zipSync({ 'word/document.xml': strToU8('x'), '../escape.xml': strToU8('x') })), /unsafe archive path/)
  const oversized = valid.slice()
  const view = new DataView(oversized.buffer)
  for (let offset = 0; offset < oversized.length - 28; offset++) if (view.getUint32(offset, true) === 0x02014b50) view.setUint32(offset + 24, 101 * 1024 * 1024, true)
  assert.throws(() => highlight.validateWordPreview(oversized), /preview size limit/)
})

test('Word preflight counts actual decompressed bytes even when both ZIP headers lie about their sizes', () => {
  const forge = (size) => {
    const archive = zipSync({ 'word/document.xml': new Uint8Array(size).fill(65) })
    const view = new DataView(archive.buffer)
    for (let offset = 0; offset < archive.length - 28; offset++) {
      const signature = view.getUint32(offset, true)
      if (signature === 0x02014b50) view.setUint32(offset + 24, 64, true)
      if (signature === 0x04034b50) view.setUint32(offset + 22, 64, true)
    }
    return archive
  }
  assert.throws(() => highlight.validateWordPreview(forge(2 * 1024 * 1024)), /incorrect expanded size/)
  assert.throws(() => highlight.validateWordPreview(forge(17 * 1024 * 1024)), /incorrect expanded size/, 'Reject the first excessive output chunk instead of expanding the whole document')
})

test('Word renderer receives normalized bounded entries rather than attacker-controlled ZIP offsets', () => {
  const content = '<document>The actually validated Word content.</document>'
  const archive = zipSync({ 'word/document.xml': strToU8(content) })
  const view = new DataView(archive.buffer)
  for (let offset = 0; offset < archive.length - 46; offset++) if (view.getUint32(offset, true) === 0x02014b50) view.setUint32(offset + 42, 1, true)
  const normalized = highlight.validateWordPreview(archive)
  assert.equal(strFromU8(unzipSync(normalized)['word/document.xml']), content)
  assert.notDeepEqual(normalized, archive)
})

test('retrieved passages and actually cited passages have distinct stable evidence numbering', () => {
  assert.deepEqual([...citations.citedEvidenceIndices('Answer [2], repeated [2], later [4], invalid [0] [9], link [1](https://example.test)', 4)], [1, 3])
  assert.equal(citations.citedEvidenceIndices('No citation in this answer.', 6).size, 0)
})

test('source viewer without an original opens an accessible modal and keeps the complete cited passage', async () => {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const text = 'Complete passage with its original supporting sentence. '.repeat(15)
  const chunk = { id: 'missing-original', source: 'no-file-on-this-device', source_filename: 'Reference.pdf', modality: 'pdf', page: 3, text }
  await act(async () => {
    root.render(React.createElement(SourceViewer, { chunk, onClose: () => {} }))
    await new Promise((resolve) => setTimeout(resolve, 20))
  })
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)) })
  const dialog = host.querySelector('dialog')
  assert.equal(dialog.open, true)
  assert.equal(dialog.getAttribute('aria-modal'), 'true')
  assert.match(host.textContent, /original file is not saved on this device/)
  assert.match(host.textContent, /p\. 3/)
  assert.ok(host.textContent.includes(text.trim()))
  assert.equal(host.querySelector('iframe'), null)
  await act(async () => root.unmount())
  host.remove()
})

test('app shares query settings and keeps an in-flight clear scoped to its original workspace', async () => {
  const originalFetch = globalThis.fetch
  const calls = []
  let finishClear, clearedWorkspace
  const suggestion = 'Explain the fixture question'
  const bundle = new Blob([zipSync({ 'manifest.json': strToU8('{"format":"lumin","version":1,"workspace_version":0}'),
    'chunks.jsonl': strToU8(''), 'graph.json': strToU8('{"nodes":[],"edges":[]}') })])
  globalThis.fetch = async (url, options = {}) => {
    const path = new URL(String(url), browser.location.href).pathname
    if (path === '/api/health') return Response.json({ features: { hosted: false } })
    if (path === '/api/auth/status') return Response.json({ authenticated: true, token_required: false })
    if (path === '/api/workspace/status') return Response.json({ loaded: true, version: 0 })
    if (path === '/api/workspace/import') return Response.json({ version: 0 })
    if (path === '/api/workspace/export') return new Response(bundle, { headers: { 'X-Workspace-Version': '0' } })
    if (path === '/api/data' && options.method === 'DELETE') {
      clearedWorkspace = options.headers.get('X-Workspace-Id')
      return new Promise((resolve) => { finishClear = resolve })
    }
    if (path === '/api/graph') return Response.json({ nodes: [], edges: [], association_version: 1 })
    if (path === '/api/suggestions') return Response.json([suggestion])
    if (path === '/api/query') {
      const body = JSON.parse(options.body)
      calls.push(body)
      return Response.json({ role: 'assistant', content: 'Fixture answer.', routing_mode: body.routing_mode ?? 'dense', evidence: [], hops: [] })
    }
    throw new Error(`Unexpected feature-check request: ${path}`)
  }
  await server.ssrLoadModule('/src/components/GraphPanel.tsx')
  const App = (await server.ssrLoadModule('/src/App.tsx')).default
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const button = (label) => [...host.querySelectorAll('button')].find((element) => element.textContent.trim() === label)
  const waitFor = async (condition) => {
    for (let attempt = 0; attempt < 100 && !condition(); attempt++) await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)) })
    assert.ok(condition(), 'The fixture app should settle into the requested state')
  }
  const input = async (element, value) => {
    const prototype = element.tagName === 'TEXTAREA' ? browser.HTMLTextAreaElement.prototype : browser.HTMLInputElement.prototype
    await act(async () => {
      Object.getOwnPropertyDescriptor(prototype, 'value').set.call(element, value)
      element.dispatchEvent(new browser.Event('input', { bubbles: true }))
    })
  }
  try {
    await act(async () => root.render(React.createElement(App)))
    await waitFor(() => button(suggestion) && button('Upload') && !button('Upload').disabled)
    await act(async () => button(suggestion).click())
    await waitFor(() => calls.length === 1 && host.textContent.includes('Fixture answer.'))
    assert.deepEqual(calls[0], { query: suggestion, query_mode: 'fast' })
    await act(async () => button('Clear').click())
    await waitFor(() => button(suggestion))
    await act(async () => {
      const depth = host.querySelector('#query-depth')
      depth.value = 'thorough'
      depth.dispatchEvent(new browser.Event('change', { bubbles: true }))
    })
    await act(async () => button('Hybrid').click())
    await act(async () => host.querySelector('button[title="Source filtering options"]').click())
    await input(host.querySelector('.lumin-composer input[type="range"]'), '0.55')
    await input(host.querySelector('.lumin-composer input[type="number"]'), '3')
    await act(async () => button(suggestion).click())
    await waitFor(() => calls.length === 2 && host.textContent.includes('Fixture answer.'))
    assert.deepEqual(calls[1], { query: suggestion, query_mode: 'thorough', routing_mode: 'hybrid', max_sources: 3, min_relevancy: 0.55 })
    await input(host.querySelector('textarea'), 'Typed fixture question')
    await act(async () => host.querySelector('.lumin-send-button').click())
    await waitFor(() => calls.length === 3)
    assert.deepEqual(calls[2], { ...calls[1], query: 'Typed fixture question' })

    const originalWorkspace = storage.activeWorkspaceId()
    await storage.saveOriginals([new File(['First workspace original.'], 'first.pdf')])
    const second = await storage.createWorkspace('Clear isolation fixture')
    await storage.saveOriginals([new File(['Keep second workspace original.'], 'second.pdf')], second.id)
    await storage.db.chats.put({ workspaceId: second.id, messages: [{ id: 'second-chat', role: 'assistant', content: 'Keep second workspace chat.', timestamp: new Date() }] })
    await act(async () => button('Clear Data').click())
    await act(async () => button('Clear knowledge base').click())
    await waitFor(() => finishClear)
    assert.equal(clearedWorkspace, originalWorkspace)
    assert.equal(host.querySelector('select[aria-label="Knowledge base"]').disabled, true)
    assert.equal(button('Import .lumin').disabled, true)
    // Another tab or an already-started operation can still change the owner.
    await act(async () => storage.selectWorkspace(second.id))
    await waitFor(() => host.textContent.includes('Keep second workspace chat.'))
    await act(async () => { finishClear(Response.json({ ok: true })); await new Promise((resolve) => setTimeout(resolve, 20)) })
    await waitFor(() => button('Clear Data') && !button('Clear Data').disabled)
    assert.equal(await storage.db.rawFiles.where('workspaceId').equals(originalWorkspace).count(), 0)
    assert.equal(await (await storage.db.rawFiles.get(`${second.id}:second.pdf`)).blob.text(), 'Keep second workspace original.')
    assert.ok(host.textContent.includes('Keep second workspace chat.'), 'A completed clear must not reset the newly selected workspace')
  } finally {
    await act(async () => root.unmount())
    host.remove()
    globalThis.fetch = originalFetch
  }
})
