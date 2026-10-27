import 'fake-indexeddb/auto'
import assert from 'node:assert/strict'
import { after, afterEach, before, test } from 'node:test'
import { createHash } from 'node:crypto'
import { createTestServer } from './helpers/vite.mjs'
import { zipSync, unzipSync, strFromU8, strToU8 } from 'fflate'

const memoryStorage = () => { const values = new Map(); return { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, String(value)), removeItem: (key) => values.delete(key) } }
globalThis.localStorage = memoryStorage()
globalThis.sessionStorage = memoryStorage()
const browser = new EventTarget()
Object.assign(browser, { isSecureContext: true, location: { href: 'https://lumin.example/', hostname: 'lumin.example', origin: 'https://lumin.example' } })
browser.history = { replaceState: (_state, _title, url) => { browser.location.href = url } }
globalThis.window = browser
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { storage: { persist: async () => true } } })
const originalFetch = globalThis.fetch
let server, storage, settings, api, workspaces, editor, sources
const emptyBundle = () => new Blob([zipSync({ 'manifest.json': strToU8('{"format_version":1}'), 'chunks.jsonl': strToU8(''), 'graph.json': strToU8('{"nodes":[],"edges":[]}') })], { type: 'application/zip' })
const snapshotBundle = (label, version) => new Blob([zipSync({
  'manifest.json': strToU8(JSON.stringify({ format: 'lumin', version: 1, workspace_version: version })),
  'chunks.jsonl': strToU8(JSON.stringify({ id: 'snapshot-passage', source_id: 'source', text: label })),
  'graph.json': strToU8('{"nodes":[],"edges":[]}'),
})], { type: 'application/zip' })

before(async () => {
  server = await createTestServer({ define: { 'import.meta.env.VITE_API_URL': JSON.stringify('https://api.example.test') } })
  storage = await server.ssrLoadModule('/src/storage/db.ts')
  settings = await server.ssrLoadModule('/src/utils/settings.ts')
  api = await server.ssrLoadModule('/src/utils/api.ts')
  workspaces = await server.ssrLoadModule('/src/utils/workspaces.ts')
  editor = await server.ssrLoadModule('/src/components/graph/useGraphStore.ts')
  sources = await server.ssrLoadModule('/src/utils/source.ts')
  await storage.initializeStorage()
  await settings.loadLlmSettings()
})
afterEach(() => { globalThis.fetch = originalFetch; browser.isSecureContext = true; editor.resetGraphHistory() })
after(async () => { storage.db.close(); await storage.db.delete(); await server.close() })

test('originals are isolated by workspace and source identity matches backend hashing', async () => {
  const first = storage.activeWorkspaceId()
  const filename = 'lecture #1.pdf'
  const digest = createHash('sha256').update(filename).digest('hex').slice(0, 12)
  assert.equal(await storage.sourceIdentity(filename), `lecture__1_pdf_${digest}`)
  await storage.saveOriginals([new File(['PDF'], filename, { type: 'application/pdf' })])
  assert.ok(storage.cachedSourceUrl(`lecture__1_pdf_${digest}`).startsWith('blob:'))
  const second = await storage.createWorkspace('Second')
  await storage.selectWorkspace(second.id)
  assert.equal(storage.cachedSourceUrl(`lecture__1_pdf_${digest}`), undefined)
  await storage.saveOriginals([new File(['other PDF'], filename)])
  assert.equal((await storage.db.rawFiles.where('workspaceId').equals(first).first()).blob.size, 3)
  assert.equal((await storage.db.rawFiles.where('workspaceId').equals(second.id).first()).blob.size, 9)
})

test('upload originals replace a source only after successful job completion', async () => {
  const previous = storage.activeWorkspaceId()
  const workspace = await storage.createWorkspace('Staged source replacement')
  await storage.selectWorkspace(workspace.id)
  const old = new File(['Original supporting sentence.'], 'notes.pdf')
  const replacement = new File(['Replacement supporting sentence.'], 'notes.pdf')
  await storage.saveOriginals([old])
  const current = async () => (await storage.db.rawFiles.get(`${workspace.id}:notes.pdf`)).blob.text()
  let staged = await storage.stageOriginals([replacement])
  assert.equal(await current(), await old.text())
  await storage.discardStagedOriginals(staged) // Rejected/cancelled request.
  assert.equal(await current(), await old.text())
  staged = await storage.stageOriginals([replacement])
  await storage.bindOriginalsToJob(staged, 'failed-job')
  assert.equal(await current(), await old.text(), 'HTTP 202 must not publish originals')
  await storage.settleIngestOriginals('failed-job', null)
  assert.equal(await current(), await old.text())
  staged = await storage.stageOriginals([replacement, new File(['Successfully ingested source.'], 'new.pdf')])
  await storage.bindOriginalsToJob(staged, 'partial-job')
  await storage.settleIngestOriginals('partial-job', [await storage.sourceIdentity('new.pdf'), 'unknown-source-id'])
  assert.equal(await current(), await old.text(), 'A failed replacement in a partially successful batch keeps its original')
  assert.equal(await (await storage.db.rawFiles.get(`${workspace.id}:new.pdf`)).blob.text(), 'Successfully ingested source.')
  staged = await storage.stageOriginals([replacement])
  await storage.bindOriginalsToJob(staged, 'legacy-job')
  await storage.settleIngestOriginals('legacy-job', undefined)
  assert.equal(await current(), await old.text(), 'An old server without positive source IDs cannot replace originals')
  staged = await storage.stageOriginals([replacement])
  await storage.bindOriginalsToJob(staged, 'successful-job')
  assert.equal(await storage.db.settings.get(staged), undefined)
  const pending = await storage.db.settings.get(`pending-originals:${workspace.id}:job:successful-job`)
  assert.equal(await pending.value.files[0].blob.text(), await replacement.text(), 'Staging remains durable until a successful poll')
  await storage.settleIngestOriginals('successful-job', [await storage.sourceIdentity('notes.pdf')])
  assert.equal(await current(), await replacement.text())
  assert.equal(await storage.db.settings.get(pending.id), undefined)
  await storage.selectWorkspace(previous)
})

test('original publication remains scoped across workspace switches and rejects cleared generations', async () => {
  const previous = storage.activeWorkspaceId()
  const first = await storage.createWorkspace('First upload owner')
  await storage.selectWorkspace(first.id)
  const staged = await storage.stageOriginals([new File(['First source'], 'same.pdf')])
  await storage.bindOriginalsToJob(staged, 'first-job')
  const second = await storage.createWorkspace('Second upload owner')
  await storage.selectWorkspace(second.id)
  await storage.saveOriginals([new File(['Second source'], 'same.pdf')])
  await storage.settleIngestOriginals('first-job', [await storage.sourceIdentity('same.pdf')], first.id)
  assert.equal(await (await storage.db.rawFiles.get(`${first.id}:same.pdf`)).blob.text(), 'First source')
  assert.equal(await (await storage.db.rawFiles.get(`${second.id}:same.pdf`)).blob.text(), 'Second source')
  const cancelled = await storage.stageOriginals([new File(['Stale replacement'], 'same.pdf')])
  await storage.bindOriginalsToJob(cancelled, 'late-job')
  const pendingId = `pending-originals:${second.id}:job:late-job`
  const stale = await storage.db.settings.get(pendingId)
  await storage.clearWorkspaceData(second.id)
  assert.equal(await storage.db.settings.get(pendingId), undefined)
  await storage.db.settings.put(stale) // A stale writer in a different tab.
  await storage.settleIngestOriginals('late-job', [await storage.sourceIdentity('same.pdf')], second.id)
  assert.equal(await storage.db.rawFiles.where('workspaceId').equals(second.id).count(), 0)
  assert.equal(await storage.db.settings.get(pendingId), undefined)
  await storage.selectWorkspace(previous)
})

test('imported Blob MIME cannot turn a PDF or HTML source into app-origin executable content', () => {
  const hostile = new Blob(['<script>steal()</script>'], { type: 'text/html' })
  assert.equal(storage.safeSourceBlob('notes.pdf', hostile).type, 'application/pdf')
  assert.equal(storage.safeSourceBlob('notes.html', hostile).type, 'text/plain')
  assert.equal(storage.safeSourceBlob('notes.svg', hostile).type, 'text/plain')
})

test('BYOK headers use memory credentials while remember-off keeps the browser database key-free', async () => {
  await settings.saveLlmSettings({ provider: 'anthropic', key: 'test-user-key', fastModel: 'fast-model', smartModel: 'smart-model', remember: false })
  assert.equal((await storage.db.settings.get('llm')).value.key, '')
  let headers
  globalThis.fetch = async (_url, options) => { headers = options.headers; return Response.json({ ok: true }) }
  await api.apiJson('/api/query', { method: 'POST', json: { query: 'Question' } })
  assert.equal(headers.get('X-LLM-Key'), 'test-user-key')
  assert.equal(headers.get('X-LLM-Provider'), 'anthropic')
  assert.equal(headers.get('X-Workspace-Id'), storage.activeWorkspaceId())
  assert.equal(headers.get('X-LLM-Model-Smart'), 'smart-model')
})

test('provider keys are attached only to operations that call an AI provider', async () => {
  const seen = []
  globalThis.fetch = async (url, options) => { seen.push([url, options.headers.get('X-LLM-Key')]); return Response.json({}) }
  const compute = ['/api/query', '/api/ingest']
  const reads = ['/api/health', '/api/graph', '/api/workspace/export', '/api/workspace/import', '/api/workspace/status', '/api/graph/ops', '/api/quiz/generate', '/api/quiz/quiz1/grade', '/api/quiz/quiz1/deep-dive/question1']
  for (const path of [...compute, ...reads]) await api.apiFetch(path)
  assert.ok(seen.slice(0, compute.length).every(([, key]) => key === 'test-user-key'))
  assert.ok(seen.slice(compute.length).every(([, key]) => key === null))
})

test('workspace eviction triggers one shared hydrate and retries original requests with the same headers', async () => {
  const id = storage.activeWorkspaceId()
  await storage.db.workspaces.update(id, { bundle: emptyBundle() })
  let loaded = false, imports = 0, queries = 0
  globalThis.fetch = async (url, options) => {
    assert.equal(options.headers.get('X-Workspace-Id'), id)
    if (url.endsWith('/api/workspace/import')) {
      imports++
      await new Promise((resolve) => setImmediate(resolve))
      assert.ok(options.body.get('file') instanceof Blob)
      loaded = true
      return Response.json({ version: 1 })
    }
    queries++
    assert.deepEqual(JSON.parse(options.body), { query: 'Saved course' })
    return loaded ? Response.json({ content: 'Recovered' }) : Response.json({ code: 'workspace_not_loaded' }, { status: 409 })
  }
  const result = await Promise.all([1, 2].map(() => api.apiJson('/api/query', { method: 'POST', json: { query: 'Saved course' } })))
  assert.equal(imports, 1)
  assert.equal(queries, 4)
  assert.equal(result[0].content, 'Recovered')
})

test('a newly created workspace hydrates explicitly as empty', async () => {
  const record = await storage.createWorkspace('Blank')
  let empty
  globalThis.fetch = async (_url, options) => { empty = options.body.get('empty'); return Response.json({ version: 0 }) }
  await workspaces.hydrateWorkspace(record.id)
  assert.equal(empty, 'true')
})

test('unsafe HTTP pages block credential transmission and backend redirects are refused', async () => {
  let calls = 0, redirect
  globalThis.fetch = async (_url, options) => { calls++; redirect = options.redirect; return Response.json({}) }
  browser.isSecureContext = false
  await assert.rejects(api.apiFetch('/api/query'), /HTTPS/)
  assert.equal(calls, 0)
  browser.isSecureContext = true
  await api.apiFetch('/api/query')
  assert.equal(redirect, 'error')
})

test('graph undo sends server inverses and redo resubmits the original stable node id', async () => {
  const forward = { op: 'add_node', node: { id: 'stable-note-id', name: 'Note', node_type: 'note', body: '' } }
  const inverse = { op: 'delete_node', id: 'stable-note-id' }
  const requests = []
  globalThis.fetch = async (_url, options) => { const { ops } = JSON.parse(options.body); requests.push(ops); return Response.json({ version: requests.length, inverses: ops[0].op === 'add_node' ? [inverse] : [forward] }) }
  await editor.applyGraphOps([forward])
  await editor.undoGraphEdit()
  await editor.redoGraphEdit()
  assert.deepEqual(requests, [[forward], [inverse], [forward]])
  assert.equal(editor.useGraphStore.getState().steps.length, 1)
})

test('a rejected graph inverse does not advance local undo history', async () => {
  let reject = false
  globalThis.fetch = async () => reject ? Response.json({ detail: 'Workspace busy' }, { status: 409 }) : Response.json({ version: 1, inverses: [{ op: 'delete_node', id: 'a' }] })
  await editor.applyGraphOps([{ op: 'add_node', node: { id: 'a', name: 'A', node_type: 'note' } }])
  reject = true
  await assert.rejects(editor.undoGraphEdit(), /Workspace busy/)
  assert.equal(editor.useGraphStore.getState().steps.length, 1)
  assert.equal(editor.useGraphStore.getState().busy, false)
})

test('automatic layout preserves add-note Undo and its Redo while manual movement remains undoable', async () => {
  const id = 'note-before-automatic-layout'
  const add = { op: 'add_node', node: { id, name: 'Study note', node_type: 'note', body: '' } }
  const remove = { op: 'restore_nodes', ids: [id], nodes: [], edges: [] }
  const positionInverse = { op: 'set_positions', positions: [{ id: 'existing-node', x: 0, y: 0 }] }
  const requests = []
  let present = false, changes = 0
  const changed = () => { changes++ }
  browser.addEventListener('lumin-graph-changed', changed)
  globalThis.fetch = async (_url, options) => {
    const { ops } = JSON.parse(options.body)
    requests.push(ops)
    if (ops[0].op === 'add_node') present = true
    if (ops[0].op === 'restore_nodes') present = false
    return Response.json({ version: requests.length, inverses: [ops[0].op === 'add_node' ? remove : positionInverse] })
  }
  try {
    await editor.applyGraphOps([add])
    await editor.applyGraphOps([{ op: 'set_positions', positions: [{ id, x: 12, y: 34 }] }], { recordHistory: false })
    assert.equal(editor.useGraphStore.getState().steps.length, 1)
    await editor.undoGraphEdit()
    assert.equal(present, false, 'One Undo removes the note after automatic positioning')
    assert.deepEqual(requests[2], [remove])
    assert.equal(editor.useGraphStore.getState().steps.length, 0)
    assert.equal(editor.useGraphStore.temporal.getState().futureStates.length, 1)
    await editor.applyGraphOps([{ op: 'set_positions', positions: [{ id: 'existing-node', x: 5, y: 6 }] }], { recordHistory: false })
    assert.equal(editor.useGraphStore.getState().steps.length, 0)
    assert.equal(editor.useGraphStore.temporal.getState().futureStates.length, 1, 'Layout after Undo preserves Redo')
    await editor.redoGraphEdit()
    assert.equal(present, true)
    assert.deepEqual(requests[4], [add])
    await editor.applyGraphOps([{ op: 'set_positions', positions: [{ id, x: 20, y: 30 }] }])
    assert.equal(editor.useGraphStore.getState().steps.length, 2, 'Manual movement still enters history')
    await editor.undoGraphEdit()
    assert.deepEqual(requests[6], [positionInverse])
    assert.equal(present, true, 'Undoing a manual move keeps the note')
    assert.equal(changes, requests.length, 'Background layout still publishes graph refresh events')
  } finally { browser.removeEventListener('lumin-graph-changed', changed) }
})

test('.lumin round trip keeps originals and chat local and omits provider credentials', async () => {
  const id = storage.activeWorkspaceId()
  const pending = await storage.stageOriginals([new File(['Uncommitted private source'], 'pending-upload.pdf')])
  await storage.db.chats.put({ workspaceId: id, messages: [{ id: 'chat1', role: 'user', content: 'My private chat', timestamp: new Date() }] })
  let downloadUrl, uploadedEntries
  globalThis.document = { createElement: () => ({ set href(value) { downloadUrl = value }, click() {} }) }
  globalThis.fetch = async (url, options) => {
    if (url.endsWith('/api/workspace/export')) return new Response(emptyBundle())
    if (url.endsWith('/api/workspace/status')) return Response.json({ loaded: true, version: 8 })
    if (url.endsWith('/api/workspace/import')) {
      uploadedEntries = unzipSync(new Uint8Array(await options.body.get('file').arrayBuffer()))
      return Response.json({ version: 8 })
    }
    throw new Error(`Unexpected request: ${url}`)
  }
  await workspaces.exportWorkspace()
  const archive = await (await originalFetch(downloadUrl)).blob()
  const entries = unzipSync(new Uint8Array(await archive.arrayBuffer()))
  const browserData = JSON.parse(strFromU8(entries['browser.json']))
  assert.equal(browserData.chats[0].content, 'My private chat')
  assert.ok(browserData.originals.length)
  assert.ok(browserData.originals.every((original) => original.name !== 'pending-upload.pdf'))
  assert.ok(Object.keys(entries).every((path) => !path.startsWith('pending-originals:')))
  await storage.discardStagedOriginals(pending)
  assert.equal(strFromU8(entries['browser.json']).includes('test-user-key'), false)
  await workspaces.importWorkspace(new File([archive], 'roundtrip.lumin'))
  assert.notEqual(storage.activeWorkspaceId(), id)
  assert.equal(uploadedEntries['browser.json'], undefined)
  assert.ok(Object.keys(uploadedEntries).every((path) => !path.startsWith('originals/')))
  assert.equal((await storage.db.chats.get(storage.activeWorkspaceId())).messages[0].content, 'My private chat')
  assert.equal((await storage.db.rawFiles.where('workspaceId').equals(storage.activeWorkspaceId()).toArray()).length, browserData.originals.length)
  assert.equal(browserData.quiz, undefined)
})

test('archive traversal is rejected before a workspace is created', async () => {
  const count = await storage.db.workspaces.count()
  const malicious = zipSync({ '../outside': strToU8('no'), 'manifest.json': strToU8('{}'), 'graph.json': strToU8('{}') })
  await assert.rejects(workspaces.importWorkspace(new File([malicious], 'bad.lumin')), /unsafe path/)
  assert.equal(await storage.db.workspaces.count(), count)
})

test('forged stored ZIP sizes are rejected before importing a workspace or contacting the backend', async () => {
  const beforeCount = await storage.db.workspaces.count()
  const bytes = zipSync({ 'manifest.json': strToU8('{}'), 'graph.json': strToU8('{"nodes":[],"edges":[]}'),
    'assets/payload.bin': [new Uint8Array(64 * 1024).fill(65), { level: 0 }] })
  const view = new DataView(bytes.buffer)
  for (let offset = 0; offset < bytes.length - 46; offset++) if (view.getUint32(offset, true) === 0x02014b50) {
    const nameLength = view.getUint16(offset + 28, true)
    const name = strFromU8(bytes.subarray(offset + 46, offset + 46 + nameLength))
    if (name === 'assets/payload.bin') view.setUint32(offset + 24, 0, true)
  }
  let requests = 0
  globalThis.fetch = async () => { requests++; return Response.json({ version: 1 }) }
  await assert.rejects(workspaces.importWorkspace(new File([bytes], 'forged.lumin')), /incorrect expanded size/)
  assert.equal(requests, 0)
  assert.equal(await storage.db.workspaces.count(), beforeCount)
})


test('malformed browser chat metadata is rejected before creating a workspace', async () => {
  const count = await storage.db.workspaces.count()
  const malformed = zipSync({ 'manifest.json': strToU8('{}'), 'graph.json': strToU8('{}'), 'browser.json': strToU8(JSON.stringify({ name: 'Bad', chats: [{ id: 'a', role: 'assistant', content: { injected: true }, timestamp: 'not-a-date' }] })) })
  await assert.rejects(workspaces.importWorkspace(new File([malformed], 'bad.lumin')))
  assert.equal(await storage.db.workspaces.count(), count)
})

test('OpenRouter connection uses PKCE and exchanges the code without persisting the returned key', async () => {
  let redirectUrl, exchange
  browser.location.assign = (url) => { redirectUrl = new URL(url) }
  await settings.connectOpenRouter()
  const saved = JSON.parse(sessionStorage.getItem('lumin-openrouter-pkce'))
  assert.equal(redirectUrl.origin, 'https://openrouter.ai')
  assert.equal(redirectUrl.searchParams.get('code_challenge_method'), 'S256')
  assert.equal(redirectUrl.searchParams.get('code_challenge'), createHash('sha256').update(saved.verifier).digest('base64url'))
  browser.location.href = 'https://lumin.example/?code=single-use-code'
  globalThis.fetch = async (url, options) => { assert.equal(url, 'https://openrouter.ai/api/v1/auth/keys'); exchange = JSON.parse(options.body); return Response.json({ key: 'test-openrouter-key' }) }
  const connected = await Promise.all([settings.finishOpenRouterConnection(), settings.finishOpenRouterConnection()])
  assert.deepEqual(connected, [true, true])
  assert.equal(exchange.code_verifier, saved.verifier)
  assert.equal(sessionStorage.getItem('lumin-openrouter-pkce'), null)
  assert.equal(new URL(browser.location.href).searchParams.has('code'), false)
  assert.equal((await storage.db.settings.get('llm')).value.key, '')
  assert.equal(settings.getLlmSettings().key, 'test-openrouter-key')
})

test('an old in-flight snapshot cannot restore data after the user clears the workspace', async () => {
  for (const clearingInThisTab of [true, false]) {
  const workspace = await storage.createWorkspace('Clear race')
  await storage.selectWorkspace(workspace.id)
  let completeExport
  globalThis.fetch = async () => new Promise((resolve) => { completeExport = resolve })
  const save = workspaces.saveWorkspaceBundle(workspace.id)
  while (!completeExport) await new Promise((resolve) => setImmediate(resolve))
  if (clearingInThisTab) await storage.clearWorkspaceData(workspace.id)
  else await storage.db.workspaces.update(workspace.id, { bundle: undefined, version: 0, generation: 1 }) // Another tab cleared it.
  completeExport(new Response(emptyBundle(), { headers: { 'X-Workspace-Version': '9' } }))
  await save
  const record = await storage.db.workspaces.get(workspace.id)
  assert.equal(record.bundle, undefined)
  assert.equal(record.version, 0)
  }
})

test('browser backup version comes from the exported snapshot rather than a later status read', async () => {
  const workspace = await storage.createWorkspace('Versioned snapshot')
  await storage.selectWorkspace(workspace.id)
  let statusCalls = 0
  globalThis.fetch = async (url) => {
    if (url.endsWith('/api/workspace/export')) return new Response(emptyBundle(), { headers: { 'X-Workspace-Version': '4' } })
    statusCalls++
    return Response.json({ version: 8 })
  }
  await workspaces.saveWorkspaceBundle(workspace.id)
  assert.equal((await storage.db.workspaces.get(workspace.id)).version, 4)
  assert.equal(statusCalls, 0)
})

test('browser storage protection is requested only through the explicit user action', async () => {
  let requests = 0
  navigator.storage.persist = async () => { requests++; return true }
  assert.equal(requests, 0)
  await storage.requestPersistentStorage()
  assert.equal(requests, 1)
  assert.equal(storage.hasPersistentStorage(), true)
})


test('legacy backup quiz metadata is ignored while materials import without a storage migration', async () => {
  const archive = zipSync({ 'manifest.json': strToU8('{}'), 'graph.json': strToU8('{"nodes":[],"edges":[]}'),
    'quizzes.json': strToU8('[{"legacy":true}]'), 'browser.json': strToU8(JSON.stringify({ name: 'Legacy materials', chats: [], quiz: { old_shape: 'Ignored legacy state' } })) })
  let uploaded
  globalThis.fetch = async (_url, options) => { uploaded = unzipSync(new Uint8Array(await options.body.get('file').arrayBuffer())); return Response.json({ version: 1 }) }
  await workspaces.importWorkspace(new File([archive], 'legacy.lumin'))
  assert.equal(uploaded['quizzes.json'], undefined)
  assert.ok((await storage.db.workspaces.get(storage.activeWorkspaceId())).bundle)
})

test('a saved citation restores its exact longer passage and source location from its own browser bundle', async () => {
  const filename = 'Biology course.pdf'
  await storage.saveOriginals([new File(['PDF original'], filename, { type: 'application/pdf' })])
  const source = await storage.sourceIdentity(filename)
  const full = 'Detailed supporting passage. '.repeat(50) + 'The complete conclusion remains visible.'
  storage.cachePassages(JSON.stringify({ id: 'saved-citation', source_id: source, text: full,
    metadata: { original_filename: filename, page_start: 2, page_end: 3, section_title: 'Cell respiration' } }))
  const legacy = { id: 'saved-citation', source, modality: 'pdf', text: full.slice(0, 500), page: null }
  const resolved = sources.evidenceWithContext(legacy)
  assert.equal(resolved.text, full)
  assert.equal(sources.sourceDisplayName(resolved), filename)
  assert.equal(sources.sourceLocation(resolved), 'pp. 2–3 · Cell respiration')
  assert.match(sources.chunkSourceUrl(resolved), /^blob:.*#page=2$/)
  assert.equal(sources.evidenceWithContext({ ...legacy, source: 'other-source' }).text, legacy.text)
  assert.equal(sources.evidenceWithContext({ ...legacy, text: 'A replacement passage' }).text, 'A replacement passage')
  assert.equal(sources.evidenceWithContext({ ...legacy, source_filename: filename }).text, legacy.text, 'Modern API text is authoritative')
})

test('backup validation accepts real nullable citation metadata without losing hybrid provenance', async () => {
  const evidence = ['dense', 'graph', 'both'].map((retrieval_source, index) => ({ id: `nullable-${index}`, source: 'course', text: `Passage ${index}`, modality: 'office', page: null, page_end: null,
    timestamp: null, source_filename: 'Lecture.pptx', title: 'Kinetics', section_title: 'Enzymes', slide_number: 3, retrieval_source, relevancy_score: null }))
  const browserData = { name: 'Cited answers', chats: [{ id: 'answer-with-evidence', role: 'assistant', content: 'Answer [1]', timestamp: new Date(), routing_mode: 'hybrid', evidence, hops: ['enzyme'] }] }
  const archive = zipSync({ 'manifest.json': strToU8('{}'), 'graph.json': strToU8('{"nodes":[],"edges":[]}'), 'browser.json': strToU8(JSON.stringify(browserData)) })
  globalThis.fetch = async () => Response.json({ version: 1 })
  await workspaces.importWorkspace(new File([archive], 'cited.lumin'))
  const saved = await storage.db.chats.get(storage.activeWorkspaceId())
  assert.deepEqual(saved.messages[0].evidence.map((chunk) => chunk.retrieval_source), ['dense', 'graph', 'both'])
  assert.equal(saved.messages[0].evidence[0].slide_number, 3)
  assert.equal(saved.messages[0].evidence[0].source_filename, 'Lecture.pptx')
})

test('self-host restart refreshes a higher browser revision without HMR importing its stale bundle', async () => {
  const workspace = await storage.createWorkspace('Persistent self-host restart')
  await storage.selectWorkspace(workspace.id)
  await storage.db.workspaces.update(workspace.id, { version: 12, bundle: snapshotBundle('Older browser materials', 12) })
  const freshServer = await createTestServer()
  let freshStorage
  try {
    // A new storage module reproduces HMR: IndexedDB and the React ref's id
    // survive, but the module's active id starts empty until initialization.
    freshStorage = await freshServer.ssrLoadModule('/src/storage/db.ts')
    const freshWorkspaces = await freshServer.ssrLoadModule('/src/utils/workspaces.ts')
    assert.equal(freshStorage.activeWorkspaceId(), '')
    let imports = 0, exports = 0
    globalThis.fetch = async (url) => {
      if (url.endsWith('/api/workspace/status')) return Response.json({ loaded: true, version: 0 })
      if (url.endsWith('/api/workspace/import')) { imports++; throw new Error('A reconnect must not overwrite persistent server materials.') }
      if (url.endsWith('/api/workspace/export')) { exports++; return new Response(snapshotBundle('Newly committed server materials', 0), { headers: { 'X-Workspace-Version': '0' } }) }
      throw new Error(`Unexpected request: ${url}`)
    }
    assert.equal(await freshWorkspaces.restoreSelectedWorkspace(workspace.id, false), workspace.id)
    const record = await freshStorage.db.workspaces.get(workspace.id)
    assert.equal(imports, 0)
    assert.equal(exports, 1)
    assert.equal(record.version, 0)
    const entries = unzipSync(new Uint8Array(await record.bundle.arrayBuffer()))
    assert.match(strFromU8(entries['chunks.jsonl']), /Newly committed server materials/)
  } finally {
    freshStorage?.db.close()
    await freshServer.close()
  }
})

test('hosted snapshots still reject a version older than the saved browser snapshot', async () => {
  const workspace = await storage.createWorkspace('Hosted stale snapshot')
  await storage.selectWorkspace(workspace.id)
  await storage.db.workspaces.update(workspace.id, { version: 9, bundle: snapshotBundle('Current browser materials', 9) })
  globalThis.fetch = async () => new Response(snapshotBundle('Stale server response', 4), { headers: { 'X-Workspace-Version': '4' } })
  await workspaces.saveWorkspaceBundle(workspace.id)
  const record = await storage.db.workspaces.get(workspace.id)
  assert.equal(record.version, 9)
  assert.match(strFromU8(unzipSync(new Uint8Array(await record.bundle.arrayBuffer()))['chunks.jsonl']), /Current browser materials/)
})

test('a failed self-host snapshot leaves the previous browser backup intact', async () => {
  const workspace = await storage.createWorkspace('Failed reconnect backup')
  await storage.selectWorkspace(workspace.id)
  await storage.db.workspaces.update(workspace.id, { version: 8, bundle: snapshotBundle('Recoverable browser materials', 8) })
  globalThis.fetch = async (url) => url.endsWith('/api/workspace/status')
    ? Response.json({ loaded: true, version: 0 })
    : Response.json({ detail: 'Server unavailable' }, { status: 503 })
  await assert.rejects(workspaces.restoreSelectedWorkspace(workspace.id, false), /503/)
  const record = await storage.db.workspaces.get(workspace.id)
  assert.equal(record.version, 8)
  assert.match(strFromU8(unzipSync(new Uint8Array(await record.bundle.arrayBuffer()))['chunks.jsonl']), /Recoverable browser materials/)
})

test('an explicit self-host workspace switch still imports the selected browser materials', async () => {
  const previous = storage.activeWorkspaceId()
  const selected = await storage.createWorkspace('Explicitly selected materials')
  await storage.db.workspaces.update(selected.id, { version: 3, bundle: snapshotBundle('Selected browser materials', 3) })
  await storage.selectWorkspace(selected.id)
  let imports = 0
  globalThis.fetch = async (url, options) => {
    if (url.endsWith('/api/workspace/status')) return Response.json({ loaded: true, version: 0 })
    if (url.endsWith('/api/workspace/import')) {
      imports++
      const entries = unzipSync(new Uint8Array(await options.body.get('file').arrayBuffer()))
      assert.match(strFromU8(entries['chunks.jsonl']), /Selected browser materials/)
      return Response.json({ loaded: true, version: 1 })
    }
    if (url.endsWith('/api/workspace/export')) return new Response(snapshotBundle('Selected browser materials', 1), { headers: { 'X-Workspace-Version': '1' } })
    throw new Error(`Unexpected request: ${url}`)
  }
  assert.equal(await workspaces.restoreSelectedWorkspace(previous, false), selected.id)
  assert.equal(imports, 1)
  assert.equal((await storage.db.workspaces.get(selected.id)).version, 1)
})
