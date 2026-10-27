import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createHash } from 'node:crypto'
import { createTestServer } from './helpers/vite.mjs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

let server, connections, styles, NodeDetailPanel
before(async () => {
  server = await createTestServer({
    // The style callbacks need no GPU. Keep Sigma's WebGL defaults out of SSR.
    plugins: [{ name: 'style-defaults-without-webgl', enforce: 'pre', transform(code, id) {
      if (id.replaceAll('\\', '/').endsWith('/graphStyles.ts')) return code.replace(/import\s*\{\s*DEFAULT_STYLES\s*\}\s*from\s*['"]sigma['"];?/, 'const DEFAULT_STYLES = { nodes: {}, edges: {} }')
    } }],
  })
  connections = await server.ssrLoadModule('/src/utils/connections.ts')
  styles = await server.ssrLoadModule('/src/components/graph/graphStyles.ts')
  NodeDetailPanel = (await server.ssrLoadModule('/src/components/NodeDetailPanel.tsx')).default
})
after(async () => { await server?.close() })

const node = (id, node_type = 'content') => ({ id, name: id, node_type, source_ids: ['source.pdf'] })
const edge = (id, source, target, extra = {}) => ({ id, source, target, relation: 'RELATED_TO', ...extra })
const support = (kind, extra = {}) => ({ kind, owner: null, source_chunk_ids: [], confidence: 0.7, verified: false, ...extra })

test('association identity matches backend hashing including Python Unicode ordering', async () => {
  assert.equal(await connections.associationId('b', 'a'), 'assoc:0473ef2dc0d324ab659d3580')
  const pair = ['\ue000', '😀'] // UTF-16 native sort reverses this Python code-point order.
  const expected = `assoc:${createHash('sha256').update(JSON.stringify(pair), 'utf8').digest('hex').slice(0, 24)}`
  assert.equal(await connections.associationId(pair[1], pair[0]), expected)
})

test('legacy reversed labels normalize idempotently without merging concepts or changing sources and positions', async () => {
  const nodes = [{ ...node('a'), x: 25, y: -30, raw_excerpt: 'Exact cited passage.', source_ids: ['paper:p3'] }, node('b'), node('topic', 'topic')]
  const hierarchy = edge('hierarchy', 'topic', 'a', { relation: 'HAS_CONTENT', supports: [] })
  const graph = { nodes, edges: [
    edge('legacy-one', 'a', 'b', { label: 'explains', confidence: 0.6, source_chunk_ids: ['p1'] }),
    edge('legacy-two', 'b', 'a', { label: 'supports', confidence: 0.8, source_chunk_ids: ['p2'] }), hierarchy,
  ] }
  const normalized = await connections.normalizeGraphConnections(graph)
  assert.equal(normalized.association_version, 1)
  assert.equal(normalized.nodes, nodes)
  assert.equal(normalized.nodes.length, 3)
  assert.equal(normalized.edges.length, 2)
  const association = normalized.edges.find((value) => value.relation === 'RELATED_TO')
  assert.equal(association.id, 'assoc:0473ef2dc0d324ab659d3580')
  assert.equal(association.label, null)
  assert.deepEqual(association.source_chunk_ids, ['p1', 'p2'])
  assert.equal(association.confidence, 0.8)
  assert.ok(association.supports.every((value) => value.kind === 'legacy' && !value.verified))
  assert.deepEqual(normalized.edges.find((value) => value.id === 'hierarchy'), hierarchy)
  assert.deepEqual(await connections.normalizeGraphConnections(normalized), normalized)
  assert.equal(graph.edges[0].label, 'explains', 'Normalize a copy, preserving the caller snapshot.')
})

test('canonical server identities and independent automatic, manual, and note backing survive deduplication', async () => {
  const id = 'assoc:0473ef2dc0d324ab659d3580'
  const normalized = await connections.normalizeGraphConnections({ nodes: [node('a', 'note'), node('b')], edges: [
    edge(id, 'a', 'b', { label: null, supports: [
      support('automatic', { source_chunk_ids: ['p1'], verified: true }), support('manual'),
      support('wikilink', { owner: 'a', source_chunk_ids: ['note:a'] }),
    ] }),
    edge('old-reversed', 'b', 'a', { label: 'refers to', supports: [support('manual', { confidence: 0.9 })] }),
  ], association_version: 1 })
  assert.equal(normalized.edges.length, 1)
  const association = normalized.edges[0]
  assert.equal(association.id, id)
  assert.equal(association.supports.length, 3)
  assert.equal(association.supports.find((value) => value.kind === 'manual').confidence, 0.9)
  assert.equal(association.supports.find((value) => value.kind === 'automatic').verified, true)
  assert.deepEqual(association.source_chunk_ids, ['note:a', 'p1'])
})

test('legacy reversed wikilinks retain the owning note and never gain verification', async () => {
  const normalized = await connections.normalizeGraphConnections({ nodes: [node('a'), node('note', 'note')], edges: [
    edge('wiki', 'a', 'note', { wikilink: true, label: 'references' }),
  ] })
  assert.deepEqual(normalized.edges[0].supports, [support('wikilink', { owner: 'note', source_chunk_ids: ['note:note'], confidence: 0.5 })])
})

test('malformed backing claims and oversized references are rejected within existing graph bounds', async () => {
  const graph = (backing) => ({ nodes: [node('a'), node('b')], edges: [edge('old', 'a', 'b', { supports: backing })] })
  await assert.rejects(connections.normalizeGraphConnections(graph([support('manual', { verified: true })])), /Only automatic/)
  await assert.rejects(connections.normalizeGraphConnections(graph([support('automatic', { verified: true })])), /supporting passages/)
  await assert.rejects(connections.normalizeGraphConnections(graph([support('wikilink', { owner: 'unrelated' })])), /owning note/)
  await assert.rejects(connections.normalizeGraphConnections(graph([support('legacy', { source_chunk_ids: ['x'.repeat(1025)] })])))
  await assert.rejects(connections.normalizeGraphConnections(graph([])))
})

test('semantic edges have no arrow or hover label while hierarchy retains direction', () => {
  for (const dark of [false, true]) {
    const edges = styles.graphOptions(dark).styles.edges
    assert.equal(edges[1].head({ relation: 'RELATED_TO' }), 'none')
    assert.equal(edges[1].head({ relation: 'HAS_CONTENT' }), 'arrow')
    assert.equal(edges[3].then.labelVisibility({ relation: 'RELATED_TO' }), 'hidden')
    assert.equal(edges[3].then.labelVisibility({ relation: 'HAS_SUBTOPIC' }), 'visible')
  }
})

test('node details show plain Connections and human hierarchy hints without exposing old verbs', () => {
  const nodes = [node('concept'), node('related'), node('topic', 'topic'), node('detail')]
  const markup = renderToStaticMarkup(createElement(NodeDetailPanel, {
    node: nodes[0], allNodes: nodes, evidence: [], editMode: true, onClose: () => {}, edges: [
      edge('old', 'related', 'concept', { label: 'causes' }),
      edge('parent', 'topic', 'concept', { relation: 'HAS_SUBTOPIC' }),
      edge('child', 'concept', 'detail', { relation: 'HAS_CONTENT' }),
    ],
  }))
  assert.match(markup, /Connections/)
  assert.match(markup, /Parent:/)
  assert.match(markup, /Content:/)
  assert.doesNotMatch(markup, /Relations|causes|RELATED_TO|HAS_SUBTOPIC|HAS_CONTENT/)
  assert.equal((markup.match(/Delete this connection/g) ?? []).length, 3)
})
