import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { UndirectedGraph } from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import { createTestServer } from './helpers/vite.mjs'

let server, layout
before(async () => {
  server = await createTestServer()
  layout = await server.ssrLoadModule('/src/utils/graphLayout.ts')
})
after(async () => { await server?.close() })

function renderer(graph, scale = 1, hidden = new Set(), width = 320, height = 600) {
  return { getGraph: () => graph, getContainer: () => ({ offsetWidth: width, offsetHeight: height }),
    getNodeState: id => ({ isHidden: hidden.has(id) }), scaleSize: size => size,
    graphToViewport: ({ x, y }) => ({ x: x * scale + width / 2, y: y * scale + height / 2 }),
    viewportToGraph: ({ x, y }) => ({ x: (x - width / 2) / scale, y: (y - height / 2) / scale }) }
}

function closestDistance(graph, scale = 1) {
  let closest = Infinity
  const nodes = graph.mapNodes((id, attrs) => attrs)
  for (let a = 0; a < nodes.length; a++) for (let b = a + 1; b < nodes.length; b++) {
    closest = Math.min(closest, Math.hypot(nodes[a].x - nodes[b].x, nodes[a].y - nodes[b].y) * scale)
  }
  return closest
}

function cluster() {
  const graph = new UndirectedGraph()
  for (let index = 0; index < 16; index++) graph.addNode(String(index), { x: index % 4, y: Math.floor(index / 4), size: 5 })
  return graph
}

test('screen-space collision removal separates a compressed cluster without changing node size', () => {
  const graph = cluster()
  layout.separateGraphNodes(renderer(graph, 0.2), 1)
  assert.ok(closestDistance(graph, 0.2) >= 19)
  assert.ok(graph.everyNode((_id, attrs) => attrs.size === 5 && Number.isFinite(attrs.x) && Number.isFinite(attrs.y)))
})

test('higher spacing produces larger pixel clearance instead of being cancelled by a refit', () => {
  const compact = cluster(), spacious = cluster()
  layout.separateGraphNodes(renderer(compact, 2), 0.3)
  layout.separateGraphNodes(renderer(spacious, 2), 3)
  assert.ok(closestDistance(spacious, 2) > closestDistance(compact, 2) + 15)
})

test('hidden and offscreen panels preserve their saved coordinates', () => {
  const graph = cluster()
  graph.addNode('hidden', { x: 0, y: 0, size: 10 })
  layout.separateGraphNodes(renderer(graph, 1, new Set(['hidden'])), 1)
  assert.deepEqual(graph.getNodeAttributes('hidden'), { x: 0, y: 0, size: 10 })
  const before = graph.export()
  layout.separateGraphNodes(renderer(graph, 1, new Set(), 0, 0), 3)
  assert.deepEqual(graph.export(), before)
})

test('bounded collision work and size-aware ForceAtlas remain finite for 3,000 nodes', () => {
  const graph = new UndirectedGraph()
  for (let index = 0; index < 3000; index++) {
    const angle = index * Math.PI * (3 - Math.sqrt(5)), radius = 12 * Math.sqrt(index + 1)
    graph.addNode(String(index), { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, size: 5 })
    if (index) graph.addEdge(String(index - 1), String(index))
  }
  const started = performance.now()
  forceAtlas2.assign(graph, { iterations: 2, settings: layout.graphLayoutSettings(1) })
  layout.separateGraphNodes(renderer(graph), 1)
  assert.ok(graph.everyNode((_id, attrs) => Number.isFinite(attrs.x) && Number.isFinite(attrs.y)))
  assert.ok(performance.now() - started < 10000)
})
