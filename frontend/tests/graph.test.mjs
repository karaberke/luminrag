import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createTestServer } from './helpers/vite.mjs'
import forceAtlas2 from 'graphology-layout-forceatlas2'
let server
let graphUtils
let graphologyAdapter
let gestures
let viewport
let connections

before(async () => {
  server = await createTestServer()
  graphUtils = await server.ssrLoadModule('/src/utils/graph.ts')
  graphologyAdapter = await server.ssrLoadModule('/src/components/graph/toGraphology.ts')
  gestures = await server.ssrLoadModule('/src/utils/graphGestures.ts')
  viewport = await server.ssrLoadModule('/src/utils/graphViewport.ts')
  connections = await server.ssrLoadModule('/src/utils/connections.ts')
})
after(async () => { await server?.close() })

const node = (id, node_type = 'content') => ({ id, name: id, node_type, source_ids: [] })
const edge = (source, target, relation = 'HAS_CONTENT') => ({ id: `${source}-${target}`, source, target, relation })

function viewportRenderer(graph, { width = 640, height = 480, hidden = [] } = {}) {
  const container = { offsetWidth: width, offsetHeight: height }
  const bounds = [], events = []
  const cameraState = { x: 0.5, y: 0.5, ratio: 1, angle: 0 }
  let customBBox = null
  const camera = { get ratio() { return cameraState.ratio }, getBoundedRatio: (ratio) => ratio,
    reset: () => { Object.assign(cameraState, { x: 0.5, y: 0.5, ratio: 1, angle: 0 }); events.push('animate') },
    cancelAnimation: () => events.push('cancel'), setState: (state) => { Object.assign(cameraState, state); events.push(state) },
    animate: (state) => { Object.assign(cameraState, state); events.push('animate') } }
  const normalize = ({ x, y }) => {
    const span = Math.max(customBBox.x[1] - customBBox.x[0], customBBox.y[1] - customBBox.y[0])
    return { x: 0.5 + (x - (customBBox.x[0] + customBBox.x[1]) / 2) / span,
      y: 0.5 + (y - (customBBox.y[0] + customBBox.y[1]) / 2) / span }
  }
  const renderer = { getGraph: () => graph, getContainer: () => container, getNodeState: (id) => ({ isHidden: hidden.includes(id) }),
    setCustomBBox: (value) => { customBBox = value; bounds.push(value); events.push('bbox') }, getCustomBBox: () => customBBox, getCamera: () => camera,
    graphToViewport: (position) => { const point = normalize(position), factor = Math.min(container.offsetWidth, container.offsetHeight) / cameraState.ratio
      return { x: container.offsetWidth / 2 + (point.x - cameraState.x) * factor, y: container.offsetHeight / 2 - (point.y - cameraState.y) * factor } },
    viewportToFramedGraph: ({ x, y }) => { const factor = Math.min(container.offsetWidth, container.offsetHeight) / cameraState.ratio
      return { x: cameraState.x + (x - container.offsetWidth / 2) / factor, y: cameraState.y - (y - container.offsetHeight / 2) / factor } },
    scaleSize: (size) => size / cameraState.ratio,
    refresh: () => events.push('refresh'), resize: () => events.push('resize') }
  return { renderer, container, bounds, events, cameraState }
}

test('collapse preserves shared descendants of expanded topics and handles hierarchy cycles', () => {
  const graph = {
    nodes: [node('a', 'topic'), node('b', 'topic'), node('shared', 'subtopic'), node('shared-content'), node('exclusive')],
    edges: [edge('a', 'shared', 'HAS_SUBTOPIC'), edge('b', 'shared', 'HAS_SUBTOPIC'), edge('shared', 'shared-content'), edge('a', 'exclusive'), edge('exclusive', 'a')],
  }
  const index = graphUtils.indexGraph(graph)
  const one = graphUtils.collapseGraph(graph, index, new Set(['a']))
  assert.deepEqual(one.filteredNodes.map((n) => n.id), ['a', 'b', 'shared', 'shared-content'])
  assert.equal(one.collapseCountMap.get('a'), 1)
  assert.ok(one.filteredEdges.every((e) => e.source !== 'exclusive' && e.target !== 'exclusive'))
  const all = graphUtils.collapseGraph(graph, index, new Set(['a', 'b']))
  assert.deepEqual(all.filteredNodes.map((n) => n.id), ['a', 'b'])
  assert.equal(all.collapseCountMap.get('a'), 3)
  assert.equal(all.collapseCountMap.get('b'), 2)
  assert.deepEqual(all.filteredEdges, [])
  const expanded = graphUtils.collapseGraph(graph, index, new Set())
  assert.equal(expanded.filteredNodes, graph.nodes)
})

test('wide collapse avoids repeated node scans and counts every descendant once', () => {
  const children = Array.from({ length: 10000 }, (_, index) => node(`n${index}`))
  const graph = { nodes: [node('topic', 'topic'), ...children], edges: children.map((n) => edge('topic', n.id)) }
  const result = graphUtils.collapseGraph(graph, graphUtils.indexGraph(graph), new Set(['topic']))
  assert.equal(result.collapseCountMap.get('topic'), 10000)
  assert.deepEqual(result.filteredNodes.map((n) => n.id), ['topic'])
  assert.deepEqual(result.filteredEdges, [])
})

test('dense neighbor traversal expands each node once and supports multiple roots', () => {
  const adjacency = new Map()
  const ids = Array.from({ length: 100 }, (_, index) => `n${index}`)
  for (const id of ids) adjacency.set(id, ids.filter((other) => other !== id))
  adjacency.set('isolated', [])
  let expansions = 0
  const originalGet = adjacency.get.bind(adjacency)
  adjacency.get = (id) => { expansions++; return originalGet(id) }
  const reachable = graphUtils.reachableNodeIds(adjacency, ['n0', 'isolated'])
  assert.equal(reachable.size, 101)
  assert.equal(expansions, 101)
})

test('edge-only changes invalidate graph layout, while a text edit preserves the viewport', () => {
  const graph = { nodes: [node('a'), node('b')], edges: [] }
  const initial = graphUtils.graphTopologyKey(graph)
  assert.notEqual(graphUtils.graphTopologyKey({ ...graph, edges: [edge('a', 'b')] }), initial)
  assert.equal(graphUtils.graphTopologyKey({ ...graph, nodes: [node('a'), { ...node('b'), name: 'Renamed' }] }), initial)
})

test('duplicate suggestions track merge, deletion and Undo without targeting absent nodes', () => {
  const original = { nodes: [node('a'), node('b'), node('c')], edges: [] }
  const pairs = [{ keep: 'a', drop: 'b', similarity: 0.99 }, { keep: 'b', drop: 'c', similarity: 0.96 },
    { keep: 'a', drop: 'c', similarity: 0.95 }, { keep: 'missing', drop: 'c', similarity: 0.99 }, { keep: 'a', drop: 'a', similarity: 1 }]
  const initial = graphUtils.duplicateSuggestionKey(original.nodes)
  const merged = { nodes: [node('a'), node('c')], edges: [] }
  assert.deepEqual(graphUtils.validDuplicateSuggestions(pairs, graphUtils.indexGraph(merged).nodes), [pairs[2]])
  assert.notEqual(graphUtils.duplicateSuggestionKey(merged.nodes), initial, 'A merge refreshes local duplicate calculations')
  assert.deepEqual(graphUtils.validDuplicateSuggestions(pairs, graphUtils.indexGraph(original).nodes), pairs.slice(0, 3), 'Undo restores eligible candidates')
  assert.deepEqual(graphUtils.validDuplicateSuggestions(pairs, graphUtils.indexGraph({ nodes: [node('a')], edges: [] }).nodes), [])
  assert.equal(graphUtils.duplicateSuggestionKey(original.nodes.map((value) => ({ ...value, x: 20, y: 30, highlighted: true }))), initial, 'Layout/highlight changes must not trigger inference')
  assert.notEqual(graphUtils.duplicateSuggestionKey([node('a'), { ...node('b'), summary: 'Changed meaning' }, node('c')]), initial)
})

test('Graphology renders one plain association while retaining directed hierarchy and self loops', async () => {
  const data = await connections.normalizeGraphConnections({ nodes: [node('a'), node('b', 'note')], edges: [
    { ...edge('a', 'b'), id: 'one', relation: 'RELATED_TO', label: 'explains' },
    { ...edge('b', 'a'), id: 'two', relation: 'RELATED_TO', label: 'contrasts' },
    { ...edge('a', 'a'), id: 'loop' },
  ] })
  const graph = graphologyAdapter.toGraphology(data)
  assert.equal(graph.order, 2)
  assert.equal(graph.size, 2)
  assert.equal(graph.directedEdges('a', 'b').length, 1)
  assert.equal(graph.getEdgeAttribute('assoc:0473ef2dc0d324ab659d3580', 'label'), null)
  assert.equal(graph.getEdgeAttribute('loop', 'relation'), 'HAS_CONTENT')
  assert.equal(graph.getNodeAttribute('b', 'node_type'), 'note')
})

test('incremental text and topology updates retain dragged positions', () => {
  const data = { nodes: [node('a'), node('b')], edges: [edge('a', 'b')] }
  const graph = graphologyAdapter.toGraphology(data)
  graph.setNodeAttribute('a', 'x', 123)
  graph.setNodeAttribute('a', 'y', -72)
  graphologyAdapter.syncGraphology(graph, { nodes: [{ ...node('a'), name: 'Edited' }, node('c')], edges: [edge('a', 'c')] })
  assert.equal(graph.getNodeAttribute('a', 'x'), 123)
  assert.equal(graph.getNodeAttribute('a', 'y'), -72)
  assert.equal(graph.getNodeAttribute('a', 'label'), 'Edited')
  assert.equal(graph.hasNode('b'), false)
  assert.equal(graph.size, 1)
})

test('lasso hit testing handles concave shapes and outside points', () => {
  const polygon = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 5, y: 5 }, { x: 0, y: 10 }]
  assert.equal(gestures.pointInPolygon({ x: 5, y: 2 }, polygon), true)
  assert.equal(gestures.pointInPolygon({ x: 5, y: 8 }, polygon), false)
  assert.equal(gestures.pointInPolygon({ x: 15, y: 5 }, polygon), false)
})

test('3,000-node graph builds once and ForceAtlas2 produces finite coordinates', () => {
  const nodes = Array.from({ length: 3000 }, (_, index) => node(`n${index}`, index % 30 ? 'content' : 'topic'))
  const data = { nodes, edges: nodes.slice(1).map((value, index) => edge(`n${index}`, value.id)) }
  const start = performance.now()
  const graph = graphologyAdapter.toGraphology(data)
  forceAtlas2.assign(graph, { iterations: 2, settings: { barnesHutOptimize: true } })
  assert.equal(graph.order, 3000)
  assert.ok(graph.everyNode((id, attrs) => Number.isFinite(attrs.x) && Number.isFinite(attrs.y)))
  assert.ok(performance.now() - start < 10000)
})

test('null and non-finite server positions cannot overwrite valid renderer coordinates', () => {
  const data = { nodes: [{ ...node('a', 'note'), x: null, y: null }, { ...node('b'), x: NaN, y: Infinity }], edges: [] }
  const graph = graphologyAdapter.toGraphology(data)
  assert.ok(graph.everyNode((id, attrs) => typeof attrs.x === 'number' && Number.isFinite(attrs.x) && typeof attrs.y === 'number' && Number.isFinite(attrs.y)))
  graph.setNodeAttribute('a', 'x', 123)
  graph.setNodeAttribute('a', 'y', -72)
  graphologyAdapter.syncGraphology(graph, data)
  assert.equal(graph.getNodeAttribute('a', 'x'), 123)
  assert.equal(graph.getNodeAttribute('a', 'y'), -72)
})

test('Fit includes newly added visible nodes and ignores hidden evidence outside the view', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [{ ...node('a'), x: 0, y: 0 }], edges: [] })
  const { renderer, bounds, events } = viewportRenderer(graph, { hidden: ['hidden'] })
  viewport.fitGraphBounds(renderer)
  graph.addNode('b', { x: 200, y: -100 })
  graph.addNode('hidden', { x: 10000, y: 10000 })
  viewport.fitGraphBounds(renderer)
  assert.ok(bounds[1].x[1] > 200)
  assert.ok(bounds[1].y[0] < -100)
  assert.ok(bounds[1].x[1] < 1000)
  assert.deepEqual(events.slice(-3), ['bbox', 'refresh', 'animate'])
})

test('hidden containers never freeze the fit to Sigma\'s 1x1 fallback stage', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [{ ...node('a'), x: 123, y: -72 }], edges: [] })
  const { renderer, container, bounds, events } = viewportRenderer(graph, { width: 0 })
  assert.equal(viewport.fitGraphBounds(renderer), false)
  assert.deepEqual(events, [])
  container.offsetWidth = 640
  container.offsetHeight = 0
  assert.equal(viewport.fitGraphBounds(renderer), false)
  assert.deepEqual(bounds, [])
  assert.equal(graph.getNodeAttribute('a', 'x'), 123)
  assert.equal(graph.getNodeAttribute('a', 'y'), -72)
})

test('reopening and resizing fit once at positive dimensions without resetting ordinary camera changes', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [{ ...node('a'), x: 0, y: 0 }, { ...node('b'), x: 200, y: 100 }], edges: [] })
  const { renderer, container, bounds, events } = viewportRenderer(graph, { width: 0 })
  let size = viewport.resizeGraphViewport(renderer, null, true)
  assert.equal(size, null)
  assert.deepEqual(events, [])
  container.offsetWidth = 640
  size = viewport.resizeGraphViewport(renderer, size, true)
  assert.deepEqual(size, { width: 640, height: 480 })
  assert.deepEqual(events, ['resize', 'bbox', 'cancel', { x: 0.5, y: 0.5, ratio: 1, angle: 0 }, 'refresh'])
  const eventCount = events.length
  assert.equal(viewport.resizeGraphViewport(renderer, size, true), size)
  assert.equal(events.length, eventCount)
  graph.setNodeAttribute('a', 'name', 'Renamed')
  assert.equal(viewport.resizeGraphViewport(renderer, size, true), size)
  assert.equal(events.length, eventCount)
  size = viewport.resizeGraphViewport(renderer, size, false)
  assert.equal(size, null)
  assert.equal(events.length, eventCount)
  size = viewport.resizeGraphViewport(renderer, size, true)
  assert.equal(bounds.length, 1)
  container.offsetWidth = 960
  size = viewport.resizeGraphViewport(renderer, size, true)
  assert.deepEqual(size, { width: 960, height: 480 })
  assert.equal(bounds.length, 1)
  assert.equal(graph.getNodeAttribute('b', 'x'), 200)
  assert.equal(graph.getNodeAttribute('b', 'y'), 100)
})

test('synchronous Fit settles camera and rebuilds normalization before viewport layout runs', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [{ ...node('a'), x: 0, y: 0 }], edges: [] })
  const { renderer, events } = viewportRenderer(graph)
  assert.equal(viewport.fitGraphBounds(renderer, { animate: false }), true)
  assert.deepEqual(events, ['bbox', 'cancel', { x: 0.5, y: 0.5, ratio: 1, angle: 0 }, 'refresh'])
})

test('camera framing brings displaced clusters onscreen without rebuilding the collision coordinate scale', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [{ ...node('a', 'topic'), x: 0, y: 0 }, { ...node('b'), x: 200, y: 100 }], edges: [] })
  const { renderer, bounds, cameraState } = viewportRenderer(graph)
  viewport.fitGraphBounds(renderer, { animate: false })
  const originalBBox = renderer.getCustomBBox()
  // A collision pass moves the clusters beyond the original fitted extent.
  graph.mergeNodeAttributes('a', { x: -500, y: 2000 })
  graph.mergeNodeAttributes('b', { x: 900, y: -600 })
  const projected = () => graph.mapNodes((id, attrs) => ({ ...renderer.graphToViewport(attrs), radius: renderer.scaleSize(attrs.size) }))
  const before = projected()
  const relativeSeparation = Math.hypot(before[0].x - before[1].x, before[0].y - before[1].y) / (before[0].radius + before[1].radius)
  assert.equal(viewport.fitGraphBounds(renderer, { animate: false, preserveScale: true }), true)
  assert.equal(renderer.getCustomBBox(), originalBBox)
  assert.equal(bounds.length, 1)
  assert.ok(cameraState.ratio > 1)
  const after = projected()
  for (const point of after) {
    assert.ok(point.x - point.radius >= 23.99 && point.x + point.radius <= 616.01)
    assert.ok(point.y - point.radius >= 23.99 && point.y + point.radius <= 456.01)
  }
  const framedSeparation = Math.hypot(after[0].x - after[1].x, after[0].y - after[1].y) / (after[0].radius + after[1].radius)
  assert.ok(Math.abs(framedSeparation - relativeSeparation) < 1e-8)
  assert.equal(graph.getNodeAttribute('a', 'x'), -500)
  assert.equal(graph.getNodeAttribute('b', 'y'), -600)
  const ratio = cameraState.ratio
  viewport.fitGraphBounds(renderer, { animate: false, preserveScale: true })
  assert.ok(Math.abs(cameraState.ratio - ratio) < 1e-8, 'Repeated framing must not cause a zoom loop.')
})

test('preserved-scale Fit recovers the overview after manual zoom-out without enlarging a lone node', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [{ ...node('a'), x: 0, y: 0 }], edges: [] })
  const { renderer, bounds, cameraState } = viewportRenderer(graph)
  viewport.fitGraphBounds(renderer, { animate: false })
  cameraState.ratio = 5
  cameraState.x = 20
  cameraState.y = -30
  assert.equal(viewport.fitGraphBounds(renderer, { animate: false, preserveScale: true }), true)
  assert.equal(bounds.length, 1)
  assert.equal(cameraState.ratio, 1)
  const point = renderer.graphToViewport(graph.getNodeAttributes('a'))
  assert.ok(Math.abs(point.x - 320) < 1e-8 && Math.abs(point.y - 240) < 1e-8)
})

test('retrieval highlighting follows only returned existing IDs and never includes an unrelated topic', () => {
  const data = { nodes: [node('biology', 'topic'), node('enzyme'), node('history', 'topic'), node('battle')],
    edges: [edge('biology', 'enzyme'), edge('history', 'battle')] }
  const traced = graphUtils.highlightGraphTrace(data, ['enzyme', 'biology', 'unknown', 'enzyme'], 'answer-1')
  assert.deepEqual(traced.nodes.filter((value) => value.highlighted).map((value) => value.id), ['biology', 'enzyme'])
  assert.equal(traced.nodes.find((value) => value.id === 'enzyme').hopIndex, 0)
  assert.deepEqual(traced.edges.filter((value) => value.highlighted).map((value) => value.id), ['biology-enzyme'])
  assert.ok(data.nodes.every((value) => !value.highlighted))
  assert.ok(graphUtils.highlightGraphTrace(data, ['unknown']).nodes.every((value) => !value.highlighted))
})

test('every new query selects Subgraph even when its node IDs match the previous query', () => {
  const data = { nodes: [node('a'), node('b')], edges: [edge('a', 'b')] }
  const first = graphUtils.highlightGraphTrace(data, ['a'], 'answer-1')
  const chosenFull = { value: 'full', trace: graphUtils.graphTraceKey(first) }
  assert.equal(graphUtils.activeGraphTab(first, chosenFull), 'full')
  const second = graphUtils.highlightGraphTrace(data, ['a'], 'answer-2')
  assert.equal(graphUtils.activeGraphTab(second, chosenFull), 'subgraph')
})

test('Subgraph has no all-node fallback and query nodes remain visible through collapsed topics', () => {
  const data = { nodes: [node('a', 'topic'), node('b'), node('unrelated')], edges: [edge('a', 'b')] }
  const traced = graphUtils.highlightGraphTrace(data, ['b'], 'answer')
  assert.deepEqual([...graphUtils.hiddenGraphNodes(traced, new Set(['a']), 'subgraph')], ['a', 'unrelated'])
  assert.deepEqual([...graphUtils.hiddenGraphNodes(traced, new Set(['a']), 'full')], ['unrelated'])
  const dense = graphUtils.highlightGraphTrace(traced, [], 'dense-answer')
  assert.deepEqual([...graphUtils.hiddenGraphNodes(dense, new Set(data.nodes.map((value) => value.id)), 'subgraph')], ['a', 'b', 'unrelated'])
})

test('neighbor focus stays within actual neighbors and an unknown selection stays empty', () => {
  const graph = graphologyAdapter.toGraphology({ nodes: [node('a'), node('b'), node('unrelated')], edges: [edge('a', 'b')] })
  assert.deepEqual([...graphUtils.selectedNeighborhood(graph, ['a'])].sort(), ['a', 'b'])
  assert.deepEqual([...graphUtils.selectedNeighborhood(graph, ['missing'])], [])
})
