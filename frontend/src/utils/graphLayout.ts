import { UndirectedGraph } from 'graphology'
import noverlap from 'graphology-layout-noverlap'
import type { ForceAtlas2Settings } from 'graphology-layout-forceatlas2'
import type Sigma from 'sigma'

const MAX_SMALL_GRAPH_ITERATIONS = 120
const MAX_LARGE_GRAPH_ITERATIONS = 40
const SMALL_GRAPH_NODES = 500

export function graphLayoutSettings(spacing: number): ForceAtlas2Settings {
  const separation = Math.max(0.3, Math.min(3, spacing))
  return { barnesHutOptimize: true, adjustSizes: true, scalingRatio: 12 * separation,
    gravity: 0.5 / separation, slowDown: 5 }
}

/** Resolve circle collisions in the same pixel units as Sigma's rendered sizes.
 *
 * The fitted camera must already be settled. Retain its normalization after
 * this pass; camera-only framing can then include displaced edge nodes.
 */
export function separateGraphNodes(renderer: Sigma, spacing: number): void {
  const graph = renderer.getGraph()
  const container = renderer.getContainer()
  if (container.offsetWidth <= 0 || container.offsetHeight <= 0) return
  const visible = new UndirectedGraph()
  graph.forEachNode((id, attrs) => {
    if (renderer.getNodeState(id).isHidden) return
    const position = renderer.graphToViewport({ x: attrs.x, y: attrs.y })
    const size = renderer.scaleSize(typeof attrs.size === 'number' ? attrs.size : 5)
    if (Number.isFinite(position.x) && Number.isFinite(position.y) && Number.isFinite(size)) {
      visible.addNode(id, { ...position, size: Math.max(1, size) })
    }
  })
  if (visible.order < 2) return
  const separation = Math.max(0.3, Math.min(3, spacing))
  const positions = noverlap(visible, {
    maxIterations: visible.order <= SMALL_GRAPH_NODES ? MAX_SMALL_GRAPH_ITERATIONS : MAX_LARGE_GRAPH_ITERATIONS,
    settings: { margin: 5 * separation, ratio: 1.1, gridSize: Math.max(20, Math.min(100, Math.ceil(Math.sqrt(visible.order)))) },
    outputReducer: (_id, point) => renderer.viewportToGraph(point),
  })
  for (const [id, position] of Object.entries(positions)) graph.mergeNodeAttributes(id, position)
}
