import { MultiDirectedGraph } from 'graphology'
import type { GraphData } from '../../types'

export function toGraphology(data: GraphData): MultiDirectedGraph {
  const graph = new MultiDirectedGraph({ allowSelfLoops: true })
  syncGraphology(graph, data)
  return graph
}

/** Retain the render model and dragged coordinates while replacing only changed topology. */
export function syncGraphology(graph: MultiDirectedGraph, data: GraphData): void {
  const nodeIds = new Set(data.nodes.map((node) => node.id))
  for (const id of graph.nodes()) if (!nodeIds.has(id)) graph.dropNode(id)
  for (let index = 0; index < data.nodes.length; index++) {
    const node = data.nodes[index]
    const angle = index * Math.PI * (3 - Math.sqrt(5))
    const radius = 12 * Math.sqrt(index + 1)
    const previous = graph.hasNode(node.id) ? graph.getNodeAttributes(node.id) : {}
    const serverX = typeof node.x === 'number' && Number.isFinite(node.x) ? node.x : undefined
    const serverY = typeof node.y === 'number' && Number.isFinite(node.y) ? node.y : undefined
    const previousX = typeof previous.x === 'number' && Number.isFinite(previous.x) ? previous.x : undefined
    const previousY = typeof previous.y === 'number' && Number.isFinite(previous.y) ? previous.y : undefined
    const x = serverX !== undefined && serverX !== previous.serverX ? serverX : previousX ?? serverX ?? Math.cos(angle) * radius
    const y = serverY !== undefined && serverY !== previous.serverY ? serverY : previousY ?? serverY ?? Math.sin(angle) * radius
    graph.mergeNode(node.id, { ...node, label: node.name, x, y, serverX, serverY,
      size: node.node_type === 'topic' ? 10 : node.node_type === 'subtopic' ? 7 : 5 })
  }
  const edgeIds = new Set(data.edges.map((edge) => edge.id))
  for (const id of graph.edges()) if (!edgeIds.has(id)) graph.dropEdge(id)
  for (const edge of data.edges) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue
    if (graph.hasEdge(edge.id) && (graph.source(edge.id) !== edge.source || graph.target(edge.id) !== edge.target)) graph.dropEdge(edge.id)
    const label = edge.relation === 'RELATED_TO' ? null : edge.relation === 'HAS_SUBTOPIC' ? 'Subtopic' : 'Content'
    graph.mergeDirectedEdgeWithKey(edge.id, edge.source, edge.target, { ...edge, label, size: 1 })
  }
}
