import type { EvidenceChunk, GraphData, GraphNode } from '../types'
import type { MultiDirectedGraph } from 'graphology'

export type GraphTab = 'full' | 'subgraph'
export interface GraphTabState { value: GraphTab; trace: string }

/** Only backend-returned, existing node IDs can become retrieval highlights. */
export function highlightGraphTrace(graph: GraphData, hops: readonly string[], traceId?: string): GraphData {
  const knownIds = new Set(graph.nodes.map((node) => node.id))
  const indices = new Map([...new Set(hops)].filter((id) => knownIds.has(id)).map((id, index) => [id, index]))
  return { ...graph, traceId,
    nodes: graph.nodes.map((node) => ({ ...node, highlighted: indices.has(node.id), hopIndex: indices.get(node.id) })),
    // The API returns a subgraph, rather than an ordered edge-by-edge route.
    edges: graph.edges.map((edge) => ({ ...edge, highlighted: indices.has(edge.source) && indices.has(edge.target) })),
  }
}

export function graphTraceKey(graph: GraphData): string {
  return graph.traceId ?? graph.nodes.filter((node) => node.highlighted).map((node) => node.id).join('|')
}

export function activeGraphTab(graph: GraphData, state: GraphTabState): GraphTab {
  return graph.nodes.some((node) => node.highlighted) && state.trace !== graphTraceKey(graph) ? 'subgraph' : state.value
}

export function hiddenGraphNodes(graph: GraphData, shown: Set<string>, tab: GraphTab): Set<string> {
  const retrieved = new Set(graph.nodes.filter((node) => node.highlighted).map((node) => node.id))
  return new Set(graph.nodes.filter((node) => tab === 'subgraph' ? !retrieved.has(node.id) : !retrieved.has(node.id) && !shown.has(node.id)).map((node) => node.id))
}

/** Missing selections stay empty; they never widen a focus to the whole graph. */
export function selectedNeighborhood(graph: Pick<MultiDirectedGraph, 'hasNode' | 'neighbors'>, selected: Iterable<string>): Set<string> {
  const neighbors = new Set<string>()
  for (const id of selected) if (graph.hasNode(id)) {
    neighbors.add(id)
    for (const neighbor of graph.neighbors(id)) neighbors.add(neighbor)
  }
  return neighbors
}

export interface GraphIndex {
  nodes: Map<string, GraphNode>
  children: Map<string, string[]>
  neighbors: Map<string, string[]>
}

export interface DuplicateCandidate { keep: string; drop: string; similarity: number }

/** Coordinates and retrieval highlights do not change duplicate suggestions. */
export function duplicateSuggestionKey(nodes: readonly GraphNode[]): string {
  return JSON.stringify(nodes.map((node) => [node.id, node.node_type, node.name, node.title,
    node.body, node.summary, node.key_terms, node.raw_excerpt]))
}

/** A stale response must never offer a merge against a removed node. */
export function validDuplicateSuggestions(pairs: readonly DuplicateCandidate[], nodes: ReadonlyMap<string, GraphNode>): DuplicateCandidate[] {
  return pairs.filter((pair) => pair.keep !== pair.drop && nodes.has(pair.keep) && nodes.has(pair.drop))
}

function addNeighbor(adjacency: Map<string, string[]>, from: string, to: string) {
  const neighbors = adjacency.get(from)
  if (neighbors) neighbors.push(to)
  else adjacency.set(from, [to])
}

/** Reuse topology lookups across collapse, selection, and neighbor highlighting. */
export function indexGraph(graph: GraphData): GraphIndex {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
  const children = new Map<string, string[]>()
  const neighbors = new Map<string, string[]>()
  for (const edge of graph.edges) {
    if (!nodes.has(edge.source) || !nodes.has(edge.target)) continue
    addNeighbor(neighbors, edge.source, edge.target)
    addNeighbor(neighbors, edge.target, edge.source)
    if (edge.relation === 'HAS_SUBTOPIC' || edge.relation === 'HAS_CONTENT') {
      addNeighbor(children, edge.source, edge.target)
    }
  }
  return { nodes, children, neighbors }
}

/** Discover on enqueue, so cycles and dense components visit each vertex once. */
export function reachableNodeIds(adjacency: Map<string, string[]>, roots: Iterable<string>): Set<string> {
  const visited = new Set(roots)
  const queue = [...visited]
  for (let position = 0; position < queue.length; position++) {
    for (const neighbor of adjacency.get(queue[position]) ?? []) {
      if (visited.has(neighbor)) continue
      visited.add(neighbor)
      queue.push(neighbor)
    }
  }
  return visited
}

export function collapseGraph(graph: GraphData, index: GraphIndex, collapsedTopics: Set<string>) {
  const countMap = new Map<string, number>()
  if (collapsedTopics.size === 0) {
    return { filteredNodes: graph.nodes, filteredEdges: graph.edges, collapseCountMap: countMap }
  }

  // A shared subtopic remains visible if any expanded topic reaches it.
  const expandedTopics = graph.nodes
    .filter((node) => node.node_type === 'topic' && !collapsedTopics.has(node.id))
    .map((node) => node.id)
  const protectedIds = reachableNodeIds(index.children, expandedTopics)
  const hiddenIds = new Set<string>()
  for (const topicId of collapsedTopics) {
    let count = 0
    for (const descendant of reachableNodeIds(index.children, [topicId])) {
      if (protectedIds.has(descendant) || index.nodes.get(descendant)?.node_type === 'topic') continue
      hiddenIds.add(descendant)
      count++
    }
    countMap.set(topicId, count)
  }
  return {
    filteredNodes: graph.nodes.filter((node) => !hiddenIds.has(node.id)),
    filteredEdges: graph.edges.filter((edge) => !hiddenIds.has(edge.source) && !hiddenIds.has(edge.target)),
    collapseCountMap: countMap,
  }
}

/** Include edges: a newly connected or removed relationship also changes layout. */
export function graphTopologyKey(graph: GraphData): string {
  return JSON.stringify([
    graph.nodes.map((node) => node.id),
    graph.edges.map((edge) => [edge.id, edge.source, edge.target]),
  ])
}

export interface NodeEvidenceState {
  nodeId: string
  version: number
  evidence: EvidenceChunk[]
}

/** A graph refresh invalidates evidence even when its deterministic node id survives. */
export function evidenceForNode(cache: NodeEvidenceState | null, nodeId: string | null, version: number): EvidenceChunk[] {
  return cache?.nodeId === nodeId && cache.version === version ? cache.evidence : []
}

/** Apply a persisted node patch while a full graph refresh is pending or unavailable. */
export function applyNodeUpdates(graph: GraphData, nodeId: string, updates: Record<string, unknown>): GraphData {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => node.id === nodeId ? { ...node, ...updates } : node),
  }
}
