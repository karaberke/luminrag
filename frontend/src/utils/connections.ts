import type { ConnectionSupport, GraphData, GraphEdge } from '../types'

export const ASSOCIATION_VERSION = 1
const SEMANTIC_RELATION = 'RELATED_TO'
const ASSOCIATION_ID = /^assoc:[0-9a-f]{24}$/
const MAX_SUPPORTS = 20_000

/** Python sorts Unicode code points; native JS sort uses UTF-16 code units. */
function compareIds(left: string, right: string): number {
  const a = Array.from(left, (character) => character.codePointAt(0)!)
  const b = Array.from(right, (character) => character.codePointAt(0)!)
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (a[index] !== b[index]) return a[index] - b[index]
  }
  return a.length - b.length
}

function connectionPair(source: string, target: string): [string, string] {
  return compareIds(source, target) <= 0 ? [source, target] : [target, source]
}

/** Match the backend's canonical JSON/SHA-256 ID without another dependency. */
export async function associationId(source: string, target: string): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(connectionPair(source, target)))
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return `assoc:${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('').slice(0, 24)}`
}

function legacySupport(edge: GraphEdge, notes: Set<string>): ConnectionSupport {
  const wikilink = 'wikilink' in edge && edge.wikilink === true
  const origin = 'origin' in edge ? edge.origin : null
  const kind = origin === 'manual' || origin === 'automatic' || origin === 'legacy' ? origin : 'legacy'
  const owner = wikilink ? notes.has(edge.source) ? edge.source : notes.has(edge.target) ? edge.target : edge.source : null
  return { kind: wikilink ? 'wikilink' : kind, owner,
    source_chunk_ids: edge.source_chunk_ids?.length ? edge.source_chunk_ids : owner ? [`note:${owner}`] : [],
    confidence: edge.confidence ?? 0.5, verified: false }
}

interface AssociationGroup { edge: GraphEdge; supports: Map<string, ConnectionSupport> }

function addSupport(group: AssociationGroup, support: ConnectionSupport): void {
  const ids = [...new Set(support.source_chunk_ids)].sort(compareIds)
  const key = JSON.stringify([support.kind, support.owner, ids, support.verified])
  const previous = group.supports.get(key)
  group.supports.set(key, { ...support, source_chunk_ids: ids, confidence: Math.max(previous?.confidence ?? 0, support.confidence) })
  if (group.supports.size > MAX_SUPPORTS) throw new Error('This connection contains too much supporting data.')
}

/** Normalize API graphs only. NetworkX backup payloads remain server-owned. */
export async function normalizeGraphConnections(data: GraphData): Promise<GraphData> {
  const { graphConnectionsSchema } = await import('./validation')
  const parsed = graphConnectionsSchema.parse(data)
  const notes = new Set(data.nodes.filter((node) => node.node_type === 'note').map((node) => node.id))
  const groups = new Map<string, AssociationGroup>()
  const edges: GraphEdge[] = []
  for (const input of parsed.edges) {
    const edge = input as GraphEdge
    if (edge.relation !== SEMANTIC_RELATION) { edges.push(edge); continue }
    const [source, target] = connectionPair(edge.source, edge.target)
    const pair = JSON.stringify([source, target])
    let group = groups.get(pair)
    if (!group) {
      group = { edge: { ...edge, source, target, label: null }, supports: new Map() }
      groups.set(pair, group)
      edges.push(group.edge)
    }
    if (ASSOCIATION_ID.test(edge.id)) group.edge.id = edge.id
    group.edge.highlighted = Boolean(group.edge.highlighted || edge.highlighted)
    for (const support of edge.supports ?? [legacySupport(edge, notes)]) addSupport(group, support)
  }
  const associations = [...groups.values()]
  // Bound native digest promises when reading an old graph with many pairs.
  for (let start = 0; start < associations.length; start += 32) {
    await Promise.all(associations.slice(start, start + 32).map(async ({ edge, supports }) => {
      if (!ASSOCIATION_ID.test(edge.id)) edge.id = await associationId(edge.source, edge.target)
      edge.supports = [...supports.values()]
      const ids = new Set<string>()
      let confidence = 0
      for (const support of edge.supports) {
        confidence = Math.max(confidence, support.confidence)
        for (const id of support.source_chunk_ids) ids.add(id)
      }
      if (ids.size > MAX_SUPPORTS) throw new Error('This connection contains too many source references.')
      edge.source_chunk_ids = [...ids].sort(compareIds)
      edge.confidence = confidence
    }))
  }
  return { ...data, edges, association_version: ASSOCIATION_VERSION }
}
