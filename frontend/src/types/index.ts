export type RoutingMode = 'dense' | 'graph' | 'none'

export interface GraphNode {
  id: string
  label: string
  type: 'concept' | 'topic' | 'entity'
  highlighted: boolean
  hopIndex?: number   // position in the retrieval hop chain; drives animation stagger
  x?: number
  y?: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  relation: string
  highlighted: boolean
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface EvidenceChunk {
  id: string
  text: string
  source: string
  modality: 'video' | 'slide' | 'pdf' | 'image' | 'audio'
  page?: number
  timestamp?: string
}

export interface ReflectionVerdict {
  needs_retrieval: boolean
  is_relevant: boolean
  is_supported: boolean
  is_useful: boolean
  reasoning: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  routing_mode?: RoutingMode
  evidence?: EvidenceChunk[]
  reflection?: ReflectionVerdict
  hops?: string[]
  timestamp: Date
}
