export type RoutingMode = 'dense' | 'graph' | 'hybrid' | 'none'
export type QueryRoutingMode = 'auto' | Exclude<RoutingMode, 'none'>

export type NodeType = 'topic' | 'subtopic' | 'content' | 'note'

export type ContentType =
  | 'definition'
  | 'theorem'
  | 'technique'
  | 'example'
  | 'question'
  | 'figure'
  | 'other'

export type IllustrationKind = 'diagram' | 'equation' | 'code' | 'image'

export interface Illustration {
  kind: IllustrationKind
  hint: string
}

export interface GraphNode {
  id: string
  name: string
  node_type: NodeType
  source_ids: string[]
  title?: string
  body?: string
  // Topic / Subtopic / Content
  summary?: string | null
  // Topic / Subtopic
  scope?: 'broad' | 'narrow' | null
  illustration?: Illustration | null
  parent_topic_keys?: string[]
  // Content
  content_type?: ContentType | null
  parent_subtopic_keys?: string[]
  raw_excerpt?: string | null
  key_terms?: string[]
  illustration_path?: string | null
  // Client-side render state
  highlighted?: boolean
  hopIndex?: number
  x?: number | null
  y?: number | null
}

export interface ConnectionSupport {
  kind: 'automatic' | 'manual' | 'wikilink' | 'legacy'
  owner: string | null
  source_chunk_ids: string[]
  confidence: number
  verified: boolean
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  relation: string         // HAS_SUBTOPIC | HAS_CONTENT | RELATED_TO
  label?: string | null    // Legacy input only; semantic connections display no label.
  confidence?: number | null
  source_chunk_ids?: string[]
  supports?: ConnectionSupport[]
  highlighted?: boolean
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  association_version?: number
  /** Client-side identity for one query result, including repeated identical traces. */
  traceId?: string
}

/** Inverse operations returned by the backend can include an internal restore payload. */
export interface GraphOp { op: string; [key: string]: unknown }
export interface GraphOpsResponse { graph?: GraphData; version: number; inverses?: GraphOp[]; inverse?: GraphOp[] }
/** `lumin-graph-changed` detail; only a position-only save for the same workspace keeps the active trace. */
export interface GraphChangedDetail { workspaceId: string; positionOnly: boolean; graph?: GraphData }

export interface EvidenceChunk {
  id: string
  text: string
  source: string
  modality: 'video' | 'slide' | 'pdf' | 'image' | 'audio' | 'note' | 'document' | 'markdown' | 'office'
  page?: number | null
  page_end?: number | null
  slide_number?: number | null
  timestamp?: string | null
  source_filename?: string | null
  title?: string | null
  section_title?: string | null
  quote?: string | null
  quote_start?: number | null
  quote_end?: number | null
  retrieval_source?: 'dense' | 'graph' | 'both' | null
  relevancy_score?: number | null  // cosine similarity 0–1, set when reranker ran
}

export interface ReflectionVerdict {
  verification_status?: 'verified' | 'skipped' | 'unavailable'
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
