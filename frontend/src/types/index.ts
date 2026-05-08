export type RoutingMode = 'dense' | 'graph' | 'hybrid' | 'none'

export type NodeType = 'topic' | 'subtopic' | 'content'

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
  x?: number
  y?: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  relation: string         // HAS_SUBTOPIC | HAS_CONTENT | RELATED_TO
  label?: string | null    // free-form label on RELATED_TO edges
  confidence?: number | null
  source_chunk_ids?: string[]
  highlighted?: boolean
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
  retrieval_source?: 'dense' | 'graph' | 'both'
  relevancy_score?: number  // cosine similarity 0–1, set when reranker ran
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

// ---------------------------------------------------------------------------
// Quiz types
// ---------------------------------------------------------------------------

export type QuestionType = 'multiple_choice' | 'short_answer' | 'true_false'
export type QuizDifficulty = 'beginner' | 'intermediate' | 'advanced'

export interface QuizConfig {
  num_questions: 5 | 10 | 20
  question_types: QuestionType[]
  focus_area?: string
  difficulty: QuizDifficulty
}

export interface QuizQuestion {
  id: string
  question_type: QuestionType
  question_text: string
  options?: string[] | null
  difficulty: string
}

export interface GeneratedQuiz {
  quiz_id: string
  questions: QuizQuestion[]
}

export interface QuizJobStatus {
  status: 'queued' | 'running' | 'done' | 'failed'
  progress: string
  quiz_id?: string | null
  questions?: QuizQuestion[] | null
  error?: string | null
}

export interface QuizAnswer {
  question_id: string
  user_answer: string
}

export interface QuestionResult {
  question_id: string
  question_text: string
  question_type: string
  user_answer: string
  correct_answer: string
  is_correct: boolean
  score: number
  explanation: string
  evidence: EvidenceChunk[]
}

export interface QuizGradeResult {
  quiz_id: string
  total_score: number
  num_correct: number
  num_total: number
  results: QuestionResult[]
  knowledge_gaps: string[]
  recommended_study_areas: string[]
}

export interface HintData {
  hint: string
  related_concepts: string[]
}

export interface DeepDiveData {
  concept: string
  study_guide: string
  key_takeaways: string[]
  evidence: EvidenceChunk[]
}
