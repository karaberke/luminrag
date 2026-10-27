import { z } from 'zod'

const text = z.string().max(2_000_000)
const identifier = z.string().min(1).max(500)
const strings = z.array(z.string().max(10_000)).max(20_000)
const sourceIds = z.array(z.string().min(1).max(1024)).max(20_000)
const connectionSupport = z.strictObject({
  kind: z.enum(['automatic', 'manual', 'wikilink', 'legacy']),
  owner: identifier.nullable().default(null),
  source_chunk_ids: sourceIds.default([]),
  confidence: z.number().finite().min(0).max(1).default(1),
  verified: z.boolean().default(false),
}).refine((support) => !support.verified || support.kind === 'automatic', 'Only automatic source checks can be verified.')
  .refine((support) => !support.verified || support.source_chunk_ids.length > 0, 'Verified connections require supporting passages.')
  .refine((support) => support.kind !== 'wikilink' || support.owner !== null, 'A note connection requires its owning note.')
export const graphConnectionsSchema = z.object({
  nodes: z.array(z.object({ id: identifier }).passthrough()).max(30_000),
  edges: z.array(z.object({
    id: z.string().min(1).max(2048), source: identifier, target: identifier,
    relation: z.string().min(1).max(200), label: z.string().max(200).nullable().optional(),
    confidence: z.number().finite().min(0).max(1).nullable().optional(),
    source_chunk_ids: sourceIds.optional(), supports: z.array(connectionSupport).max(20_000).optional(),
  }).passthrough().refine((edge) => edge.relation !== 'RELATED_TO' || edge.supports === undefined || edge.supports.length > 0,
    'A connection requires supporting data.').refine((edge) => edge.supports?.every((support) => support.kind !== 'wikilink' || support.owner === edge.source || support.owner === edge.target) ?? true,
    'A note connection must include its owning note.')).max(100_000),
  association_version: z.literal(1).optional(),
}).passthrough()
const evidence = z.object({ id: identifier, text, source: identifier,
  modality: z.enum(['video', 'slide', 'pdf', 'image', 'audio', 'note', 'document', 'markdown', 'office']),
  page: z.number().int().positive().nullable().optional(), page_end: z.number().int().positive().nullable().optional(),
  slide_number: z.number().int().positive().nullable().optional(), timestamp: z.string().max(50).nullable().optional(),
  source_filename: z.string().max(2048).nullable().optional(), title: z.string().max(2048).nullable().optional(), section_title: z.string().max(2048).nullable().optional(),
  quote: z.string().max(100_000).nullable().optional(), quote_start: z.number().int().nonnegative().nullable().optional(), quote_end: z.number().int().nonnegative().nullable().optional(),
  retrieval_source: z.enum(['dense', 'graph', 'both']).nullable().optional(), relevancy_score: z.number().finite().nullable().optional(),
})
const message = z.object({ id: identifier, role: z.enum(['user', 'assistant']), content: text, timestamp: z.coerce.date(),
  routing_mode: z.enum(['dense', 'graph', 'hybrid', 'none']).optional(), evidence: z.array(evidence).max(10_000).optional(),
  reflection: z.object({ needs_retrieval: z.boolean(), is_relevant: z.boolean(), is_supported: z.boolean(), is_useful: z.boolean(), reasoning: text }).optional(),
  hops: strings.optional(),
})
export const browserMetadataSchema = z.object({
  name: z.string().min(1).max(100).default('Imported knowledge base'),
  originals: z.array(z.object({ path: z.string().startsWith('originals/').max(500), name: z.string().min(1).max(500),
    source: identifier, type: z.string().max(200).optional() })).max(1000).default([]),
  chats: z.array(message).max(10000).default([]),
})
