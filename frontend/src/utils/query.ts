import type { Message, QueryRoutingMode } from '../types'
import { apiJson } from './api'

export interface QuerySettings {
  routingMode: QueryRoutingMode
  queryMode: 'fast' | 'thorough'
  maxSources?: number
  minRelevancy?: number
}
export interface QueryOptions extends QuerySettings {
  signal?: AbortSignal
}
export const DEFAULT_QUERY_SETTINGS: Readonly<QuerySettings> = Object.freeze({ routingMode: 'auto', queryMode: 'fast' })

/** Preserve the server's graph IDs, evidence order and per-chunk provenance verbatim. */
export function queryWorkspace(query: string, options: QueryOptions): Promise<Omit<Message, 'id' | 'timestamp'>> {
  const body: Record<string, string | number> = { query, query_mode: options.queryMode }
  if (options.routingMode !== 'auto') body.routing_mode = options.routingMode
  if (options.maxSources !== undefined) body.max_sources = options.maxSources
  if (options.minRelevancy !== undefined && options.minRelevancy > 0) body.min_relevancy = options.minRelevancy
  return apiJson('/api/query', { method: 'POST', json: body, signal: options.signal })
}
