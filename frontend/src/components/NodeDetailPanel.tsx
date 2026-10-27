import { useState } from 'react'
import type { GraphNode, GraphEdge, EvidenceChunk } from '../types'
import { useTheme } from '../theme'
import RichText from './LazyRichText'
import { evidenceWithContext, openSourceEvidence, sourceDisplayName, sourceLocation, sourceUrl } from '../utils/source'
import { errorMessage, resolveAssetUrl } from '../utils/api'

/**
 * Detail view for one graph node. Render with `key={node.id}` so that edit
 * state starts fresh whenever a different node is selected.
 */
interface Props {
  node: GraphNode
  edges: GraphEdge[]
  allNodes: GraphNode[]
  evidence: EvidenceChunk[]
  evidenceLoading?: boolean
  onClose: () => void
  editMode?: boolean
  onDeleteNode?: (nodeId: string) => void
  onDeleteEdge?: (source: string, relation: string, target: string) => void
  /** Resolves once the update is persisted; rejects with a displayable error. */
  onUpdateNode?: (nodeId: string, updates: Record<string, unknown>) => Promise<void>
}

const CONTENT_TYPES = ['definition', 'theorem', 'technique', 'example', 'question', 'figure', 'other']

/** Group evidence chunks by source_id so a paper with 5 cited pages shows up once. */
function groupEvidenceBySource(evidence: EvidenceChunk[]): Array<{
  source: string
  modality: string
  chunks: EvidenceChunk[]
}> {
  const map = new Map<string, { source: string; modality: string; chunks: EvidenceChunk[] }>()
  for (const chunk of evidence) {
    const key = chunk.source
    const entry = map.get(key) ?? { source: key, modality: chunk.modality, chunks: [] }
    entry.chunks.push(chunk)
    map.set(key, entry)
  }
  return Array.from(map.values())
}

function typeChip(nodeType: string, isDark: boolean): string {
  if (isDark) {
    switch (nodeType) {
      case 'topic':    return 'bg-indigo-950 text-indigo-300 border-indigo-800'
      case 'subtopic': return 'bg-purple-950 text-purple-300 border-purple-800'
      case 'content':  return 'bg-teal-950 text-teal-300 border-teal-800'
      case 'note': return 'bg-amber-950 text-amber-300 border-amber-800'
      default:         return 'bg-slate-800 text-slate-400 border-slate-700'
    }
  }
  switch (nodeType) {
    case 'topic':    return 'bg-indigo-50 text-indigo-700 border-indigo-200'
    case 'subtopic': return 'bg-purple-50 text-purple-700 border-purple-200'
    case 'content':  return 'bg-teal-50 text-teal-700 border-teal-200'
    case 'note': return 'bg-amber-50 text-amber-700 border-amber-200'
    default:         return 'bg-slate-50 text-slate-600 border-slate-200'
  }
}

interface Draft {
  name: string
  body: string
  summary: string
  scope: string
  content_type: string
  raw_excerpt: string
  key_terms: string   // comma-separated; split on save
  source_ids: string[]
}

function nodeToDraft(node: GraphNode): Draft {
  return {
    name: node.name ?? '',
    body: node.body ?? '',
    summary: node.summary ?? '',
    scope: node.scope ?? 'broad',
    content_type: node.content_type ?? 'other',
    raw_excerpt: node.raw_excerpt ?? '',
    key_terms: (node.key_terms ?? []).join(', '),
    source_ids: [...(node.source_ids ?? [])],
  }
}

export default function NodeDetailPanel({
  node,
  edges,
  allNodes,
  evidence,
  evidenceLoading = false,
  onClose,
  editMode = false,
  onDeleteNode,
  onDeleteEdge,
  onUpdateNode,
}: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const nodeMap = Object.fromEntries(allNodes.map((n) => [n.id, n]))
  const [excerptOpen, setExcerptOpen] = useState(false)
  const [illustrationOpen, setIllustrationOpen] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [draft, setDraft] = useState<Draft>(() => nodeToDraft(node))
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const connections = edges.filter((e) => e.source === node.id || e.target === node.id)
  const keyTerms = (node.key_terms ?? []).filter((t) => t && t.trim())
  const rawExcerpt = (node.raw_excerpt ?? '').trim()
  const illustrationSrc = resolveAssetUrl(node.illustration_path)

  const handleDeleteNode = () => {
    if (!confirmDelete) { setConfirmDelete(true); return }
    onDeleteNode?.(node.id)
    onClose()
  }

  const startEditing = () => {
    setDraft(nodeToDraft(node))
    setSaveError(null)
    setIsEditing(true)
  }

  const cancelEditing = () => {
    setSaveError(null)
    setIsEditing(false)
  }

  const handleSave = async () => {
    if (saving) return
    const updates: Record<string, unknown> = {
      name: draft.name.trim() || node.name,
      summary: draft.summary.trim(),
    }
    if (node.node_type === 'topic') updates.scope = draft.scope
    if (node.node_type === 'note') { updates.body = draft.body; updates.title = draft.name.trim() || node.name }
    if (node.node_type === 'content') {
      updates.content_type = draft.content_type
      updates.raw_excerpt = draft.raw_excerpt.trim()
      updates.key_terms = draft.key_terms
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean)
    }
    updates.source_ids = draft.source_ids
    setSaving(true)
    setSaveError(null)
    try {
      await onUpdateNode?.(node.id, updates)
      setIsEditing(false)
    } catch (e) {
      setSaveError(`Save failed: ${errorMessage(e)}`)
    } finally {
      setSaving(false)
    }
  }

  // Shared input/textarea styles
  const inputCls = `w-full px-2 py-1 rounded-lg border text-xs outline-none transition-colors ${
    isDark
      ? 'bg-slate-800 border-slate-600 text-slate-200 placeholder-slate-500 focus:border-indigo-500'
      : 'bg-white border-slate-300 text-slate-800 placeholder-slate-400 focus:border-indigo-400'
  }`

  const panelBase = `text-sm overflow-y-auto ${isDark ? 'bg-slate-900' : 'bg-white'}`
  const panelClass = isFullscreen
    ? `fixed inset-0 z-50 p-6 ${panelBase}`
    : `border-t p-4 max-h-[50vh] ${panelBase} ${isDark ? 'border-slate-700' : 'border-slate-200'}`

  return (
    <div className={panelClass}>
      {confirmDelete && <div role="alertdialog" aria-label="Delete node" className="rounded-lg border border-red-500/30 p-3 mb-3 text-xs"><p>Delete “{node.name}” and its connections?</p><div className="flex gap-3 mt-2"><button onClick={handleDeleteNode} className="text-red-500">Delete node</button><button onClick={() => setConfirmDelete(false)}>Cancel</button></div></div>}
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <input
              value={draft.name}
              onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
              className={`${inputCls} font-semibold text-sm mb-1`}
              placeholder="Node name"
            />
          ) : (
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className={`font-semibold text-base ${isDark ? 'text-white' : 'text-slate-900'}`}>
                {node.name}
              </h3>
              <span className={`text-[10px] uppercase tracking-wider font-medium px-1.5 py-0.5 rounded border ${typeChip(node.node_type, isDark)}`}>
                {node.node_type}
              </span>
              {node.scope && (
                <span className={`text-[10px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  ({node.scope})
                </span>
              )}
              {node.content_type && node.content_type !== 'other' && (
                <span className={`text-[10px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  ({node.content_type})
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0 ml-2">
          {editMode && !isEditing && (
            <>
              <button
                onClick={startEditing}
                className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                  isDark
                    ? 'text-indigo-400 hover:text-indigo-300 border-indigo-800 hover:border-indigo-600'
                    : 'text-indigo-600 hover:text-indigo-700 border-indigo-300 hover:border-indigo-400'
                }`}
                title="Edit this node"
              >
                Edit
              </button>
              <button
                onClick={handleDeleteNode}
                className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                  isDark
                    ? 'text-red-400 hover:text-red-300 border-red-900 hover:border-red-700'
                    : 'text-red-600 hover:text-red-700 border-red-300 hover:border-red-400'
                }`}
                title="Delete this node and all its edges"
              >
                Delete
              </button>
            </>
          )}
          {isEditing && (
            <>
              <button
                onClick={handleSave}
                disabled={saving}
                className={`text-xs px-2 py-0.5 rounded border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  isDark
                    ? 'text-teal-400 hover:text-teal-300 border-teal-800 hover:border-teal-600'
                    : 'text-teal-700 hover:text-teal-800 border-teal-300 hover:border-teal-400'
                }`}
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
              <button
                onClick={cancelEditing}
                disabled={saving}
                className={`text-xs px-2 py-0.5 rounded border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  isDark
                    ? 'text-slate-400 hover:text-slate-300 border-slate-700'
                    : 'text-slate-500 hover:text-slate-700 border-slate-300'
                }`}
              >
                Cancel
              </button>
            </>
          )}
          {!isEditing && (
            <>
              <button
                onClick={() => setIsFullscreen((v) => !v)}
                className={`text-sm leading-none transition-colors ${isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'}`}
                title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
              >
                {isFullscreen ? '⊡' : '⊞'}
              </button>
              <button
                onClick={onClose}
                className={`text-lg leading-none transition-colors ml-1 ${isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'}`}
              >
                ×
              </button>
            </>
          )}
        </div>
      </div>

      {isEditing && saveError && (
        <p role="alert" className={`text-xs rounded-lg px-2 py-1.5 mb-2 border break-words ${
          isDark ? 'text-red-400 bg-red-950/40 border-red-800' : 'text-red-600 bg-red-50 border-red-300'
        }`}>
          {saveError}
        </p>
      )}

      {/* Scope selector — topic nodes, edit mode only */}
      {isEditing && node.node_type === 'topic' && (
        <div className="flex items-center gap-3 mb-2 text-xs">
          <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>Scope</span>
          {(['broad', 'narrow'] as const).map((s) => (
            <label key={s} className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="scope"
                value={s}
                checked={draft.scope === s}
                onChange={() => setDraft((d) => ({ ...d, scope: s }))}
                className="accent-indigo-500"
              />
              <span className={isDark ? 'text-slate-300' : 'text-slate-700'}>{s}</span>
            </label>
          ))}
        </div>
      )}

      {/* Content type selector — content nodes, edit mode only */}
      {isEditing && node.node_type === 'content' && (
        <div className="mb-2">
          <label className={`text-[10px] uppercase tracking-wider font-medium block mb-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Content type
          </label>
          <select
            value={draft.content_type}
            onChange={(e) => setDraft((d) => ({ ...d, content_type: e.target.value }))}
            className={`${inputCls}`}
          >
            {CONTENT_TYPES.map((ct) => (
              <option key={ct} value={ct}>{ct}</option>
            ))}
          </select>
        </div>
      )}

      {/* Summary */}
      {isEditing ? (
        <div className="mb-3">
          <label className={`text-[10px] uppercase tracking-wider font-medium block mb-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Summary
          </label>
          <textarea
            value={draft.summary}
            onChange={(e) => setDraft((d) => ({ ...d, summary: e.target.value }))}
            rows={4}
            className={`${inputCls} resize-y`}
            placeholder="Node summary"
          />
        </div>
      ) : node.summary ? (
        <RichText
          text={node.summary}
          className={`text-xs leading-relaxed mb-3 prose-sm ${isDark ? 'text-slate-300' : 'text-slate-600'}`}
        />
      ) : null}

      {/* Key terms — pill chips (read-only) or comma input (edit) */}
      {node.node_type === 'note' && (isEditing ? <div className="mb-4 space-y-2"><label className="block text-xs text-slate-500">Note · Markdown, LaTeX and [[wikilinks]]</label><textarea aria-label="Note body" value={draft.body} onChange={(event) => setDraft((previous) => ({ ...previous, body: event.target.value }))} className={`${inputCls} min-h-36 resize-y font-mono`} /><p className="text-[10px] uppercase text-slate-500">Live preview</p><RichText text={draft.body} className="text-xs rounded-lg border border-slate-500/20 p-3" /></div> : <RichText text={node.body ?? ''} className="text-xs leading-relaxed mb-4" />)}
      {isEditing && node.node_type === 'content' && <div className="mb-3"><p className="text-[10px] uppercase text-slate-500 mb-1">Live preview</p><RichText text={draft.summary} className="text-xs border border-slate-500/20 rounded-lg p-2" /></div>}
      {isEditing && node.node_type === 'content' ? (
        <div className="mb-4">
          <label className={`text-[10px] uppercase tracking-wider font-medium block mb-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Key terms <span className={`normal-case font-normal ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>(comma-separated)</span>
          </label>
          <input
            value={draft.key_terms}
            onChange={(e) => setDraft((d) => ({ ...d, key_terms: e.target.value }))}
            className={inputCls}
            placeholder="term1, term2, term3"
          />
        </div>
      ) : keyTerms.length > 0 ? (
        <div className="flex flex-wrap gap-1 mb-4">
          {keyTerms.map((term) => (
            <span
              key={term}
              className={`text-[10px] px-2 py-0.5 rounded-full border ${
                isDark
                  ? 'bg-violet-950 text-violet-300 border-violet-800'
                  : 'bg-violet-50 text-violet-700 border-violet-200'
              }`}
            >
              {term}
            </span>
          ))}
        </div>
      ) : null}

      {/* Source Text — collapsible (read-only) or textarea (edit) */}
      {isEditing && node.node_type === 'content' ? (
        <div className="mb-4">
          <label className={`text-[10px] uppercase tracking-wider font-medium block mb-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Source excerpt
          </label>
          <textarea
            value={draft.raw_excerpt}
            onChange={(e) => setDraft((d) => ({ ...d, raw_excerpt: e.target.value }))}
            rows={3}
            className={`${inputCls} resize-y font-mono`}
            placeholder="Verbatim quote from source"
          />
        </div>
      ) : rawExcerpt ? (
        <div className="mb-4">
          <button
            onClick={() => setExcerptOpen((v) => !v)}
            className={`text-[10px] uppercase tracking-wider font-medium flex items-center gap-1 ${
              isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'
            }`}
          >
            <span>{excerptOpen ? '▼' : '▶'}</span> Source Text
          </button>
          {excerptOpen && (
            <div
              className={`mt-1.5 text-xs italic font-mono leading-relaxed rounded p-2 border ${
                isDark
                  ? 'border-slate-700 bg-slate-800/50 text-slate-300'
                  : 'border-slate-200 bg-slate-50 text-slate-700'
              }`}
            >
              <RichText text={rawExcerpt} />
            </div>
          )}
        </div>
      ) : null}

      {/* Source IDs — removable list in edit mode */}
      {isEditing && draft.source_ids.length > 0 && (
        <div className="mb-4">
          <p className={`text-[10px] uppercase tracking-wider font-medium mb-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Source documents
          </p>
          <div className="flex flex-col gap-1">
            {draft.source_ids.map((sid) => (
              <div key={sid} className={`flex items-center justify-between rounded px-2 py-1 border text-xs ${
                isDark ? 'border-slate-700 bg-slate-800/50 text-slate-300' : 'border-slate-200 bg-slate-50 text-slate-700'
              }`}>
                <span className="truncate">{sid}</span>
                <button
                  onClick={() => setDraft((d) => ({ ...d, source_ids: d.source_ids.filter((s) => s !== sid) }))}
                  className="ml-2 shrink-0 text-red-500 hover:text-red-400 leading-none"
                  title="Remove this source"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Illustration — collapsible; image if resolved, KaTeX for equations, pending badge for others */}
      {(illustrationSrc || node.illustration) && (
        <div className="mb-4">
          <button
            onClick={() => setIllustrationOpen((v) => !v)}
            className={`text-[10px] uppercase tracking-wider font-medium flex items-center gap-1 mb-1 ${
              isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'
            }`}
          >
            <span>{illustrationOpen ? '▼' : '▶'}</span>
            Illustration{node.illustration ? ` · ${node.illustration.kind}` : ''}
            {!illustrationSrc && node.illustration && node.illustration.kind !== 'equation' && (
              <span className={`ml-1 text-[9px] px-1.5 py-0.5 rounded-full border ${
                isDark ? 'bg-amber-950 text-amber-400 border-amber-800' : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}>pending</span>
            )}
          </button>
          {illustrationOpen && (
            illustrationSrc ? (
              <img
                src={illustrationSrc}
                alt={node.illustration?.hint ?? node.name}
                className={`rounded border w-full ${isDark ? 'border-slate-700' : 'border-slate-200'}`}
                loading="lazy"
              />
            ) : node.illustration?.kind === 'equation' && node.illustration.hint ? (
              <div className={`rounded border p-3 ${isDark ? 'border-slate-700 bg-slate-800/50' : 'border-slate-200 bg-slate-50'}`}>
                <RichText text={node.illustration.hint.includes('$') ? node.illustration.hint : `$$\n${node.illustration.hint}\n$$`} />
              </div>
            ) : node.illustration ? (
              <p className={`text-xs ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{node.illustration.hint}</p>
            ) : null
          )}
        </div>
      )}

      {/* Connections retain hierarchy without implying a semantic direction. */}
      {connections.length > 0 && (
        <div className="space-y-1 mb-3">
          <p className={`text-[10px] uppercase tracking-wider font-medium ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Connections
          </p>
          {connections.map((e) => {
            const otherId = e.source === node.id ? e.target : e.source
            const hierarchy = e.relation === 'RELATED_TO' ? null : e.target === node.id ? 'Parent' : e.relation === 'HAS_SUBTOPIC' ? 'Subtopic' : 'Content'
            return <div key={e.id} className={`flex items-center gap-2 text-xs group ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              <span className="shrink-0">{hierarchy ? `${hierarchy}:` : '—'}</span>
              <span className={`font-medium truncate ${isDark ? 'text-slate-300' : 'text-slate-700'}`} title={nodeMap[otherId]?.name ?? otherId}>
                {nodeMap[otherId]?.name ?? otherId}
              </span>
              {editMode && (
                <button
                  onClick={() => onDeleteEdge?.(e.source, e.relation, e.target)}
                  className="ml-auto opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-300 transition-opacity text-xs leading-none px-1 shrink-0"
                  title="Delete this connection"
                >
                  ×
                </button>
              )}
            </div>
          })}
        </div>
      )}

      {/* Sources — one card per unique source file, with deep-link buttons */}
      <div>
        <p className={`text-xs mb-1.5 font-semibold uppercase tracking-wider ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
          Sources
        </p>
        {evidenceLoading && evidence.length === 0 ? (
          <p className={`text-xs italic ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Loading…</p>
        ) : evidence.length === 0 ? (
          <p className={`text-xs italic ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            No sources linked to this node.
          </p>
        ) : (
          groupEvidenceBySource(evidence).map((group) => (
            <SourceCard
              key={group.source}
              group={group}
              isDark={isDark}
            />
          ))
        )}
      </div>
    </div>
  )
}


function SourceCard({
  group,
  isDark,
}: {
  group: { source: string; modality: string; chunks: EvidenceChunk[] }
  isDark: boolean
}) {
  const first = evidenceWithContext(group.chunks[0])
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null)
  const primaryUrl = sourceUrl(group.source, group.modality, first.page, first.timestamp)

  return (
    <div
      className={`text-xs rounded p-2 mb-2 border ${
        isDark ? 'border-slate-700 bg-slate-800/50' : 'border-slate-200 bg-slate-50'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="min-w-0">
          <p className={`font-medium truncate ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
            {sourceDisplayName(first)}
          </p>
          <p className={`text-[10px] uppercase tracking-wider ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            {group.modality} · {group.chunks.length} chunk{group.chunks.length > 1 ? 's' : ''}
          </p>
        </div>
        {group.modality !== 'note' && <a
          href={primaryUrl}
          target="_blank"
          rel="noreferrer"
          className={`text-[10px] px-2 py-0.5 rounded border shrink-0 transition-colors ${
            isDark
              ? 'text-indigo-300 border-indigo-700 hover:bg-indigo-900/40'
              : 'text-indigo-700 border-indigo-300 hover:bg-indigo-50'
          }`}
          title={`Open original ${sourceDisplayName(first)}`}
        >
          Open ↗
        </a>}
      </div>
      <ul className="space-y-1">
        {group.chunks.slice(0, 4).map((chunk) => {
          const full = evidenceWithContext(chunk)
          const deepUrl = sourceUrl(full.source, full.modality, full.page, full.timestamp)
          const locator = sourceLocation(full)
          const expanded = expandedChunkId === chunk.id
          return (
            <li key={chunk.id} className={`${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              <div className="flex items-baseline gap-1.5">
                {locator && (
                  <a
                    href={deepUrl}
                    target="_blank"
                    rel="noreferrer"
                    className={`text-[10px] font-mono shrink-0 ${
                      isDark ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-800'
                    }`}
                    title="Open at this location"
                  >
                    {locator}
                  </a>
                )}
                {!expanded && <span className="line-clamp-2 text-[11px] leading-snug">{full.text}</span>}
              </div>
              {expanded && <RichText text={full.text} className="text-xs leading-relaxed whitespace-pre-wrap mt-2" />}
              <button aria-expanded={expanded} className="mt-1 text-[11px] underline underline-offset-2" onClick={() => setExpandedChunkId(expanded ? null : chunk.id)}>{expanded ? 'Collapse passage' : 'Show full passage'}</button>
              <button className="mt-1 ml-3 text-[11px] underline underline-offset-2" onClick={() => openSourceEvidence(chunk)}>View in source</button>
            </li>
          )
        })}
        {group.chunks.length > 4 && (
          <li className={`text-[10px] italic ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            + {group.chunks.length - 4} more chunk{group.chunks.length - 4 > 1 ? 's' : ''}
          </li>
        )}
      </ul>
    </div>
  )
}
