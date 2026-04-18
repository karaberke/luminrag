import type { GraphNode, GraphEdge, EvidenceChunk } from '../types'
import { useTheme } from '../ThemeContext'

interface Props {
  node: GraphNode
  edges: GraphEdge[]
  allNodes: GraphNode[]
  evidence: EvidenceChunk[]
  onClose: () => void
  editMode?: boolean
  onDeleteNode?: (nodeId: string) => void
  onDeleteEdge?: (source: string, relation: string, target: string) => void
}

export default function NodeDetailPanel({
  node,
  edges,
  allNodes,
  evidence,
  onClose,
  editMode = false,
  onDeleteNode,
  onDeleteEdge,
}: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const nodeMap = Object.fromEntries(allNodes.map((n) => [n.id, n]))

  const outgoing = edges.filter((e) => e.source === node.id)
  const incoming = edges.filter((e) => e.target === node.id)

  const handleDeleteNode = () => {
    if (!window.confirm(`Delete "${node.label}" and all its edges?`)) return
    onDeleteNode?.(node.id)
    onClose()
  }

  return (
    <div className={`border-t p-4 text-sm ${
      isDark ? 'border-slate-700 bg-slate-900' : 'border-slate-200 bg-white'
    }`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className={`font-semibold text-base ${isDark ? 'text-white' : 'text-slate-900'}`}>{node.label}</h3>
          <span className={`text-xs capitalize ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>{node.type}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {editMode && (
            <button
              onClick={handleDeleteNode}
              className="text-xs text-red-400 hover:text-red-300 border border-red-900 hover:border-red-700 px-2 py-0.5 rounded transition-colors"
              title="Delete this node and all its edges"
            >
              Delete
            </button>
          )}
          <button
            onClick={onClose}
            className={`text-lg leading-none transition-colors ml-1 ${isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'}`}
          >
            ×
          </button>
        </div>
      </div>

      {/* Relations */}
      <div className="space-y-2 mb-3">
        {outgoing.map((e) => (
          <div key={e.id} className={`flex items-center gap-2 text-xs group ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            <span className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{node.label}</span>
            <span className="text-indigo-400 font-mono">—{e.relation}→</span>
            <span className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{nodeMap[e.target]?.label ?? e.target}</span>
            {editMode && (
              <button
                onClick={() => onDeleteEdge?.(e.source, e.relation, e.target)}
                className="ml-auto opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-300 transition-opacity text-xs leading-none px-1"
                title="Delete this edge"
              >
                ×
              </button>
            )}
          </div>
        ))}
        {incoming.map((e) => (
          <div key={e.id} className={`flex items-center gap-2 text-xs group ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            <span className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{nodeMap[e.source]?.label ?? e.source}</span>
            <span className="text-indigo-400 font-mono">—{e.relation}→</span>
            <span className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{node.label}</span>
            {editMode && (
              <button
                onClick={() => onDeleteEdge?.(e.source, e.relation, e.target)}
                className="ml-auto opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-300 transition-opacity text-xs leading-none px-1"
                title="Delete this edge"
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Related evidence */}
      {evidence.length > 0 && (
        <div>
          <p className={`text-xs mb-1 font-semibold uppercase tracking-wider ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Related Sources
          </p>
          {evidence.slice(0, 2).map((chunk) => (
            <div
              key={chunk.id}
              className={`text-xs rounded p-2 mb-1.5 border ${
                isDark ? 'border-slate-700 bg-slate-800/50 text-slate-400' : 'border-slate-200 bg-slate-50 text-slate-500'
              }`}
            >
              <p className={`font-medium mb-0.5 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{chunk.source}</p>
              <p className="line-clamp-2">{chunk.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
