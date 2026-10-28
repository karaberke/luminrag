import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type Sigma from 'sigma'
import type { GraphData, GraphNode, GraphOp, NodeType } from '../types'
import { useTheme } from '../theme'
import { activeGraphTab, collapseGraph, duplicateSuggestionKey, graphTraceKey, hiddenGraphNodes, indexGraph, validDuplicateSuggestions, type DuplicateCandidate, type GraphTabState } from '../utils/graph'
import { apiJson, errorMessage } from '../utils/api'
import { fitGraphBounds } from '../utils/graphViewport'
import ContextMenu from './graph/ContextMenu'
import type { CanvasMenu } from './graph/GraphCanvas'
import { applyGraphOps, redoGraphEdit, undoGraphEdit, useGraphStore } from './graph/useGraphStore'
const GraphCanvas = lazy(() => import('./graph/GraphCanvas'))

interface Props {
  graph: GraphData; onNodeClick: (node: GraphNode) => void; selectedNodeId: string | null
  maximized?: boolean; onMaximize?: () => void; fullscreen?: boolean; onFullscreen?: () => void
  editMode?: boolean; onEditModeChange?: (value: boolean) => void
  onAddNode?: (payload: { name: string; node_type: NodeType; body?: string; x?: number; y?: number }) => void
  onDeleteNode?: (id: string) => void
  onConnectNodes?: (source: string, target: string, relation: string) => void
  onViewDetails?: (id: string) => void; active?: boolean
}
const CONNECTION_TYPES = [
  { value: 'RELATED_TO', title: 'Related concepts' },
  { value: 'HAS_SUBTOPIC', title: 'Parent → subtopic' },
  { value: 'HAS_CONTENT', title: 'Parent → content' },
]
const NODE_TYPES: NodeType[] = ['topic', 'subtopic', 'content', 'note']

export default function GraphPanel({ graph, onNodeClick, selectedNodeId, maximized, onMaximize, fullscreen,
  onFullscreen, editMode = false, onEditModeChange, onAddNode, onConnectNodes, onViewDetails, active = true }: Props) {
  const { theme } = useTheme()
  const dark = theme === 'dark'
  const [tabState, setTab] = useState<GraphTabState>({ value: 'full', trace: '' })
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [expandedAll, setExpandedAll] = useState(false)
  const [neighborMode, setNeighborMode] = useState(false)
  const [spacing, setSpacing] = useState(1)
  const [committedSpacing, setCommittedSpacing] = useState(1)
  const [layoutRevision, setLayoutRevision] = useState(0)
  const [menu, setMenu] = useState<CanvasMenu | null>(null)
  const [edge, setEdge] = useState<{ source: string; target: string } | null>(null)
  const [relation, setRelation] = useState('RELATED_TO')
  const [connectFrom, setConnectFrom] = useState<string | null>(null)
  const [addForm, setAddForm] = useState(false)
  const [newName, setNewName] = useState('')
  const [newType, setNewType] = useState<NodeType>('note')
  const [duplicates, setDuplicates] = useState<DuplicateCandidate[] | null>(null)
  const [duplicatesResultKey, setDuplicatesResultKey] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [promptDialog, setPromptDialog] = useState<{ title: string; value: string; submit: (value: string) => void } | null>(null)
  const rendererRef = useRef<Sigma | null>(null)
  const duplicatesRequest = useRef<AbortController | null>(null)
  const selectedIds = useGraphStore((state) => state.selectedIds)
  const busy = useGraphStore((state) => state.busy)
  const stepCount = useGraphStore((state) => state.steps.length)
  const highlightedKey = graphTraceKey(graph)
  const tab = activeGraphTab(graph, tabState)
  const hasGraphTrace = graph.nodes.some((node) => node.highlighted)
  const index = useMemo(() => indexGraph(graph), [graph])
  const duplicatesOpen = duplicates !== null
  const duplicatesKey = useMemo(() => duplicatesOpen ? duplicateSuggestionKey(graph.nodes) : '', [duplicatesOpen, graph.nodes])
  const duplicatesLoading = duplicatesOpen && duplicatesResultKey !== duplicatesKey
  const currentDuplicates = useMemo(() => validDuplicateSuggestions(duplicates ?? [], index.nodes), [duplicates, index.nodes])
  const automaticCollapsed = useMemo(() => graph.nodes.length > 300 && !expandedAll
    ? new Set(graph.nodes.filter((node) => node.node_type === 'topic').map((node) => node.id)) : new Set<string>(), [graph.nodes, expandedAll])
  const effectiveCollapsed = useMemo(() => new Set([...automaticCollapsed, ...collapsed]), [automaticCollapsed, collapsed])
  const visible = useMemo(() => collapseGraph(graph, index, effectiveCollapsed), [graph, index, effectiveCollapsed])
  const hiddenIds = useMemo(() => {
    const shown = new Set(visible.filteredNodes.map((node) => node.id))
    return hiddenGraphNodes(graph, shown, tab)
  }, [graph, visible, tab])
  const run = useCallback(async (operation: () => Promise<void>) => {
    setError(null)
    try { await operation() } catch (cause) { setError(errorMessage(cause)) }
  }, [])
  const mutate = (ops: GraphOp[]) => { run(() => applyGraphOps(ops)); setMenu(null) }
  const beginConnect = useCallback((source: string, target: string) => { setEdge({ source, target }); setRelation('RELATED_TO') }, [])
  const openContext = useCallback((value: CanvasMenu) => {
    const size = rendererRef.current?.getDimensions() ?? { width: 300, height: 400 }
    setMenu({ ...value, left: Math.min(value.left, size.width - 180), top: Math.min(value.top, size.height - 260) })
  }, [])
  const clickNode = useCallback((node: GraphNode) => {
    setMenu(null)
    if (connectFrom && connectFrom !== node.id) { beginConnect(connectFrom, node.id); setConnectFrom(null) }
    else onNodeClick(node)
  }, [connectFrom, beginConnect, onNodeClick])
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (!active || !editMode || !(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'z') return
      if ((event.target as HTMLElement)?.closest('input, textarea, [contenteditable="true"]')) return
      event.preventDefault()
      run(event.shiftKey ? redoGraphEdit : undoGraphEdit)
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  }, [active, editMode, run])
  const button = `rounded-lg border px-2 py-1 text-[11px] transition-colors disabled:opacity-40 ${dark ? 'border-slate-700 hover:border-indigo-500 text-slate-400' : 'border-slate-300 hover:border-indigo-400 text-slate-600'}`
  const menuButton = 'block w-full text-left px-3 py-1.5 rounded-md hover:bg-indigo-500/15'
  const menuNode = menu?.node ? index.nodes.get(menu.node) : undefined
  const ask = (title: string, value: string, submit: (value: string) => void) => { setMenu(null); setPromptDialog({ title, value, submit }) }
  const toggleCollapse = (id: string) => {
    if (automaticCollapsed.has(id)) { setExpandedAll(true); setCollapsed(new Set([...automaticCollapsed].filter((key) => key !== id))) }
    else setCollapsed((previous) => { const next = new Set(previous); if (next.has(id)) next.delete(id); else next.add(id); return next })
  }
  const newNote = (linked?: GraphNode) => {
    const position = menu
    ask('Note title', 'New note', (name) => {
      const node = { id: crypto.randomUUID(), name, title: name, node_type: 'note', body: linked ? `[[${linked.name}]]` : '', x: position?.x, y: position?.y }
      mutate([{ op: 'add_node', node }])
    })
  }
  const refreshDuplicates = useCallback((key: string) => {
    duplicatesRequest.current?.abort()
    const controller = new AbortController()
    duplicatesRequest.current = controller
    apiJson<DuplicateCandidate[] | { pairs?: DuplicateCandidate[]; duplicates?: DuplicateCandidate[] }>('/api/graph/duplicates', { signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) { setDuplicates(Array.isArray(result) ? result : result.pairs ?? result.duplicates ?? []); setDuplicatesResultKey(key); setError(null) } })
      .catch((cause) => { if (!controller.signal.aborted) { setDuplicates([]); setError(errorMessage(cause)); setDuplicatesResultKey(key) } })
  }, [])
  useEffect(() => {
    if (!duplicatesOpen || !active) return
    refreshDuplicates(duplicatesKey)
    return () => duplicatesRequest.current?.abort()
  }, [duplicatesOpen, duplicatesKey, active, refreshDuplicates])
  const showDuplicates = () => { setDuplicatesResultKey(''); if (duplicatesOpen) refreshDuplicates(duplicatesKey); else setDuplicates([]) }
  const savePng = () => {
    const renderer = rendererRef.current
    if (!renderer) return
    requestAnimationFrame(() => {
      if (rendererRef.current !== renderer) return
      // WebGL's drawing buffer is cleared between frames; capture directly after drawing.
      renderer.refresh()
      const canvas = renderer.getContainer().querySelector('canvas')
      canvas?.toBlob((blob) => {
        if (!blob) return
        const url = URL.createObjectURL(blob), link = document.createElement('a')
        link.href = url; link.download = 'luminrag-graph.png'; link.click()
        setTimeout(() => URL.revokeObjectURL(url), 1000)
      })
    })
  }
  return <div className="lumin-graph-panel flex flex-col h-full min-h-64 overflow-hidden">
    <div className={`lumin-graph-tabs flex items-center gap-1 p-2 border-b ${dark ? 'border-slate-800' : 'border-slate-200'}`}>
      {(['full', 'subgraph'] as const).map((value) => <button key={value} className={`${button} ${tab === value ? 'bg-indigo-500/15 border-indigo-500' : ''}`} onClick={() => setTab({ value, trace: highlightedKey })}>{value === 'full' ? 'Full Graph' : 'Subgraph'}</button>)}
      <button className={`${button} ml-auto`} onClick={onMaximize} title={maximized ? 'Shrink graph' : 'Widen graph'}>↔</button>
      <button className={button} onClick={onFullscreen} title={fullscreen ? 'Exit full screen' : 'Full screen'}>⛶</button>
      <button className={button} onClick={savePng} title="Export graph image">PNG</button>
    </div>
    <div className="lumin-graph-controls flex flex-wrap items-center gap-1.5 p-2 border-b border-slate-500/15">
      <button className={`${button} ${editMode ? 'bg-indigo-500/15' : ''}`} onClick={() => onEditModeChange?.(!editMode)}>{editMode ? 'Editing on' : 'Edit graph'}</button>
      <button className={button} onClick={() => setNeighborMode(!neighborMode)} disabled={!selectedNodeId}>{neighborMode ? 'All nodes' : 'Neighbors'}</button>
      <button className={button} onClick={() => { if (effectiveCollapsed.size) { setExpandedAll(true); setCollapsed(new Set()) } else setCollapsed(new Set(graph.nodes.filter((node) => node.node_type === 'topic').map((node) => node.id))) }}>{effectiveCollapsed.size ? 'Expand topics' : 'Collapse topics'}</button>
      <button className={button} onClick={() => { if (rendererRef.current) fitGraphBounds(rendererRef.current, { preserveScale: true }) }}>Fit</button>
      {editMode && <><button className={button} onClick={() => setAddForm(!addForm)}>+ Node</button><button className={button} disabled={busy || stepCount === 0} onClick={() => run(undoGraphEdit)}>Undo</button><button className={button} disabled={busy} onClick={() => run(redoGraphEdit)}>Redo</button><button className={button} onClick={showDuplicates}>Duplicates</button></>}
      <label className="flex items-center gap-1 text-[10px] text-slate-500">Spacing <input aria-label="Graph spacing" type="range" min="0.3" max="3" step="0.1" value={spacing} onChange={(event) => setSpacing(Number(event.target.value))} onPointerUp={() => { setCommittedSpacing(spacing); setLayoutRevision((value) => value + 1) }} onKeyUp={() => { setCommittedSpacing(spacing); setLayoutRevision((value) => value + 1) }} className="w-16 accent-indigo-500" /></label>
    </div>
    {error && <p role="alert" className="text-xs text-red-500 p-2">{error}</p>}
    {connectFrom && <p className="text-xs text-indigo-400 p-2">Select the target node. <button onClick={() => setConnectFrom(null)}>Cancel</button></p>}
    {editMode && <p className="px-2 py-1 text-[10px] text-slate-500">Drag to move · Shift+drag to connect · Ctrl+drag to select · right-click for actions</p>}
    {selectedIds.length > 0 && editMode && <div className="flex gap-2 items-center px-2 py-1.5 text-xs bg-indigo-500/10"><span>{selectedIds.length} selected</span><button className={button} onClick={() => { mutate(selectedIds.map((id) => ({ op: 'delete_node', id }))); useGraphStore.getState().setSelection([]) }}>Delete</button><button className={button} disabled={selectedIds.length < 2} onClick={() => { const keep = selectedIds[0]; mutate(selectedIds.slice(1).map((drop) => ({ op: 'merge_nodes', keep, drop }))); useGraphStore.getState().setSelection([keep]) }}>Merge</button><button className={button} onClick={() => useGraphStore.getState().setSelection([])}>Clear</button></div>}
    {editMode && addForm && <form className="p-2 flex flex-wrap gap-2" onSubmit={(event) => { event.preventDefault(); if (!newName.trim()) return; onAddNode?.({ name: newName.trim(), node_type: newType, ...(newType === 'note' ? { body: '' } : {}) }); setNewName(''); setAddForm(false) }}><input autoFocus aria-label="Node name" placeholder="Node title" value={newName} onChange={(event) => setNewName(event.target.value)} className="border border-slate-500/30 bg-transparent rounded-lg text-xs p-1.5 flex-1 min-w-20" /><select aria-label="Node type" value={newType} onChange={(event) => setNewType(event.target.value as NodeType)} className="bg-transparent text-xs">{NODE_TYPES.map((type) => <option key={type} className="text-slate-900">{type}</option>)}</select><button className={button}>Add</button></form>}
    <p className="px-3 py-1 text-[10px] text-slate-500">Zoom in for more names · select a node for details.</p>
    <div className="relative flex-1 min-h-48" onClick={() => { if (menu) setMenu(null) }}>
      {tab === 'subgraph' && !hasGraphTrace && graph.nodes.length > 0 && <div role="status" className="absolute inset-0 z-10 flex items-center justify-center text-center text-xs text-slate-500 p-6 pointer-events-none">No graph evidence for this answer. Choose Graph or Hybrid to retrieve a subgraph.</div>}
      {graph.nodes.length === 0 ? <div className="lumin-graph-empty absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-slate-500"><span>Upload material or add your first note.</span>{editMode && <button className={button} onClick={() => newNote()}>New note</button>}</div> : <Suspense fallback={<div className="p-4 text-xs text-slate-500">Loading graph…</div>}><GraphCanvas data={graph} dark={dark} editMode={editMode} selectedNodeId={selectedNodeId} hiddenIds={hiddenIds} neighborMode={neighborMode} spacing={committedSpacing} layoutRevision={layoutRevision} active={active} onNodeClick={clickNode} onContext={openContext} onConnect={beginConnect} rendererRef={rendererRef} /></Suspense>}
      {menu && <ContextMenu left={menu.left} top={menu.top} dark={dark}>
        {menuNode ? <>
          <button role="menuitem" className={menuButton} onClick={() => { onViewDetails?.(menuNode.id); setMenu(null) }}>View / edit content</button>
          {menuNode.node_type === 'topic' && <button role="menuitem" className={menuButton} onClick={() => toggleCollapse(menuNode.id)}>{effectiveCollapsed.has(menuNode.id) ? 'Expand' : 'Collapse'}</button>}
          {editMode && <><button role="menuitem" className={menuButton} onClick={() => ask('Node name', menuNode.name, (name) => mutate([{ op: 'update_node', id: menuNode.id, updates: { name } }]))}>Rename</button>
          <label className={`${menuButton} flex gap-2 items-center`}>Change type<select value={menuNode.node_type} className="bg-transparent" onChange={(event) => mutate([{ op: 'update_node', id: menuNode.id, updates: { node_type: event.target.value } }])}>{NODE_TYPES.map((type) => <option key={type} className="text-slate-900">{type}</option>)}</select></label>
          <button role="menuitem" className={menuButton} onClick={() => ask('Child title', '', (name) => { const id = crypto.randomUUID(), type = menuNode.node_type === 'topic' ? 'subtopic' : 'content'; mutate([{ op: 'add_node', node: { id, name, node_type: type } }, { op: 'add_edge', edge: { source: menuNode.id, target: id, relation: type === 'subtopic' ? 'HAS_SUBTOPIC' : 'HAS_CONTENT' } }]) })}>Add child</button>
          <button role="menuitem" className={menuButton} onClick={() => { setConnectFrom(menuNode.id); setMenu(null) }}>Connect to…</button><button role="menuitem" className={menuButton} onClick={() => newNote(menuNode)}>Add linked note</button>
          <button role="menuitem" className={menuButton} onClick={() => ask('Merge into node (name or ID)', '', (target) => { const keep = graph.nodes.find((node) => node.id === target || node.name.toLowerCase() === target.toLowerCase()); if (keep && keep.id !== menuNode.id) mutate([{ op: 'merge_nodes', keep: keep.id, drop: menuNode.id }]); else setError('Choose an existing, different node.') })}>Merge with…</button>
          <button role="menuitem" className={`${menuButton} text-red-500`} onClick={() => mutate([{ op: 'delete_node', id: menuNode.id }])}>Delete</button></>}
        </> : editMode ? <button role="menuitem" className={menuButton} onClick={() => newNote()}>New note here</button> : <span className="block p-2 text-slate-500">Enable editing to add notes.</span>}
      </ContextMenu>}
      {duplicates !== null && <div className={`absolute inset-y-0 right-0 w-72 max-w-full z-20 border-l p-3 overflow-auto text-xs ${dark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'}`}><div className="flex justify-between mb-3"><strong>Possible duplicates</strong><button aria-label="Close duplicate suggestions" onClick={() => setDuplicates(null)}>×</button></div>{duplicatesLoading ? <p role="status" className="text-slate-500">Updating duplicate suggestions…</p> : currentDuplicates.length === 0 && <p className="text-slate-500">No duplicate suggestions.</p>}{currentDuplicates.map((pair) => <div key={`${pair.keep}:${pair.drop}`} className="mb-3 rounded-lg border border-slate-500/20 p-2"><p>{index.nodes.get(pair.keep)!.name} ↔ {index.nodes.get(pair.drop)!.name}</p><p className="text-slate-500 mt-1">{Math.round(pair.similarity * 100)}% similarity</p><button className={`${button} mt-2`} disabled={!editMode || busy || duplicatesLoading} onClick={() => { run(async () => { await applyGraphOps([{ op: 'merge_nodes', keep: pair.keep, drop: pair.drop }]); setDuplicates((previous) => previous?.filter((candidate) => candidate.keep !== pair.drop && candidate.drop !== pair.drop) ?? null) }) }}>Merge</button></div>)}</div>}
      {editMode && edge && <form className={`absolute left-3 right-3 bottom-3 z-30 rounded-xl border p-3 text-xs shadow-xl space-y-2 ${dark ? 'bg-slate-900 border-slate-600' : 'bg-white border-slate-200'}`} onSubmit={(event) => { event.preventDefault(); onConnectNodes?.(edge.source, edge.target, relation); setEdge(null) }}><p>{index.nodes.get(edge.source)?.name} {relation === 'RELATED_TO' ? '—' : '→'} {index.nodes.get(edge.target)?.name}</p><select aria-label="Connection type" value={relation} onChange={(event) => setRelation(event.target.value)} className="bg-transparent w-full border border-slate-500/30 rounded p-1">{CONNECTION_TYPES.map(({ value, title }) => <option key={value} value={value} className="text-slate-900">{title}</option>)}</select><div className="flex gap-2"><button className={button}>Connect</button><button className={button} type="button" onClick={() => setEdge(null)}>Cancel</button></div></form>}
    </div>
    {editMode && promptDialog && <div className="fixed inset-0 z-[80] bg-black/50 flex items-center justify-center p-4" onClick={() => setPromptDialog(null)}><form role="dialog" aria-modal="true" aria-labelledby="graph-dialog-title" onClick={(event) => event.stopPropagation()} onSubmit={(event) => { event.preventDefault(); const value = promptDialog.value.trim(); if (!value) return; promptDialog.submit(value); setPromptDialog(null) }} className={`w-full max-w-sm rounded-xl border p-5 shadow-xl space-y-3 ${dark ? 'bg-slate-900 border-slate-600' : 'bg-white border-slate-200'}`}><h3 id="graph-dialog-title" className="font-semibold text-sm">{promptDialog.title}</h3><input autoFocus aria-label={promptDialog.title} value={promptDialog.value} onChange={(event) => setPromptDialog({ ...promptDialog, value: event.target.value })} className="w-full rounded-lg border border-slate-500/40 bg-transparent p-2 text-sm" /><div className="flex justify-end gap-2"><button type="button" className={button} onClick={() => setPromptDialog(null)}>Cancel</button><button type="submit" className={`${button} bg-indigo-500/20`} disabled={!promptDialog.value.trim()}>Save</button></div></form></div>}
    <div className="px-2 py-1 text-[10px] text-slate-500 border-t border-slate-500/15 flex gap-3"><span>{graph.nodes.length - hiddenIds.size} nodes · {graph.edges.length} edges</span><span className="text-amber-500">■ Notes</span><span className="ml-auto">{busy ? 'Saving…' : 'WebGL 2'}</span></div>
  </div>
}
