import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import type { EvidenceChunk, GraphData, GraphEdge, GraphNode, Message } from './types'
const GraphPanel = lazy(() => import('./components/GraphPanel'))
import MessageBubble from './components/MessageBubble'
import QueryInput, { type RoutingMode } from './components/QueryInput'
import NodeDetailPanel from './components/NodeDetailPanel'
import UploadModal from './components/UploadModal'
import PaperView from './components/PaperView'
import QuizPanel from './components/QuizPanel'
import { useTheme } from './ThemeContext'

type View = 'chat' | 'paper' | 'quiz'

const EMPTY_GRAPH: GraphData = { nodes: [], edges: [] }
const API_BASE = import.meta.env.VITE_API_URL ?? ''
const INGEST_JOB_KEY = 'lumin-ingest-job'



function generateId() {
  return Math.random().toString(36).slice(2)
}

async function realQuery(
  query: string,
  routingMode: RoutingMode,
  maxSources?: number,
  minRelevancy?: number,
): Promise<Omit<Message, 'id' | 'timestamp'>> {
  const body: Record<string, string | number> = { query }
  if (routingMode !== 'auto') body.routing_mode = routingMode
  if (maxSources !== undefined) body.max_sources = maxSources
  if (minRelevancy !== undefined && minRelevancy > 0) body.min_relevancy = minRelevancy
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Backend error ${res.status}: ${text}`)
  }
  return res.json()
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [graph, setGraph] = useState<GraphData>(EMPTY_GRAPH)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [selectedNodeEvidence, setSelectedNodeEvidence] = useState<EvidenceChunk[]>([])
  const [selectedNodeLoading, setSelectedNodeLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [maximized, setMaximized] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const [isGraphLoading, setIsGraphLoading] = useState(true)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [backendReady, setBackendReady] = useState(false)
  const [wakingUp, setWakingUp] = useState(false)
  const [editMode, setEditMode] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [view, setView] = useState<View>('chat')
  const [ingestJob, setIngestJob] = useState<{ id: string; stage: string } | null>(() =>
    localStorage.getItem(INGEST_JOB_KEY) ? { id: localStorage.getItem(INGEST_JOB_KEY)!, stage: 'Connecting…' } : null
  )
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const baseGraphRef = useRef<GraphData>(EMPTY_GRAPH)
  const nodeDetailRef = useRef<HTMLDivElement>(null)
  const ingestPollRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  // Poll /api/health until the backend responds, then load the graph
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const ping = async () => {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 10_000)
      try {
        const res = await fetch(`${API_BASE}/api/health`, {
          cache: 'no-store',
          signal: controller.signal,
        })
        if (res.ok && !cancelled) {
          setBackendReady(true)
          setWakingUp(false)
          return
        }
      } catch {
        // server not yet up (or timed out) — retry
      } finally {
        clearTimeout(timeout)
      }
      if (!cancelled) {
        setWakingUp(true)
        timer = setTimeout(ping, 3000)
      }
    }

    ping()
    return () => { cancelled = true; clearTimeout(timer) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadGraph = () => {
    setIsGraphLoading(true)
    fetch(`${API_BASE}/api/graph`)
      .then((res) => {
        if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`)
        return res.json() as Promise<GraphData>
      })
      .then((data) => {
        if (data.nodes.length > 0) {
          const unhighlighted: GraphData = {
            nodes: data.nodes.map((n) => ({ ...n, highlighted: false })),
            edges: data.edges.map((e) => ({ ...e, highlighted: false })),
          }
          baseGraphRef.current = unhighlighted
          setGraph(unhighlighted)
        }
      })
      .catch((err) => {
        console.warn('Could not load graph from backend:', err)
      })
      .finally(() => setIsGraphLoading(false))
  }

  const loadSuggestions = () => {
    fetch(`${API_BASE}/api/suggestions`)
      .then((res) => (res.ok ? res.json() as Promise<string[]> : Promise.resolve([])))
      .then(setSuggestions)
      .catch(() => {})
  }

  // Load the graph and suggestions once the backend is confirmed ready
  useEffect(() => {
    if (backendReady) {
      loadGraph()
      loadSuggestions()
    }
  }, [backendReady]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Background ingest-job polling — persisted via localStorage so it survives
  // tab closes and modal unmounts.
  // ---------------------------------------------------------------------------

  const startPolling = (jobId: string) => {
    fetch(`${API_BASE}/api/ingest/jobs/${jobId}`)
      .then((res) => {
        if (res.status === 404) {
          // Backend restarted — job gone; silently clear
          localStorage.removeItem(INGEST_JOB_KEY)
          setIngestJob(null)
          return null
        }
        return res.ok ? res.json() : Promise.reject(new Error(`${res.status}`))
      })
      .then((job) => {
        if (!job) return
        if (job.status === 'done') {
          localStorage.removeItem(INGEST_JOB_KEY)
          setIngestJob(null)
          loadGraph()
          loadSuggestions()
        } else if (job.status === 'failed') {
          localStorage.removeItem(INGEST_JOB_KEY)
          setIngestJob({ id: jobId, stage: `Failed: ${job.error ?? 'unknown error'}` })
          ingestPollRef.current = setTimeout(() => setIngestJob(null), 8000)
        } else {
          setIngestJob({ id: jobId, stage: job.progress_stage || 'Processing…' })
          ingestPollRef.current = setTimeout(() => startPolling(jobId), 3000)
        }
      })
      .catch(() => {
        // Network hiccup — retry
        ingestPollRef.current = setTimeout(() => startPolling(jobId), 5000)
      })
  }

  // Resume polling on mount if a job was in-flight when the tab was closed
  useEffect(() => {
    const saved = localStorage.getItem(INGEST_JOB_KEY)
    if (saved) startPolling(saved)
    return () => { if (ingestPollRef.current) clearTimeout(ingestPollRef.current) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Multi-tab sync: if another tab finishes/clears the job, stop polling here too
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === INGEST_JOB_KEY && e.newValue === null) {
        if (ingestPollRef.current) clearTimeout(ingestPollRef.current)
        setIngestJob(null)
        loadGraph()
        loadSuggestions()
      }
    }
    window.addEventListener('storage', handler)
    return () => window.removeEventListener('storage', handler)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleJobStarted = (jobId: string) => {
    if (ingestPollRef.current) clearTimeout(ingestPollRef.current)
    localStorage.setItem(INGEST_JOB_KEY, jobId)
    setIngestJob({ id: jobId, stage: 'Queued…' })
    setUploadOpen(false)
    startPolling(jobId)
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleNodeClick = (node: GraphNode) => {
    setSelectedNode((prev) => (prev?.id === node.id ? null : node))
  }

  // Fetch the rich detail (including the evidence trail joined from SQLite)
  // whenever the selected node changes.
  useEffect(() => {
    if (!selectedNode) {
      setSelectedNodeEvidence([])
      return
    }
    let cancelled = false
    setSelectedNodeLoading(true)
    fetch(`${API_BASE}/api/graph/node/${encodeURIComponent(selectedNode.id)}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res.statusText)))
      .then((data: { evidence?: EvidenceChunk[] }) => {
        if (!cancelled) setSelectedNodeEvidence(data.evidence ?? [])
      })
      .catch(() => { if (!cancelled) setSelectedNodeEvidence([]) })
      .finally(() => { if (!cancelled) setSelectedNodeLoading(false) })
    return () => { cancelled = true }
  }, [selectedNode])

  const handleFirstKeystroke = () => {
    setGraph((prev) => {
      if (!prev.nodes.some((n) => n.highlighted)) return prev
      return {
        nodes: prev.nodes.map((n) => ({ ...n, highlighted: false, hopIndex: undefined })),
        edges: prev.edges.map((e) => ({ ...e, highlighted: false })),
      }
    })
  }

  const handleQuery = async (query: string, routingMode: RoutingMode = 'auto', maxSources?: number, minRelevancy?: number) => {
    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)
    setGraph(baseGraphRef.current)

    try {
      const result = await realQuery(query, routingMode, maxSources, minRelevancy)
      const assistantMsg: Message = { id: generateId(), timestamp: new Date(), ...result }
      setMessages((prev) => [...prev, assistantMsg])
      if (result.hops && result.hops.length > 0) {
        const hopIndex = new Map(result.hops.map((id, i) => [id, i]))
        setGraph({
          nodes: baseGraphRef.current.nodes.map((n: GraphNode) => ({
            ...n,
            highlighted: hopIndex.has(n.id),
            hopIndex: hopIndex.get(n.id),
          })),
          edges: baseGraphRef.current.edges.map((e: GraphEdge) => ({
            ...e,
            highlighted: hopIndex.has(e.source) && hopIndex.has(e.target),
          })),
        })
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: 'assistant',
          content: 'Failed to get a response. Please check the backend is running.',
          timestamp: new Date(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleClearData = async () => {
    if (!window.confirm('This will permanently delete all chunks, the vector index, the concept graph, and all uploaded files. Continue?')) return
    setClearing(true)
    try {
      const res = await fetch(`${API_BASE}/api/data`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      setMessages([])
      setGraph(EMPTY_GRAPH)
      baseGraphRef.current = EMPTY_GRAPH
      setSelectedNode(null)
    } catch (e) {
      alert(`Clear failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setClearing(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Graph edit handlers
  // ---------------------------------------------------------------------------

  const handleAddNode = async (payload: {
    name: string
    node_type: 'topic' | 'subtopic' | 'content'
    summary?: string
    scope?: 'broad' | 'narrow'
    content_type?: string
    summary_beginner?: string
    summary_intermediate?: string
    summary_expert?: string
    parent_topic_keys?: string[]
    parent_subtopic_keys?: string[]
  }) => {
    try {
      const res = await fetch(`${API_BASE}/api/graph/node`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const text = await res.text()
        alert(`Add node failed: ${text}`)
        return
      }
      loadGraph()
    } catch (e) {
      alert(`Add node failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleDeleteNode = async (nodeId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/graph/node/${encodeURIComponent(nodeId)}`, { method: 'DELETE' })
      if (!res.ok) {
        const text = await res.text()
        alert(`Delete node failed: ${text}`)
        return
      }
      if (selectedNode?.id === nodeId) setSelectedNode(null)
      loadGraph()
    } catch (e) {
      alert(`Delete node failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleUpdateNode = async (nodeId: string, updates: Record<string, unknown>) => {
    try {
      const res = await fetch(`${API_BASE}/api/graph/node/${encodeURIComponent(nodeId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      })
      if (!res.ok) {
        const text = await res.text()
        alert(`Update node failed: ${text}`)
        return
      }
      loadGraph()
    } catch (e) {
      alert(`Update node failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleConnectNodes = async (
    sourceId: string,
    targetId: string,
    relation: string,
    label?: string,
    confidence?: number,
  ) => {
    try {
      const body: Record<string, unknown> = {
        source: sourceId,
        target: targetId,
        relation,
      }
      if (label) body.label = label
      if (confidence !== undefined) body.confidence = confidence
      const res = await fetch(`${API_BASE}/api/graph/edge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const text = await res.text()
        alert(`Add edge failed: ${text}`)
        return
      }
      loadGraph()
    } catch (e) {
      alert(`Add edge failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleDeleteEdge = async (
    source: string,
    relation: string,
    target: string,
    label?: string,
  ) => {
    try {
      const params = new URLSearchParams({ source, relation, target })
      if (label) params.set('label', label)
      const res = await fetch(`${API_BASE}/api/graph/edge?${params}`, { method: 'DELETE' })
      if (!res.ok) {
        const text = await res.text()
        alert(`Delete edge failed: ${text}`)
        return
      }
      loadGraph()
    } catch (e) {
      alert(`Delete edge failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return (
    <div className={`flex h-screen overflow-hidden transition-colors duration-200 ${
      isDark ? 'bg-slate-950 text-slate-200' : 'bg-[#f0f4ff] text-slate-800'
    }`}>
      {/* Sidebar — Graph Panel */}
      <aside
        className={`flex flex-col overflow-hidden ${
          view === 'paper' || view === 'quiz' ? 'hidden' : ''
        } ${isDark ? 'bg-slate-900' : 'bg-[#f8faff]'} ${fullscreen
          ? `fixed inset-0 z-50 w-screen ${isDark ? 'border-slate-800' : 'border-[#dde5f5]'}`
          : `border-r transition-all duration-300 shrink-0 ${isDark ? 'border-slate-800' : 'border-[#dde5f5]'} ${
              sidebarOpen ? (maximized ? 'w-[55vw]' : 'w-80 xl:w-96') : 'w-0'
            }`
        }`}
      >
        <div className={`flex-1 overflow-hidden flex flex-col ${
          fullscreen ? 'min-w-full' : maximized ? 'min-w-[55vw]' : 'min-w-80 xl:min-w-96'
        }`}>
          {isGraphLoading && (
            <div className={`flex items-center justify-center gap-2 px-3 py-1.5 text-xs border-b ${
              isDark
                ? 'text-slate-500 border-slate-800 bg-slate-900/40'
                : 'text-slate-400 border-[#dde5f5] bg-[#f0f4ff]/40'
            }`}>
              <div className={`w-3 h-3 border border-t-indigo-400 rounded-full animate-spin ${
                isDark ? 'border-slate-600' : 'border-slate-300'
              }`} />
              Loading graph…
            </div>
          )}
          <Suspense fallback={<div className="flex-1 flex items-center justify-center text-xs text-slate-500">Loading graph…</div>}>
            <GraphPanel
              graph={graph}
              onNodeClick={handleNodeClick}
              selectedNodeId={selectedNode?.id ?? null}
              maximized={maximized}
              onMaximize={() => setMaximized((v) => !v)}
              fullscreen={fullscreen}
              onFullscreen={() => setFullscreen((v) => !v)}
              editMode={editMode}
              onEditModeChange={setEditMode}
              onAddNode={handleAddNode}
              onDeleteNode={handleDeleteNode}
              onConnectNodes={handleConnectNodes}
              onViewDetails={(id) => {
                const node = graph.nodes.find((n) => n.id === id)
                if (node) {
                  setSelectedNode(node)
                  setTimeout(() => nodeDetailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
                }
              }}
            />
          </Suspense>
          {selectedNode && (
            <div ref={nodeDetailRef}>
            <NodeDetailPanel
              node={selectedNode}
              edges={graph.edges}
              allNodes={graph.nodes}
              evidence={selectedNodeEvidence}
              evidenceLoading={selectedNodeLoading}
              apiBase={API_BASE}
              onClose={() => setSelectedNode(null)}
              editMode={editMode}
              onDeleteNode={handleDeleteNode}
              onDeleteEdge={handleDeleteEdge}
              onUpdateNode={handleUpdateNode}
            />
            </div>
          )}
        </div>
      </aside>

      {/* Main — Chat */}
      <main className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className={`flex items-center gap-3 px-4 py-3 border-b backdrop-blur shrink-0 ${
          isDark
            ? 'border-slate-800 bg-slate-900/60'
            : 'border-[#dde5f5] bg-white/80'
        }`}>
          {view === 'chat' && (
            <button
              onClick={() => setSidebarOpen((v) => !v)}
              className={`transition-colors p-1 rounded ${
                isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'
              }`}
              title={sidebarOpen ? 'Hide graph' : 'Show graph'}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                <rect x="3" y="3" width="7" height="18" rx="1" />
                <path d="M14 3h7M14 12h7M14 21h7" strokeLinecap="round" />
              </svg>
            </button>
          )}

          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold">
              L
            </div>
            <div>
              <span className={`font-semibold text-sm ${isDark ? 'text-white' : 'text-slate-900'}`}>LuminRAG</span>
              <span className={`text-xs ml-2 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Educational QA</span>
            </div>
          </div>

          <nav className={`ml-4 flex items-center gap-1 rounded-lg border p-0.5 ${
            isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-300 bg-white'
          }`}>
            <button
              onClick={() => setView('chat')}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                view === 'chat'
                  ? (isDark ? 'bg-indigo-600 text-white' : 'bg-indigo-500 text-white')
                  : (isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-600 hover:text-slate-800')
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setView('paper')}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                view === 'paper'
                  ? (isDark ? 'bg-indigo-600 text-white' : 'bg-indigo-500 text-white')
                  : (isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-600 hover:text-slate-800')
              }`}
            >
              Paper
            </button>
            <button
              onClick={() => setView('quiz')}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                view === 'quiz'
                  ? (isDark ? 'bg-indigo-600 text-white' : 'bg-indigo-500 text-white')
                  : (isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-600 hover:text-slate-800')
              }`}
            >
              Quiz
            </button>
          </nav>

          <div className="ml-auto flex items-center gap-2">
            {ingestJob && (
              <span className={`text-xs flex items-center gap-1.5 ${
                ingestJob.stage.startsWith('Failed')
                  ? (isDark ? 'text-red-400' : 'text-red-500')
                  : (isDark ? 'text-indigo-400' : 'text-indigo-600')
              }`}>
                {!ingestJob.stage.startsWith('Failed') && (
                  <div className={`w-3 h-3 border border-t-indigo-400 rounded-full animate-spin ${isDark ? 'border-slate-600' : 'border-slate-300'}`} />
                )}
                {ingestJob.stage}
              </span>
            )}
            {view === 'chat' && !ingestJob && (
              <span className={`text-xs ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
                {graph.nodes.filter((n) => n.highlighted).length > 0
                  ? `${graph.nodes.filter((n) => n.highlighted).length} nodes highlighted`
                  : 'No active trace'}
              </span>
            )}

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className={`w-8 h-8 rounded-lg flex items-center justify-center transition-colors ${
                isDark
                  ? 'text-slate-400 hover:text-slate-200 border border-slate-700 hover:border-slate-600'
                  : 'text-slate-500 hover:text-slate-700 border border-slate-300 hover:border-slate-400 bg-white'
              }`}
              title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDark ? (
                /* Sun icon */
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" strokeLinecap="round" />
                </svg>
              ) : (
                /* Moon icon */
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                  <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </button>

            {view === 'chat' && (<>
            <button
              onClick={handleClearData}
              disabled={clearing}
              className={`text-xs border px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed ${
                isDark
                  ? 'text-red-500 hover:text-red-300 border-red-900 hover:border-red-700'
                  : 'text-red-500 hover:text-red-600 border-red-200 hover:border-red-400 bg-red-50 hover:bg-red-100'
              }`}
              title="Clear all data"
            >
              {clearing ? (
                <div className="w-3.5 h-3.5 border border-red-500/30 border-t-red-400 rounded-full animate-spin" />
              ) : (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                  <polyline points="3 6 5 6 21 6" strokeLinecap="round" />
                  <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" strokeLinecap="round" />
                  <path d="M10 11v6M14 11v6" strokeLinecap="round" />
                  <path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2" strokeLinecap="round" />
                </svg>
              )}
              Clear Data
            </button>
            <button
              onClick={() => setUploadOpen(true)}
              className={`text-xs border px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 ${
                isDark
                  ? 'text-slate-400 hover:text-slate-200 border-slate-700 hover:border-indigo-600'
                  : 'text-slate-600 hover:text-slate-800 border-slate-300 hover:border-indigo-400 bg-white hover:bg-indigo-50'
              }`}
              title="Upload course content"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Upload
            </button>
            <button
              onClick={() => { setMessages([]); setGraph(baseGraphRef.current); setSelectedNode(null) }}
              className={`text-xs border px-2.5 py-1 rounded-lg transition-colors ${
                isDark
                  ? 'text-slate-500 hover:text-slate-300 border-slate-700 hover:border-slate-600'
                  : 'text-slate-500 hover:text-slate-700 border-slate-300 hover:border-slate-400 bg-white hover:bg-slate-50'
              }`}
            >
              Clear
            </button>
            </>)}
          </div>
        </header>

        {view === 'paper' ? (
          <PaperView />
        ) : view === 'quiz' ? (
          <QuizPanel isDark={isDark} />
        ) : (
        <>
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-3xl font-bold shadow-lg">
                L
              </div>
              <div>
                <h1 className={`text-xl font-semibold mb-1 ${isDark ? 'text-white' : 'text-slate-900'}`}>LuminRAG</h1>
                <p className={`text-sm max-w-sm ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
                  Ask a question about your course material. The system will retrieve relevant knowledge,
                  trace its reasoning through the concept graph, and verify its answer.
                </p>
              </div>
              {suggestions.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg mt-2">
                  {suggestions.slice(0, 4).map((q) => (
                    <button
                      key={q}
                      onClick={() => handleQuery(q)}
                      className={`text-left text-xs border rounded-xl p-3 transition-all ${
                        isDark
                          ? 'text-slate-400 border-slate-700 hover:border-indigo-600 hover:text-slate-200 bg-slate-800/50 hover:bg-slate-800'
                          : 'text-slate-600 border-slate-200 hover:border-indigo-300 hover:text-slate-800 bg-white hover:bg-indigo-50/50 shadow-sm'
                      }`}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {loading && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold shrink-0">
                L
              </div>
              <div className={`rounded-2xl rounded-tl-sm border px-4 py-3 ${
                isDark ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200 shadow-sm'
              }`}>
                <div className="flex gap-1.5 items-center h-5">
                  <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className={`px-4 pb-4 pt-2 border-t backdrop-blur shrink-0 ${
          isDark ? 'border-slate-800 bg-slate-900/60' : 'border-[#dde5f5] bg-white/80'
        }`}>
          <div className="max-w-3xl mx-auto">
            <QueryInput onSubmit={handleQuery} loading={loading} onFirstKeystroke={handleFirstKeystroke} suggestions={suggestions} />
          </div>
        </div>
        </>
        )}
      </main>

      {uploadOpen && (
        <UploadModal
          onClose={() => setUploadOpen(false)}
          onJobStarted={handleJobStarted}
        />
      )}

      {wakingUp && (
        <div className={`fixed inset-0 z-50 flex flex-col items-center justify-center backdrop-blur gap-4 ${
          isDark ? 'bg-slate-950/90' : 'bg-[#f0f4ff]/90'
        }`}>
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-2xl font-bold shadow-lg">
            L
          </div>
          <div className={`flex items-center gap-2 text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
            <div className={`w-4 h-4 border-2 border-t-indigo-400 rounded-full animate-spin ${
              isDark ? 'border-slate-600' : 'border-slate-300'
            }`} />
            Waking up the server…
          </div>
          <p className={`text-xs ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
            This can take up to 60 seconds on first load.
          </p>
        </div>
      )}
    </div>
  )
}
