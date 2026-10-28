import BrandMark from './components/BrandMark'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { liveQuery } from 'dexie'
import type { EvidenceChunk, GraphData, GraphNode, Message, NodeType } from './types'
const GraphPanel = lazy(() => import('./components/GraphPanel'))
const SourceViewer = lazy(() => import('./components/source/SourceViewer'))
import MessageBubble from './components/MessageBubble'
import QueryInput from './components/QueryInput'
import NodeDetailPanel from './components/NodeDetailPanel'
import UploadModal from './components/UploadModal'
import AccessGate from './components/AccessGate'
import SettingsModal from './components/SettingsModal'
import WorkspaceToolbar from './components/WorkspaceToolbar'
import { useTheme } from './theme'
import {
  ApiError,
  apiFetch,
  apiJson,
  apiRequest,
  errorMessage,
  isAbortError,
  isAuthError,
} from './utils/api'
import { endSession, fetchAuthStatus, observeSessionStatus } from './utils/auth'
import { applyNodeUpdates, evidenceForNode, highlightGraphTrace, type NodeEvidenceState } from './utils/graph'
import { activeWorkspaceId, clearWorkspaceData, db, pendingIngestJobs, preloadSources, settleIngestOriginals, withOriginalsLock } from './storage/db'
import { restoreSelectedWorkspace, scheduleWorkspaceSave } from './utils/workspaces'
import { finishOpenRouterConnection, getLlmSettings, loadLlmSettings } from './utils/settings'
import { applyGraphOps, resetGraphHistory } from './components/graph/useGraphStore'
import { DEFAULT_QUERY_SETTINGS, queryWorkspace, type QuerySettings } from './utils/query'
import { ASSOCIATION_VERSION, normalizeGraphConnections } from './utils/connections'

/** connecting: waking server / checking status · locked: access key needed · error: status check failed */
type AuthPhase = 'connecting' | 'locked' | 'error' | 'ready'

interface IngestJobState {
  id: string
  stage: string
  failed?: boolean
}

interface AddNodePayload {
  name: string
  node_type: NodeType
  body?: string
  x?: number
  y?: number
  summary?: string
  scope?: 'broad' | 'narrow'
  content_type?: string
  summary_beginner?: string
  summary_intermediate?: string
  summary_expert?: string
  parent_topic_keys?: string[]
  parent_subtopic_keys?: string[]
}

const EMPTY_GRAPH: GraphData = { nodes: [], edges: [], association_version: ASSOCIATION_VERSION }
const INGEST_JOB_KEY = 'lumin-ingest-job'
const ingestJobKey = () => `${INGEST_JOB_KEY}:${activeWorkspaceId()}`
const HEALTH_TIMEOUT_MS = 10_000
const HEALTH_RETRY_MS = 3000

function generateId() {
  return Math.random().toString(36).slice(2)
}

function unhighlighted(data: GraphData): GraphData {
  return {
    ...data,
    nodes: data.nodes.map((n) => ({ ...n, highlighted: false })),
    edges: data.edges.map((e) => ({ ...e, highlighted: false })),
  }
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [workspaceId, setWorkspaceId] = useState(activeWorkspaceId)
  const [messagesLoadedFor, setMessagesLoadedFor] = useState<string | null>(null)
  const [hosted, setHosted] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [activeSource, setActiveSource] = useState<EvidenceChunk | null>(null)
  const [providerReady, setProviderReady] = useState(false)
  const [storageError, setStorageError] = useState<string | null>(null)
  const [workspaceLoading, setWorkspaceLoading] = useState(true)
  const [graph, setGraph] = useState<GraphData>(EMPTY_GRAPH)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [evidenceVersion, setEvidenceVersion] = useState(0)
  const [evidenceState, setEvidenceState] = useState<NodeEvidenceState | null>(null)
  const [loading, setLoading] = useState(false)
  const [querySettings, setQuerySettings] = useState<QuerySettings>({ ...DEFAULT_QUERY_SETTINGS })
  const [sidebarOpen, setSidebarOpen] = useState(() => typeof window === 'undefined' || window.innerWidth >= 768)
  const [maximized, setMaximized] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const [isGraphLoading, setIsGraphLoading] = useState(true)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [clearDataOpen, setClearDataOpen] = useState(false)
  const [authPhase, setAuthPhase] = useState<AuthPhase>('connecting')
  const [authError, setAuthError] = useState<string | null>(null)
  const [tokenRequired, setTokenRequired] = useState(false)
  const [connectAttempt, setConnectAttempt] = useState(0)
  const [wakingUp, setWakingUp] = useState(false)
  const [editMode, setEditMode] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [ingestJob, setIngestJob] = useState<IngestJobState | null>(() => {
    const id = localStorage.getItem(ingestJobKey())
    return id ? { id, stage: 'Connecting…' } : null
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const baseGraphRef = useRef<GraphData>(EMPTY_GRAPH)
  const nodeDetailRef = useRef<HTMLDivElement>(null)
  const graphLoadRef = useRef<AbortController | null>(null)
  const suggestionsLoadRef = useRef<AbortController | null>(null)
  const queryAbortRef = useRef<AbortController | null>(null)
  const previousWorkspaceRef = useRef<string | null>(null)
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'
  const canConnect = authPhase === 'ready' && (!hosted || providerReady)
  const isReady = canConnect && !workspaceLoading && !clearing

  const resumeIngestJobs = useCallback(async (ownerWorkspace: string, failure?: IngestJobState) => {
    const pending = await pendingIngestJobs(ownerWorkspace)
    const legacy = localStorage.getItem(`${INGEST_JOB_KEY}:${ownerWorkspace}`)
    const ids = legacy && !pending.includes(legacy) ? [...pending, legacy] : pending
    if (activeWorkspaceId() !== ownerWorkspace) return
    setIngestJob((current) => {
      const next = current && !current.failed && ids.includes(current.id) ? current.id : ids[0]
      return next ? current?.id === next && !current.failed ? current : { id: next, stage: 'Connecting…' } : failure ?? (current?.failed ? current : null)
    })
  }, [])

  const finishIngestJob = useCallback(async (ownerWorkspace: string, jobId: string, failure?: IngestJobState) => {
    const key = `${INGEST_JOB_KEY}:${ownerWorkspace}`
    if (localStorage.getItem(key) === jobId) localStorage.removeItem(key)
    await resumeIngestJobs(ownerWorkspace, failure)
  }, [resumeIngestJobs])

  useEffect(() => {
    let disposed = false
    loadLlmSettings().then(() => finishOpenRouterConnection()).then(() => {
      if (!disposed) { setProviderReady(!!getLlmSettings().key); setWorkspaceId(activeWorkspaceId()) }
    }).catch((error) => { if (!disposed) { setStorageError(errorMessage(error)); setSettingsOpen(true) } })
    const settings = () => { setProviderReady(!!getLlmSettings().key) }
    const failure = (event: Event) => setStorageError(String((event as CustomEvent).detail))
    const save = (event: Event) => scheduleWorkspaceSave((event as CustomEvent<string>).detail)
    const source = (event: Event) => setActiveSource((event as CustomEvent<EvidenceChunk>).detail)
    window.addEventListener('lumin-settings-changed', settings)
    window.addEventListener('lumin-storage-error', failure)
    window.addEventListener('lumin-workspace-mutated', save)
    window.addEventListener('lumin-open-source', source)
    return () => { disposed = true; window.removeEventListener('lumin-settings-changed', settings); window.removeEventListener('lumin-storage-error', failure); window.removeEventListener('lumin-workspace-mutated', save); window.removeEventListener('lumin-open-source', source) }
  }, [])

  const cancelGraphLoad = useCallback(() => {
    graphLoadRef.current?.abort()
    graphLoadRef.current = null
    setIsGraphLoading(false)
  }, [])

  const cancelActiveQuery = useCallback(() => {
    queryAbortRef.current?.abort()
    queryAbortRef.current = null
    setLoading(false)
  }, [])

  // Expiry, disconnect and dataset deletion all discard the same protected state.
  const resetLoadedData = useCallback(() => {
    cancelGraphLoad()
    suggestionsLoadRef.current?.abort()
    suggestionsLoadRef.current = null
    cancelActiveQuery()
    setSuggestions([])
    setIngestJob(null)
    setUploadOpen(false)
    setActiveSource(null)
    setEvidenceState(null)
    setEditMode(false)
    setMessagesLoadedFor(null)
    setMessages([])
    setGraph(EMPTY_GRAPH)
    baseGraphRef.current = EMPTY_GRAPH
    setSelectedNodeId(null)
  }, [cancelGraphLoad, cancelActiveQuery])

  useEffect(() => () => {
    graphLoadRef.current?.abort()
    suggestionsLoadRef.current?.abort()
    queryAbortRef.current?.abort()
    graphLoadRef.current = null
    suggestionsLoadRef.current = null
    queryAbortRef.current = null
  }, [])

  // ---------------------------------------------------------------------------
  // Connection + auth: poll /api/health until the server is awake, then ask
  // /api/auth/status whether we need an access key before loading anything.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let pingController: AbortController | null = null
    const statusController = new AbortController()

    const checkStatus = async () => {
      try {
        const status = await fetchAuthStatus(statusController.signal)
        if (cancelled) return
        setTokenRequired(status.token_required)
        setAuthError(null)
        if (!status.authenticated) resetLoadedData()
        setAuthPhase(status.authenticated ? 'ready' : 'locked')
      } catch (e) {
        if (cancelled || isAbortError(e)) return
        if (e instanceof ApiError && e.status === 404) {
          // Server predates the auth endpoints — no access control to satisfy
          setAuthPhase('ready')
          return
        }
        setAuthError(`Could not check access status: ${errorMessage(e)}`)
        setAuthPhase('error')
      }
    }

    const ping = async () => {
      const controller = new AbortController()
      pingController = controller
      const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS)
      let awake = false
      try {
        const res = await apiFetch('/api/health', {
          cache: 'no-store',
          signal: controller.signal,
          skipAuthCheck: true,
        })
        if (res.ok) { const health = await res.json(); setHosted(Boolean(health.features?.hosted ?? health.hosted)) }
        // An auth rejection also proves the server is up — never keep polling on 401/403
        awake = res.ok || res.status === 401 || res.status === 403
      } catch {
        // server not yet up (or timed out) — retry
      } finally {
        clearTimeout(timeout)
      }
      if (cancelled) return
      if (awake) {
        setWakingUp(false)
        await checkStatus()
        return
      }
      setWakingUp(true)
      timer = setTimeout(ping, HEALTH_RETRY_MS)
    }

    ping()
    return () => {
      cancelled = true
      clearTimeout(timer)
      pingController?.abort()
      statusController.abort()
    }
  }, [connectAttempt, resetLoadedData])

  const retryConnect = () => {
    setAuthError(null)
    setAuthPhase('connecting')
    setConnectAttempt((n) => n + 1)
  }

  // A protected request returned 401/403 — re-check the session and show the
  // access screen if it has expired. One check at a time; never loops.
  useEffect(() => {
    if (!isReady) return
    return observeSessionStatus(
      (status) => {
        setTokenRequired(status.token_required)
        if (!status.authenticated) {
          resetLoadedData()
          setAuthPhase('locked')
        }
      },
      (error) => { console.warn('Auth re-check failed:', error) },
    )
  }, [isReady, resetLoadedData])

  // ---------------------------------------------------------------------------
  // Graph + suggestions. Each load supersedes (aborts) the previous one so a
  // slow, stale response can never overwrite a newer graph or a clear.
  // ---------------------------------------------------------------------------
  const loadGraph = useCallback(() => {
    graphLoadRef.current?.abort()
    const controller = new AbortController()
    graphLoadRef.current = controller
    return apiJson<GraphData>('/api/graph', { signal: controller.signal, cache: 'no-store' })
      .then(async (data) => {
        const normalized = await normalizeGraphConnections(data)
        if (controller.signal.aborted) return
        const next = normalized.nodes.length > 0 ? unhighlighted(normalized) : EMPTY_GRAPH
        // Old inverse operations can refer to removed parallel edge identities.
        if (baseGraphRef.current.association_version !== next.association_version) resetGraphHistory()
        baseGraphRef.current = next
        setGraph(next)
        setSelectedNodeId((id) => id && next.nodes.some((node) => node.id === id) ? id : null)
        setEvidenceVersion((version) => version + 1)
        return true
      })
      .catch((err) => {
        if (!controller.signal.aborted && !isAbortError(err)) {
          console.warn('Could not load graph from backend:', err)
        }
        return false
      })
      .finally(() => {
        if (graphLoadRef.current === controller) {
          graphLoadRef.current = null
          setIsGraphLoading(false)
        }
      })
  }, [])

  const loadSuggestions = useCallback(() => {
    suggestionsLoadRef.current?.abort()
    const controller = new AbortController()
    suggestionsLoadRef.current = controller
    return apiJson<string[]>('/api/suggestions', { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setSuggestions(Array.isArray(data) ? data : [])
      })
      .catch((err) => {
        if (!controller.signal.aborted && !isAbortError(err)) setSuggestions([])
      })
  }, [])

  const clearChat = () => {
    cancelActiveQuery()
    setMessages([])
    setGraph(baseGraphRef.current)
    setSelectedNodeId(null)
  }

  // Restore the browser's workspace before protected reads, including after a server restart.
  useEffect(() => {
    if (!canConnect) return
    let cancelled = false
    setIsGraphLoading(true)
    restoreSelectedWorkspace(previousWorkspaceRef.current, hosted).then(async (id) => {
      if (cancelled || id !== activeWorkspaceId()) return
      previousWorkspaceRef.current = id
      setWorkspaceId(id)
      const saved = await db.chats.get(id)
      if (cancelled) return
      setMessages((saved?.messages ?? []).map((message) => ({ ...message, timestamp: new Date(message.timestamp) })))
      setMessagesLoadedFor(id)
      setWorkspaceLoading(false)
      await resumeIngestJobs(id)
      loadGraph(); loadSuggestions()
    }).catch((error) => { if (!cancelled) { setStorageError(errorMessage(error)); setIsGraphLoading(false); setWorkspaceLoading(false) } })
    return () => { cancelled = true }
  }, [canConnect, workspaceId, hosted, loadGraph, loadSuggestions, resumeIngestJobs])

  useEffect(() => {
    const changing = () => { cancelActiveQuery(); cancelGraphLoad(); setActiveSource(null); setWorkspaceLoading(true) }
    const ended = () => setWorkspaceLoading(false)
    const change = () => {
      cancelGraphLoad(); cancelActiveQuery(); suggestionsLoadRef.current?.abort()
      setMessagesLoadedFor(null); setMessages([]); setGraph(EMPTY_GRAPH); baseGraphRef.current = EMPTY_GRAPH
      setWorkspaceLoading(true)
      setSelectedNodeId(null); setEvidenceState(null); setSuggestions([]); setIngestJob(null)
      setActiveSource(null)
      setWorkspaceId(activeWorkspaceId()); resetGraphHistory()
    }
    const graphChanged = () => { loadGraph() }
    const assetsChanged = () => setEvidenceVersion((version) => version + 1)
    window.addEventListener('lumin-workspace-changed', change)
    window.addEventListener('lumin-workspace-changing', changing)
    window.addEventListener('lumin-workspace-transition-ended', ended)
    window.addEventListener('lumin-graph-changed', graphChanged)
    window.addEventListener('lumin-assets-updated', assetsChanged)
    return () => { window.removeEventListener('lumin-workspace-changed', change); window.removeEventListener('lumin-workspace-changing', changing); window.removeEventListener('lumin-workspace-transition-ended', ended); window.removeEventListener('lumin-graph-changed', graphChanged); window.removeEventListener('lumin-assets-updated', assetsChanged) }
  }, [cancelGraphLoad, cancelActiveQuery, loadGraph])

  useEffect(() => {
    if (!workspaceId || messagesLoadedFor !== workspaceId) return
    const timer = setTimeout(() => { db.chats.put({ workspaceId, messages }).catch((error) => setStorageError(errorMessage(error))) }, 150)
    return () => clearTimeout(timer)
  }, [messages, messagesLoadedFor, workspaceId])

  // ---------------------------------------------------------------------------
  // Background ingest-job polling — persisted via localStorage so it survives
  // tab closes and modal unmounts. Paused while access is not confirmed.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!workspaceId || !isReady) return
    const subscription = liveQuery(() => pendingIngestJobs(workspaceId)).subscribe({
      next: () => { resumeIngestJobs(workspaceId).catch((error) => setStorageError(errorMessage(error))) },
      error: (error) => setStorageError(errorMessage(error)),
    })
    return () => subscription.unsubscribe()
  }, [workspaceId, isReady, resumeIngestJobs])

  useEffect(() => {
    if (!workspaceId) return
    const subscription = liveQuery(() => db.rawFiles.where('workspaceId').equals(workspaceId).toArray()).subscribe({
      next: () => {
        preloadSources(workspaceId).then(() => {
          if (activeWorkspaceId() === workspaceId) window.dispatchEvent(new Event('lumin-assets-updated'))
        }).catch((error) => setStorageError(errorMessage(error)))
      },
      error: (error) => setStorageError(errorMessage(error)),
    })
    return () => subscription.unsubscribe()
  }, [workspaceId])

  const pollingJobId = ingestJob && !ingestJob.failed ? ingestJob.id : null

  useEffect(() => {
    if (!pollingJobId || !isReady) return
    const pollingWorkspaceId = workspaceId
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      try {
        const res = await apiFetch(`/api/ingest/jobs/${encodeURIComponent(pollingJobId)}`, {
          signal: controller.signal,
          cache: 'no-store',
          workspaceId: pollingWorkspaceId,
        })
        if (res.status === 404) {
          // A missing job has no reliable completion outcome; keep committed originals.
          const originals = await settleIngestOriginals(pollingJobId, null, pollingWorkspaceId)
          if (activeWorkspaceId() !== pollingWorkspaceId) return
          if (originals === 'discarded') setStorageError('This upload\'s final status is unavailable. Saved originals were not replaced. Re-upload the files to finish processing and refresh their previews.')
          await finishIngestJob(pollingWorkspaceId, pollingJobId)
          return
        }
        if (!res.ok) throw new ApiError(res.status, res.statusText)
        const job: { status: string; progress_stage?: string; error?: string | null; result?: { ingested_source_ids?: string[]; failed?: string[] } } = await res.json()
        if (controller.signal.aborted) return
        if (job.status === 'done') {
          const committedSources = job.result?.ingested_source_ids
          const originals = await settleIngestOriginals(pollingJobId, committedSources, pollingWorkspaceId)
          if (activeWorkspaceId() !== pollingWorkspaceId) return
          if (originals === 'discarded' && !Array.isArray(committedSources)) setStorageError('This server did not confirm which files were processed. Saved originals were not replaced. Update the server and re-upload the files to refresh their previews.')
          else if (originals === 'ambiguous') setStorageError('Overlapping uploads could not be matched to an exact original revision. Saved originals were kept. Re-upload those files one batch at a time to refresh their previews.')
          else if (job.result?.failed?.length) setStorageError(`Some files could not be processed: ${job.result.failed.join(', ')}. Their saved originals were kept. Re-upload those files to try again.`)
          await finishIngestJob(pollingWorkspaceId, pollingJobId)
          loadGraph()
          loadSuggestions()
          scheduleWorkspaceSave()
        } else if (job.status === 'failed' || job.status === 'cancelled') {
          await settleIngestOriginals(pollingJobId, null, pollingWorkspaceId)
          if (activeWorkspaceId() !== pollingWorkspaceId) return
          const reason = job.status === 'cancelled' ? 'Ingestion cancelled because the data was cleared.' : `Failed: ${job.error ?? 'unknown error'}`
          await finishIngestJob(pollingWorkspaceId, pollingJobId, { id: pollingJobId, stage: reason, failed: true })
        } else {
          setIngestJob({ id: pollingJobId, stage: job.progress_stage || 'Processing…' })
          timer = setTimeout(poll, 3000)
        }
      } catch (err) {
        if (controller.signal.aborted || isAbortError(err)) return
        // Auth problems are handled by the access gate; polling resumes once ready again
        if (isAuthError(err)) return
        // Network hiccup — retry
        timer = setTimeout(poll, 5000)
      }
    }

    poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [pollingJobId, isReady, workspaceId, loadGraph, loadSuggestions, finishIngestJob])

  // Hide a failed-job notice after a few seconds
  const ingestFailed = ingestJob?.failed ?? false
  useEffect(() => {
    if (!ingestFailed) return
    const t = setTimeout(() => setIngestJob(null), 8000)
    return () => clearTimeout(t)
  }, [ingestFailed])

  // Multi-tab sync: if another tab finishes/clears the job, stop polling here too
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === ingestJobKey()) {
        resumeIngestJobs(activeWorkspaceId()).catch((error) => setStorageError(errorMessage(error)))
        if (isReady) {
          loadGraph()
          loadSuggestions()
        }
      }
    }
    window.addEventListener('storage', handler)
    return () => window.removeEventListener('storage', handler)
  }, [isReady, loadGraph, loadSuggestions, resumeIngestJobs])

  const handleJobStarted = useCallback((jobId: string, ownerWorkspace = activeWorkspaceId()) => {
    localStorage.setItem(`${INGEST_JOB_KEY}:${ownerWorkspace}`, jobId)
    if (ownerWorkspace === activeWorkspaceId()) setIngestJob({ id: jobId, stage: 'Queued…' })
  }, [])

  const closeUpload = useCallback(() => setUploadOpen(false), [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ---------------------------------------------------------------------------
  // Selection — only the id is stored; the node itself is always read from the
  // latest graph so edits are reflected immediately.
  // ---------------------------------------------------------------------------
  const selectedNode = useMemo(
    () => (selectedNodeId ? graph.nodes.find((n) => n.id === selectedNodeId) ?? null : null),
    [graph.nodes, selectedNodeId],
  )

  const handleNodeClick = (node: GraphNode) => {
    setSelectedNodeId((prev) => (prev === node.id ? null : node.id))
  }

  // Fetch the rich detail (including the evidence trail joined from SQLite)
  // whenever the selected node changes or is edited.
  useEffect(() => {
    if (!selectedNodeId || !isReady) return
    const controller = new AbortController()
    const settle = (evidence: EvidenceChunk[]) =>
      setEvidenceState({ nodeId: selectedNodeId, version: evidenceVersion, evidence })
    apiJson<{ evidence?: EvidenceChunk[] }>(`/api/graph/node/${encodeURIComponent(selectedNodeId)}`, {
      signal: controller.signal,
    })
      .then((data) => {
        if (!controller.signal.aborted) settle(data.evidence ?? [])
      })
      .catch((err) => {
        if (!controller.signal.aborted && !isAbortError(err)) settle([])
      })
    return () => controller.abort()
  }, [selectedNodeId, evidenceVersion, isReady])

  const selectedNodeEvidence = evidenceForNode(evidenceState, selectedNodeId, evidenceVersion)
  const selectedNodeLoading =
    selectedNodeId !== null &&
    (!evidenceState || evidenceState.nodeId !== selectedNodeId || evidenceState.version !== evidenceVersion)

  const handleFirstKeystroke = () => {
    setGraph((prev) => {
      if (!prev.nodes.some((n) => n.highlighted)) return prev
      return {
        nodes: prev.nodes.map((n) => ({ ...n, highlighted: false, hopIndex: undefined })),
        edges: prev.edges.map((e) => ({ ...e, highlighted: false })),
      }
    })
  }

  const handleQuery = async (query: string) => {
    if (!isReady || clearing) return
    queryAbortRef.current?.abort()
    const controller = new AbortController()
    queryAbortRef.current = controller
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
      const result = await queryWorkspace(query, { ...querySettings, signal: controller.signal })
      if (controller.signal.aborted) return
      const assistantMsg: Message = { id: generateId(), timestamp: new Date(), ...result }
      setMessages((prev) => [...prev, assistantMsg])
      setGraph(highlightGraphTrace(baseGraphRef.current, result.hops ?? [], assistantMsg.id))
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) return
      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: 'assistant',
          content: error instanceof ApiError && [409, 422, 429, 503].includes(error.status)
            ? errorMessage(error)
            : 'Failed to get a response. Please check the backend is running.',
          timestamp: new Date(),
        },
      ])
    } finally {
      if (queryAbortRef.current === controller) {
        queryAbortRef.current = null
        setLoading(false)
      }
    }
  }

  const handleClearData = async () => {
    const ownerWorkspace = activeWorkspaceId()
    setClearDataOpen(false)
    setClearing(true)
    try {
      await withOriginalsLock(ownerWorkspace, async () => {
        await apiRequest('/api/data', { method: 'DELETE', workspaceId: ownerWorkspace })
        await clearWorkspaceData(ownerWorkspace)
        localStorage.removeItem(`${INGEST_JOB_KEY}:${ownerWorkspace}`)
      })
      if (activeWorkspaceId() !== ownerWorkspace) return
      resetGraphHistory()
      resetLoadedData()
    } catch (e) {
      setStorageError(`Clear failed: ${errorMessage(e)}`)
    } finally {
      setClearing(false)
    }
  }

  const handleDisconnect = async () => {
    try {
      await endSession()
    } catch (e) {
      setStorageError(`Disconnect failed: ${errorMessage(e)}`)
      return
    }
    resetLoadedData()
    setAuthPhase('locked')
  }

  // ---------------------------------------------------------------------------
  // Graph edit handlers
  // ---------------------------------------------------------------------------

  const handleAddNode = async (payload: AddNodePayload) => {
    if (!isReady) return
    try { await applyGraphOps([{ op: 'add_node', node: { ...payload, id: crypto.randomUUID() } }]) }
    catch (error) { setStorageError(errorMessage(error)) }
  }

  const handleDeleteNode = async (id: string) => {
    if (!isReady) return
    try { await applyGraphOps([{ op: 'delete_node', id }]); setSelectedNodeId(null) }
    catch (error) { setStorageError(errorMessage(error)) }
  }

  const handleUpdateNode = async (id: string, updates: Record<string, unknown>) => {
    if (!isReady) return
    await applyGraphOps([{ op: 'update_node', id, updates }])
    baseGraphRef.current = applyNodeUpdates(baseGraphRef.current, id, updates)
    setGraph((previous) => applyNodeUpdates(previous, id, updates))
    setEvidenceVersion((version) => version + 1)
  }

  const handleConnectNodes = async (source: string, target: string, relation: string) => {
    if (!isReady) return
    try { await applyGraphOps([{ op: 'add_edge', edge: { source, target, relation } }]) }
    catch (error) { setStorageError(errorMessage(error)) }
  }

  const handleDeleteEdge = async (source: string, relation: string, target: string) => {
    if (!isReady) return
    try { await applyGraphOps([{ op: 'delete_edge', source, relation, target }]) }
    catch (error) { setStorageError(errorMessage(error)) }
  }

  return (
    <div className={`lumin-shell flex h-screen overflow-hidden transition-colors duration-200 ${
      isDark ? 'bg-slate-950 text-slate-200' : 'bg-[var(--lumin-background)] text-slate-800'
    }`}>
      {/* Sidebar — Graph Panel */}
      <aside
        aria-hidden={!sidebarOpen && !fullscreen}
        inert={!sidebarOpen && !fullscreen}
        className={`lumin-graph-sidebar flex flex-col overflow-hidden ${isDark ? 'bg-slate-900' : 'bg-[var(--lumin-paper-soft)]'} ${fullscreen
          ? `fixed inset-0 z-50 w-screen ${isDark ? 'border-slate-800' : 'border-[var(--lumin-border)]'}`
          : `border-r transition-all duration-300 shrink-0 ${isDark ? 'border-slate-800' : 'border-[var(--lumin-border)]'} ${
              sidebarOpen ? `${maximized ? 'w-[55vw]' : 'w-80 xl:w-96'} max-md:fixed max-md:inset-0 max-md:z-50 max-md:w-screen` : 'w-0'
            }`
        }`}
      >
        <div className={`flex-1 overflow-hidden flex flex-col ${
          fullscreen ? 'min-w-full' : maximized ? 'min-w-[55vw]' : 'min-w-80 xl:min-w-96'
        } max-md:min-w-0 max-md:w-full`}>
          <div className="md:hidden px-3 py-2 flex justify-between items-center border-b border-slate-500/20 text-xs"><span>Knowledge graph</span><button aria-label="Close graph" onClick={() => { setSidebarOpen(false); setFullscreen(false) }} className="px-2 py-1 rounded-lg border border-slate-500/30">Close ×</button></div>
          {isReady && isGraphLoading && (
            <div className={`flex items-center justify-center gap-2 px-3 py-1.5 text-xs border-b ${
              isDark
                ? 'text-slate-500 border-slate-800 bg-slate-900/40'
                : 'text-slate-400 border-[var(--lumin-border)] bg-[var(--lumin-background)]/40'
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
              active={!activeSource && (sidebarOpen || fullscreen)}
              maximized={maximized}
              onMaximize={() => setMaximized((v) => !v)}
              fullscreen={fullscreen}
              onFullscreen={() => setFullscreen((v) => !v)}
              editMode={editMode && isReady}
              onEditModeChange={setEditMode}
              onAddNode={handleAddNode}
              onDeleteNode={handleDeleteNode}
              onConnectNodes={handleConnectNodes}
              onViewDetails={(id) => {
                if (graph.nodes.some((n) => n.id === id)) {
                  setSelectedNodeId(id)
                  setTimeout(() => nodeDetailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
                }
              }}
            />
          </Suspense>
          {selectedNode && (
            <div ref={nodeDetailRef}>
            <NodeDetailPanel
              key={selectedNode.id}
              node={selectedNode}
              edges={graph.edges}
              allNodes={graph.nodes}
              evidence={selectedNodeEvidence}
              evidenceLoading={selectedNodeLoading}
              onClose={() => setSelectedNodeId(null)}
              editMode={editMode && isReady}
              onDeleteNode={handleDeleteNode}
              onDeleteEdge={handleDeleteEdge}
              onUpdateNode={handleUpdateNode}
            />
            </div>
          )}
        </div>
      </aside>

      {/* Main — Chat */}
      <main className="lumin-main flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className={`lumin-header flex flex-wrap sm:flex-nowrap items-center gap-3 px-4 py-3 border-b backdrop-blur shrink-0 ${
          isDark
            ? 'border-slate-800 bg-slate-900/60'
            : 'border-[var(--lumin-border)] bg-white/80'
        }`}>
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

          <div className="flex items-center gap-2">
            <BrandMark size="small" />
            <div>
              <span className={`lumin-brand-name font-semibold text-sm ${isDark ? 'text-white' : 'text-slate-900'}`}>LuminRAG</span>
              <span className={`lumin-brand-caption ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>A little space to learn</span>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2">
            {ingestJob && (
              <span className={`text-xs flex items-center gap-1.5 ${
                ingestJob.failed
                  ? (isDark ? 'text-red-400' : 'text-red-500')
                  : (isDark ? 'text-indigo-400' : 'text-indigo-600')
              }`}>
                {!ingestJob.failed && (
                  <div className={`w-3 h-3 border border-t-indigo-400 rounded-full animate-spin ${isDark ? 'border-slate-600' : 'border-slate-300'}`} />
                )}
                {ingestJob.stage}
              </span>
            )}
            {!ingestJob && (
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

            {tokenRequired && isReady && (
              <button
                onClick={handleDisconnect}
                className={`text-xs border px-2.5 py-1 rounded-lg transition-colors ${
                  isDark
                    ? 'text-slate-500 hover:text-slate-300 border-slate-700 hover:border-slate-600'
                    : 'text-slate-500 hover:text-slate-700 border-slate-300 hover:border-slate-400 bg-white hover:bg-slate-50'
                }`}
                title="End this session"
              >
                Disconnect
              </button>
            )}

            <button
              onClick={() => setClearDataOpen(true)}
              disabled={clearing || !isReady}
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
              disabled={!isReady}
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
              onClick={clearChat}
              className={`text-xs border px-2.5 py-1 rounded-lg transition-colors ${
                isDark
                  ? 'text-slate-500 hover:text-slate-300 border-slate-700 hover:border-slate-600'
                  : 'text-slate-500 hover:text-slate-700 border-slate-300 hover:border-slate-400 bg-white hover:bg-slate-50'
              }`}
            >
              Clear
            </button>
          </div>
        </header>

        <WorkspaceToolbar disabled={!canConnect || workspaceLoading || clearing} onSettings={() => setSettingsOpen(true)} />
        {storageError && <div role="alert" className="text-xs text-red-500 px-4 py-2 border-b border-red-500/20">{storageError}<button onClick={() => setStorageError(null)} className="ml-3">Dismiss</button></div>}
        {/* Messages */}
        <div className="lumin-chat-feed flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="lumin-welcome">
              <div className="lumin-welcome-art" aria-hidden="true"><span className="lumin-welcome-spark">✿</span><span className="lumin-welcome-spark">✦</span><BrandMark size="large" /></div>
              <div><p className="lumin-eyebrow">A little curiosity goes a long way</p><h1>Welcome to your study nook</h1></div>
              <p className="lumin-welcome-copy">Bring your course material, ask a question, and watch the ideas connect. One small discovery at a time.</p>
              <button className="lumin-welcome-action" disabled={!isReady} onClick={() => setUploadOpen(true)}>Add your first materials</button>
              <div className="lumin-welcome-tags"><span>Notes &amp; books</span><span>Connected ideas</span><span>Your own pace</span></div>
              {suggestions.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg mt-2">
                  {suggestions.slice(0, 4).map((q) => (
                    <button
                      key={q}
                      onClick={() => handleQuery(q)}
                      className={`lumin-suggestion text-left text-xs border rounded-xl p-3 transition-all ${
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
              <BrandMark size="small" />
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
        <div className={`lumin-composer-dock px-4 pb-4 pt-2 border-t backdrop-blur shrink-0 ${
          isDark ? 'border-slate-800 bg-slate-900/60' : 'border-[var(--lumin-border)] bg-white/80'
        }`}>
          <div className="max-w-3xl mx-auto">
            <QueryInput onSubmit={handleQuery} options={querySettings} onOptionsChange={setQuerySettings} loading={loading || !isReady || clearing} onFirstKeystroke={handleFirstKeystroke} suggestions={suggestions} />
          </div>
        </div>
      </main>

      {uploadOpen && (
        <UploadModal
          onClose={closeUpload}
          onJobStarted={handleJobStarted}
        />
      )}

      {(authPhase === 'locked' || authPhase === 'error') && (
        <AccessGate
          isDark={isDark}
          statusError={authPhase === 'error' ? authError : null}
          onRetryStatus={retryConnect}
          onAuthenticated={() => setAuthPhase('ready')}
        />
      )}

      {(settingsOpen || hosted && !providerReady && authPhase === 'ready') && <SettingsModal hosted={hosted} required={hosted && !providerReady} onClose={() => { setSettingsOpen(false); setProviderReady(!!getLlmSettings().key) }} />}
      {activeSource && <Suspense fallback={<div role="status" className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 text-white">Opening source…</div>}><SourceViewer key={activeSource.id} chunk={activeSource} onClose={() => setActiveSource(null)} /></Suspense>}
      {clearDataOpen && <div className="fixed inset-0 z-[80] bg-black/60 flex items-center justify-center p-4"><div role="alertdialog" aria-modal="true" aria-labelledby="clear-data-title" className={`max-w-sm w-full p-5 rounded-2xl border space-y-3 ${isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'}`}><h2 id="clear-data-title" className="font-semibold">Clear this knowledge base?</h2><p className="text-sm text-slate-500">This removes its materials, graph, original files and chats from this browser and its server cache. Export a .lumin backup first if you want to keep them.</p><div className="flex justify-end gap-3 text-sm"><button onClick={() => setClearDataOpen(false)}>Cancel</button><button onClick={handleClearData} className="text-red-500">Clear knowledge base</button></div></div></div>}

      {wakingUp && (
        <div className={`fixed inset-0 z-50 flex flex-col items-center justify-center backdrop-blur gap-4 ${
          isDark ? 'bg-slate-950/90' : 'bg-[var(--lumin-background)]/90'
        }`}>
          <BrandMark size="medium" />
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
