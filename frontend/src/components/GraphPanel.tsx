import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
// @ts-expect-error — no bundled types for fcose
import fcose from 'cytoscape-fcose'
import type { GraphData, GraphNode } from '../types'
import { useTheme } from '../ThemeContext'

// Register layout extension once at module level
if (!(cytoscape as any)._fcoseRegistered) {
  cytoscape.use(fcose)
  ;(cytoscape as any)._fcoseRegistered = true
}

const STRUCTURAL_RELATIONS = ['HAS_SUBTOPIC', 'HAS_CONTENT'] as const
const SEMANTIC_RELATIONS = ['RELATED_TO'] as const
const RELATIONS = [...STRUCTURAL_RELATIONS, ...SEMANTIC_RELATIONS] as const

type NewNodeType = 'topic' | 'subtopic' | 'content'

interface AddNodePayload {
  name: string
  node_type: NewNodeType
}

interface Props {
  graph: GraphData
  onNodeClick: (node: GraphNode) => void
  selectedNodeId: string | null
  maximized?: boolean
  onMaximize?: () => void
  fullscreen?: boolean
  onFullscreen?: () => void
  editMode?: boolean
  onEditModeChange?: (v: boolean) => void
  onAddNode?: (payload: AddNodePayload) => void
  onDeleteNode?: (nodeId: string) => void
  onConnectNodes?: (
    sourceId: string,
    targetId: string,
    relation: string,
    label?: string,
    confidence?: number,
  ) => void
  onViewDetails?: (nodeId: string) => void
}

interface PendingEdge {
  sourceId: string
  sourceLabel: string
  targetId: string
  targetLabel: string
}

type Tab = 'full' | 'subgraph'

// Base layout options — `fit` is handled manually in runLayout so we can
// clamp zoom for large graphs. Params are overridden for large node counts.
const LAYOUT_BASE = {
  name: 'fcose',
  animate: false,
  fit: false,          // we fit manually after layout to control zoom
  padding: 40,
  edgeElasticity: 0.45,
  gravityRange: 0.5,
  randomize: true,
  tile: true,
  tilingPaddingVertical: 150,
  tilingPaddingHorizontal: 150,
} as const

/** Minimum zoom at which node labels are still readable (~9px font) */
const MIN_READABLE_ZOOM = 0.3

export default function GraphPanel({
  graph,
  onNodeClick,
  selectedNodeId,
  maximized,
  onMaximize,
  fullscreen = false,
  onFullscreen,
  editMode = false,
  onEditModeChange,
  onAddNode,
  onDeleteNode,
  onConnectNodes,
  onViewDetails,
}: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const [activeTab, setActiveTab] = useState<Tab>('full')
  const [connectingFromId, setConnectingFromId] = useState<string | null>(null)
  const [pendingEdge, setPendingEdge] = useState<PendingEdge | null>(null)
  const [pendingRelation, setPendingRelation] = useState<string>('HAS_SUBTOPIC')
  const [showAddNodeForm, setShowAddNodeForm] = useState(false)
  const [addNodeName, setAddNodeName] = useState('')
  const [addNodeType, setAddNodeType] = useState<NewNodeType>('topic')
  const addNodeInputRef = useRef<HTMLInputElement>(null)
  const [collapsedTopics, setCollapsedTopics] = useState<Set<string>>(new Set())
  const [spacing, setSpacing] = useState(1) // 0.3–3× multiplier for layout spread
  // Layout-effect keys off this; slider commits to it on release so drag doesn't trigger relayouts.
  const [committedSpacing, setCommittedSpacing] = useState(1)
  const [neighborMode, setNeighborMode] = useState(false)

  const cyFullRef = useRef<cytoscape.Core | null>(null)
  const cySubRef = useRef<cytoscape.Core | null>(null)

  // Deferred-layout flags: set when layout is skipped due to hidden container
  const fullNeedsLayoutRef = useRef(false)
  const subNeedsLayoutRef  = useRef(false)

  // Warm-start flags: once a cy instance has been laid out, re-layouts start
  // from current node positions (randomize=false) and converge in far fewer
  // iterations than a cold spectral init.
  const fullHasLaidOutRef = useRef(false)
  const subHasLaidOutRef  = useRef(false)

  // -----------------------------------------------------------------------
  // "Latest state" ref — Cytoscape event handlers read current React state
  // from here instead of closing over stale render-time values.
  // Updated every render so handlers never see old values.
  // -----------------------------------------------------------------------
  const latestRef = useRef({
    connectingFromId, graph, onNodeClick, pendingEdge, showAddNodeForm,
    fullscreen, onFullscreen,
  })
  latestRef.current = {
    connectingFromId, graph, onNodeClick, pendingEdge, showAddNodeForm,
    fullscreen, onFullscreen,
  }

  // Keys for topology-change detection — computed once per render, not inside effects
  const prevFullKeyRef = useRef('__unset__')
  const prevSubKeyRef  = useRef('__unset__')
  const prevHlCountRef = useRef(0)

  // -----------------------------------------------------------------------
  // Collapse/expand topic clusters
  // -----------------------------------------------------------------------
  const toggleCollapse = useCallback((topicId: string) => {
    setCollapsedTopics(prev => {
      const next = new Set(prev)
      if (next.has(topicId)) next.delete(topicId)
      else next.add(topicId)
      return next
    })
  }, [])

  const allTopicIds = useMemo(
    () => graph.nodes.filter(n => n.node_type === 'topic').map(n => n.id),
    [graph.nodes],
  )

  const collapseAll = useCallback(() => {
    setCollapsedTopics(new Set(allTopicIds))
  }, [allTopicIds])

  const expandAll = useCallback(() => {
    setCollapsedTopics(new Set())
  }, [])

  // Prune stale collapsed topics when graph changes
  useEffect(() => {
    setCollapsedTopics(prev => {
      const validIds = new Set(graph.nodes.filter(n => n.node_type === 'topic').map(n => n.id))
      let changed = false
      for (const tid of prev) {
        if (!validIds.has(tid)) { changed = true; break }
      }
      if (!changed) return prev
      const next = new Set<string>()
      for (const tid of prev) {
        if (validIds.has(tid)) next.add(tid)
      }
      return next
    })
  }, [graph.nodes])

  // Compute filtered nodes/edges based on collapsed topics
  const { filteredNodes, filteredEdges, collapseCountMap } = useMemo(() => {
    if (collapsedTopics.size === 0) {
      return { filteredNodes: graph.nodes, filteredEdges: graph.edges, collapseCountMap: new Map<string, number>() }
    }
    // Build downward adjacency (topic/subtopic → children) via HAS_* edges.
    const HIERARCHY_RELS = new Set(['HAS_SUBTOPIC', 'HAS_CONTENT'])
    const out = new Map<string, string[]>()
    for (const e of graph.edges) {
      if (HIERARCHY_RELS.has(e.relation)) {
        if (!out.has(e.source)) out.set(e.source, [])
        out.get(e.source)!.push(e.target)
      }
    }
    const descendantsOf = (rootId: string): Set<string> => {
      const seen = new Set<string>()
      const queue = [rootId]
      while (queue.length) {
        const cur = queue.shift()!
        for (const nxt of out.get(cur) ?? []) {
          if (!seen.has(nxt)) { seen.add(nxt); queue.push(nxt) }
        }
      }
      return seen
    }

    // Protect descendants reachable from any non-collapsed topic.
    const protectedIds = new Set<string>()
    for (const n of graph.nodes) {
      if (n.node_type !== 'topic' || collapsedTopics.has(n.id)) continue
      for (const d of descendantsOf(n.id)) protectedIds.add(d)
    }

    const hiddenNodeIds = new Set<string>()
    const countMap = new Map<string, number>()
    for (const topicId of collapsedTopics) {
      let count = 0
      for (const d of descendantsOf(topicId)) {
        if (protectedIds.has(d)) continue
        const child = graph.nodes.find(n => n.id === d)
        if (child?.node_type === 'topic') continue
        hiddenNodeIds.add(d)
        count++
      }
      countMap.set(topicId, count)
    }
    return {
      filteredNodes: graph.nodes.filter(n => !hiddenNodeIds.has(n.id)),
      filteredEdges: graph.edges.filter(e => !hiddenNodeIds.has(e.source) && !hiddenNodeIds.has(e.target)),
      collapseCountMap: countMap,
    }
  }, [graph.nodes, graph.edges, collapsedTopics])

  const fullNodeKey = useMemo(
    () => filteredNodes.map((n) => n.id).join(','),
    [filteredNodes],
  )
  const subNodeKey = useMemo(
    () => filteredNodes.filter((n) => n.highlighted).map((n) => n.id).sort().join(','),
    [filteredNodes],
  )

  // -----------------------------------------------------------------------
  // Memoised stylesheet — rebuilt only when theme flips
  // -----------------------------------------------------------------------
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const stylesheet = useMemo<any[]>(
    () => [
      {
        selector: 'node',
        style: {
          'background-color': isDark ? '#334155' : '#e0e7ff',
          'border-color': isDark ? '#475569' : '#a5b4fc',
          'border-width': 1.5,
          label: 'data(label)',
          'font-size': 9,
          color: isDark ? '#94a3b8' : '#4b5563',
          'text-halign': 'center',
          'text-valign': 'center',
          'text-wrap': 'wrap',
          'text-max-width': '80px',
          width: 90,
          height: 36,
          shape: 'round-rectangle',
          'min-zoomed-font-size': 4,
        },
      },
      {
        selector: 'node.topic',
        style: {
          'background-color': isDark ? '#312e81' : '#e0e7ff',
          'border-color': isDark ? '#4f46e5' : '#818cf8',
          'border-width': 2,
          width: 110,
          height: 44,
          'font-size': 11,
          'font-weight': 'bold',
        },
      },
      {
        selector: 'node.subtopic',
        style: {
          'background-color': isDark ? '#581c87' : '#faf5ff',
          'border-color': isDark ? '#a855f7' : '#d8b4fe',
          width: 100,
          height: 40,
        },
      },
      {
        selector: 'node.content',
        style: {
          'background-color': isDark ? '#0f766e' : '#ccfbf1',
          'border-color': isDark ? '#0d9488' : '#5eead4',
          width: 85,
          height: 32,
          'font-size': 9,
        },
      },
      {
        selector: 'node.collapsed',
        style: {
          'background-color': isDark ? '#1e3a3a' : '#b2f5ea',
          'border-color': isDark ? '#14b8a6' : '#0d9488',
          'border-width': 2.5,
          'border-style': 'dashed',
          shape: 'round-rectangle',
        },
      },
      {
        selector: 'node.highlighted',
        style: {
          'background-color': '#6366f1',
          'border-color': isDark ? '#a5b4fc' : '#4f46e5',
          'border-width': 2.5,
          color: '#fff',
          'font-weight': 'bold',
        },
      },
      {
        selector: 'node.selected',
        style: { 'border-color': '#f59e0b', 'border-width': 3 },
      },
      {
        selector: 'node.connecting-src',
        style: {
          'background-color': isDark ? '#92400e' : '#fef3c7',
          'border-color': '#f59e0b',
          'border-width': 3,
        },
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'line-color': isDark ? '#334155' : '#c7d2fe',
          'target-arrow-color': isDark ? '#475569' : '#a5b4fc',
          'arrow-scale': 0.8,
          width: 1.2,
          opacity: 0.5,
          label: 'data(label)',
          'font-size': 7,
          color: isDark ? '#475569' : '#94a3b8',
          'text-rotation': 'autorotate',
          'text-margin-y': -6,
          'min-zoomed-font-size': 4,
        },
      },
      {
        selector: 'edge.highlighted',
        style: {
          'line-color': isDark ? '#818cf8' : '#6366f1',
          'target-arrow-color': isDark ? '#818cf8' : '#6366f1',
          width: 2.5,
          opacity: 0.9,
          color: isDark ? '#a5b4fc' : '#4f46e5',
          'font-weight': 'bold',
        },
      },
      {
        selector: 'node.dimmed',
        style: { opacity: 0.12 },
      },
    ],
    [isDark],
  )

  // -----------------------------------------------------------------------
  // Memoised element arrays — react-cytoscapejs diffs these against
  // the live graph and only patches what changed (classes, data, etc.)
  // -----------------------------------------------------------------------
  const highlightedIds = useMemo(
    () => new Set(filteredNodes.filter((n) => n.highlighted).map((n) => n.id)),
    [filteredNodes],
  )

  // BFS over all edges (undirected) to find the full connected component of
  // the selected node. Used for neighbor-highlight mode.
  const neighborSet = useMemo<Set<string>>(() => {
    if (!neighborMode || !selectedNodeId) return new Set()
    const adj = new Map<string, Set<string>>()
    for (const e of graph.edges) {
      if (!adj.has(e.source)) adj.set(e.source, new Set())
      if (!adj.has(e.target)) adj.set(e.target, new Set())
      adj.get(e.source)!.add(e.target)
      adj.get(e.target)!.add(e.source)
    }
    const visited = new Set<string>()
    const queue = [selectedNodeId]
    while (queue.length) {
      const cur = queue.shift()!
      if (visited.has(cur)) continue
      visited.add(cur)
      adj.get(cur)?.forEach((nb) => { if (!visited.has(nb)) queue.push(nb) })
    }
    return visited
  }, [neighborMode, selectedNodeId, graph.edges])

  // Reset neighbor mode whenever the selected node changes
  useEffect(() => { setNeighborMode(false) }, [selectedNodeId])

  const fullElements = useMemo<cytoscape.ElementDefinition[]>(
    () => [
      ...filteredNodes.map((n) => ({
        data: {
          id: n.id,
          label: collapsedTopics.has(n.id)
            ? `${n.name} (${collapseCountMap.get(n.id) ?? 0})`
            : n.name,
        },
        classes: [
          n.node_type,                                          // topic | subtopic | content
          n.highlighted                                          ? 'highlighted'    : '',
          n.id === selectedNodeId                                ? 'selected'       : '',
          n.id === connectingFromId                              ? 'connecting-src' : '',
          collapsedTopics.has(n.id)                             ? 'collapsed'      : '',
          neighborMode && neighborSet.size > 0 && !neighborSet.has(n.id) ? 'dimmed' : '',
        ].filter(Boolean).join(' '),
      })),
      ...filteredEdges.map((e) => ({
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          label: e.relation === 'RELATED_TO' && e.label ? e.label : e.relation,
        },
        classes: e.highlighted ? 'highlighted' : '',
      })),
    ],
    [filteredNodes, filteredEdges, selectedNodeId, connectingFromId, collapsedTopics, collapseCountMap, neighborMode, neighborSet],
  )

  const subElements = useMemo<cytoscape.ElementDefinition[]>(
    () => [
      ...filteredNodes
        .filter((n) => n.highlighted)
        .map((n) => ({
          data: {
            id: n.id,
            label: collapsedTopics.has(n.id)
              ? `${n.name} (${collapseCountMap.get(n.id) ?? 0})`
              : n.name,
          },
          classes: [
            'highlighted',
            n.node_type,                                     // topic | subtopic | content
            n.id === selectedNodeId        ? 'selected'  : '',
            collapsedTopics.has(n.id)      ? 'collapsed' : '',
          ].filter(Boolean).join(' '),
        })),
      ...filteredEdges
        .filter((e) => highlightedIds.has(e.source) && highlightedIds.has(e.target))
        .map((e) => ({
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.relation === 'RELATED_TO' && e.label ? e.label : e.relation,
          },
          classes: 'highlighted',
        })),
    ],
    [filteredNodes, filteredEdges, highlightedIds, selectedNodeId, collapsedTopics, collapseCountMap],
  )

  // -----------------------------------------------------------------------
  // Layout runner — guards against zero-dimension (hidden) containers
  // -----------------------------------------------------------------------
  const runLayout = useCallback((cy: cytoscape.Core, force = false) => {
    if (cy.nodes().length === 0) return
    const container = cy.container()
    if (!force && container && (container.clientWidth === 0 || container.clientHeight === 0)) {
      // Container is hidden (inactive tab) — defer layout until it becomes visible
      if (cy === cyFullRef.current) fullNeedsLayoutRef.current = true
      if (cy === cySubRef.current)  subNeedsLayoutRef.current  = true
      return
    }

    const s = committedSpacing // multiplier from slider (0.3–3); only updates on release
    const nodeCount = cy.nodes().length

    // Warm-start: once the instance has been laid out, start from current
    // positions rather than a fresh spectral init. Converges in far fewer
    // iterations without the up-front cost of randomization.
    const hasLaidOutRef = cy === cyFullRef.current ? fullHasLaidOutRef : subHasLaidOutRef
    const warmStart = hasLaidOutRef.current

    // Adaptive layout params — repulsion and edge length must exceed node
    // dimensions (90×36) to prevent overlap.  Spacing multiplier scales
    // repulsion / edge length / separation while inversely scaling gravity.
    const opts: Record<string, unknown> = {
      ...LAYOUT_BASE,
      randomize: !warmStart,
      tilingPaddingVertical:   Math.round(150 * s),
      tilingPaddingHorizontal: Math.round(150 * s),
    }
    if (nodeCount > 150) {
      opts.gravity = 0.02 / s
      opts.nodeRepulsion = 500000 * s * s
      opts.idealEdgeLength = Math.round(800 * s)
      opts.nodeSeparation = Math.round(400 * s)
      opts.numIter = warmStart ? 400 : 1500
    } else if (nodeCount > 50) {
      opts.gravity = 0.01 / s
      opts.nodeRepulsion = 400000 * s * s
      opts.idealEdgeLength = Math.round(700 * s)
      opts.nodeSeparation = Math.round(350 * s)
      opts.numIter = warmStart ? 250 : 800
    } else {
      opts.gravity = 0.005 / s
      opts.nodeRepulsion = 300000 * s * s
      opts.idealEdgeLength = Math.round(600 * s)
      opts.nodeSeparation = Math.round(300 * s)
      opts.numIter = warmStart ? 150 : 400
    }

    cy.layout(opts as any).run()
    hasLaidOutRef.current = true

    // Fit all nodes into view, then clamp zoom so labels stay readable.
    // For large graphs this means the user sees a portion and can pan around.
    cy.fit(undefined, 40)
    if (cy.zoom() < MIN_READABLE_ZOOM) {
      cy.zoom({
        level: MIN_READABLE_ZOOM,
        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
      })
      cy.center()
    }
  }, [committedSpacing])

  // Re-run layout when spacing slider is released (committed), not on every drag tick.
  const prevSpacingRef = useRef(committedSpacing)
  useEffect(() => {
    if (prevSpacingRef.current === committedSpacing) return
    prevSpacingRef.current = committedSpacing
    const cy = activeTab === 'full' ? cyFullRef.current : cySubRef.current
    if (cy) runLayout(cy, true)
  }, [committedSpacing, activeTab, runLayout])

  // Trigger layout when full-graph topology changes.
  useEffect(() => {
    if (fullNodeKey === prevFullKeyRef.current) return
    prevFullKeyRef.current = fullNodeKey
    const cy = cyFullRef.current
    if (cy) runLayout(cy)
  }, [fullNodeKey, runLayout])

  // Trigger layout when subgraph topology changes; auto-switch tabs
  useEffect(() => {
    const hlCount = filteredNodes.filter((n) => n.highlighted).length
    if (hlCount > 0 && prevHlCountRef.current === 0) setActiveTab('subgraph')
    if (hlCount === 0 && prevHlCountRef.current > 0) setActiveTab('full')
    prevHlCountRef.current = hlCount

    if (subNodeKey === prevSubKeyRef.current) return
    prevSubKeyRef.current = subNodeKey
    const cy = cySubRef.current
    if (cy && hlCount > 0) runLayout(cy)
  }, [subNodeKey, filteredNodes, runLayout])

  // -----------------------------------------------------------------------
  // Cytoscape event wiring.
  // Registered ONCE per cy instance. All state is read from latestRef.current
  // so handlers are always fresh without ever being re-bound.
  // -----------------------------------------------------------------------
  const wireCy = useCallback((cy: cytoscape.Core) => {
    cy.removeAllListeners()

    cy.on('tap', 'node', (evt) => {
      const nodeId: string = evt.target.id()
      const { connectingFromId: srcId, graph: g, onNodeClick: cb } = latestRef.current

      if (srcId) {
        if (nodeId === srcId) { setConnectingFromId(null); return }
        setPendingEdge({
          sourceId: srcId,
          sourceLabel: cy.getElementById(srcId).data('label') ?? srcId,
          targetId: nodeId,
          targetLabel: evt.target.data('label'),
        })
        setPendingRelation('HAS_SUBTOPIC')
        setConnectingFromId(null)
        return
      }

      const graphNode = g.nodes.find((n) => n.id === nodeId)
      if (graphNode) cb(graphNode)
    })

    cy.on('tap', (evt) => {
      if (evt.target === cy && latestRef.current.connectingFromId) {
        setConnectingFromId(null)
      }
    })

    // Double-tap a topic node to collapse/expand
    cy.on('dbltap', 'node.topic', (evt) => {
      toggleCollapse(evt.target.id())
    })
  }, [toggleCollapse]) // toggleCollapse is stable (no deps)

  // Resize + deferred layout when tab becomes visible (hidden div has 0 dimensions).
  // Guarded by prevActiveTabRef so this only fires on a real tab change, not
  // whenever runLayout's identity updates (e.g. committedSpacing change).
  const prevActiveTabRef = useRef<Tab>('full')
  useEffect(() => {
    if (prevActiveTabRef.current === activeTab) return
    prevActiveTabRef.current = activeTab

    const cy = activeTab === 'full' ? cyFullRef.current : cySubRef.current
    const needsRef = activeTab === 'full' ? fullNeedsLayoutRef : subNeedsLayoutRef
    if (!cy) return
    const t = setTimeout(() => {
      cy.resize()
      if (needsRef.current) {
        needsRef.current = false
        runLayout(cy, true) // force=true bypasses the dimension check
      }
      // Skip cy.fit() when no layout is needed — viewport was already fit
      // when the tab was last visible and resize() alone doesn't change it.
    }, 50)
    return () => clearTimeout(t)
  }, [activeTab, runLayout])

  // Clear edit state when editMode turns off
  useEffect(() => {
    if (!editMode) {
      setConnectingFromId(null)
      setPendingEdge(null)
      setShowAddNodeForm(false)
      setAddNodeName('')
    }
  }, [editMode])

  // Focus add-node input when form opens
  useEffect(() => {
    if (showAddNodeForm) setTimeout(() => addNodeInputRef.current?.focus(), 50)
  }, [showAddNodeForm])

  // Escape key — reads from latestRef so the handler is registered once
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const {
        pendingEdge: pe, connectingFromId: cid, showAddNodeForm: saf,
        fullscreen: fs, onFullscreen: exitFs,
      } = latestRef.current
      if (pe)  { setPendingEdge(null);    return }
      if (cid) { setConnectingFromId(null); return }
      if (saf) { setShowAddNodeForm(false); setAddNodeName(''); return }
      if (fs && exitFs) exitFs()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, []) // stable forever — state access via latestRef

  // -----------------------------------------------------------------------
  // Edit mode actions
  // -----------------------------------------------------------------------
  const handleDeleteNodeClick = useCallback(
    (nodeId: string, label: string) => {
      if (!window.confirm(`Delete "${label}" and all its edges?`)) return
      onDeleteNode?.(nodeId)
    },
    [onDeleteNode],
  )

  const confirmAddNode = useCallback(() => {
    const name = addNodeName.trim()
    if (!name) return
    onAddNode?.({ name, node_type: addNodeType })
    setShowAddNodeForm(false)
    setAddNodeName('')
    setAddNodeType('topic')
  }, [addNodeName, addNodeType, onAddNode])

  const [pendingRelationLabel, setPendingRelationLabel] = useState('')

  const confirmEdge = useCallback(() => {
    if (!pendingEdge) return
    const relation = pendingRelation.trim().toUpperCase()
    if (!relation) return
    if (relation === 'RELATED_TO') {
      const label = pendingRelationLabel.trim()
      if (!label) return
      onConnectNodes?.(pendingEdge.sourceId, pendingEdge.targetId, relation, label)
    } else {
      onConnectNodes?.(pendingEdge.sourceId, pendingEdge.targetId, relation)
    }
    setPendingEdge(null)
    setPendingRelationLabel('')
  }, [pendingEdge, pendingRelation, pendingRelationLabel, onConnectNodes])

  // -----------------------------------------------------------------------
  // Zoom helpers
  // -----------------------------------------------------------------------
  const activeCy = () => (activeTab === 'subgraph' ? cySubRef.current : cyFullRef.current)
  const zoomIn   = () => { const cy = activeCy(); cy?.zoom({ level: cy.zoom() * 1.25, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }) }
  const zoomOut  = () => { const cy = activeCy(); cy?.zoom({ level: cy.zoom() * 0.8,  renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }) }
  const resetZoom = () => activeCy()?.fit(undefined, 30)

  const captureGraph = () => {
    const cy = activeCy()
    if (!cy || cy.nodes().length === 0) return

    // Temporarily apply a print-friendly stylesheet: black text, bold, bigger
    // nodes, light palette — always readable on a white background regardless
    // of the current UI theme. Both cy.style() and cy.png() are synchronous so
    // the user never sees the style swap.
    const printStylesheet = [
      ...stylesheet,
      { selector: 'node',           style: { color: '#000000', 'font-weight': 'bold', 'font-size': 12, width: 126, height: 50, 'text-max-width': '108px', 'background-color': '#f8faff', 'border-color': '#94a3b8' } },
      { selector: 'node.topic',     style: { 'background-color': '#e0e7ff', 'border-color': '#4f46e5', color: '#000000', 'font-weight': 'bold', width: 154, height: 62, 'font-size': 14 } },
      { selector: 'node.subtopic',  style: { 'background-color': '#faf5ff', 'border-color': '#a855f7', color: '#000000', width: 140, height: 56 } },
      { selector: 'node.content',   style: { 'background-color': '#ccfbf1', 'border-color': '#0d9488', color: '#000000', width: 119, height: 45, 'font-size': 11 } },
      { selector: 'node.highlighted', style: { 'background-color': '#6366f1', 'border-color': '#4338ca', color: '#ffffff' } },
      { selector: 'node.selected',  style: { 'border-color': '#d97706', 'border-width': 3 } },
      { selector: 'edge',           style: { 'line-color': '#94a3b8', 'target-arrow-color': '#94a3b8', color: '#374151', opacity: 1 } },
      { selector: 'edge.highlighted', style: { 'line-color': '#6366f1', 'target-arrow-color': '#6366f1', color: '#4f46e5' } },
    ]

    cy.style(printStylesheet as any)
    const dataUrl = cy.png({ full: true, scale: 2, output: 'base64uri', bg: '#ffffff' })
    cy.style(stylesheet as any)

    const a = document.createElement('a')
    a.href = dataUrl
    a.download = `${activeTab}-graph.png`
    a.click()
  }

  // -----------------------------------------------------------------------
  // Derived display values
  // -----------------------------------------------------------------------
  const subgraphNodeCount = filteredNodes.filter((n) => n.highlighted).length
  const subEdgeCount = useMemo(
    () => filteredEdges.filter((e) => highlightedIds.has(e.source) && highlightedIds.has(e.target)).length,
    [filteredEdges, highlightedIds],
  )
  const connectingLabel = useMemo(
    () => graph.nodes.find((n) => n.id === connectingFromId)?.name ?? connectingFromId,
    [graph.nodes, connectingFromId],
  )
  const selectedLabel = useMemo(
    () => graph.nodes.find((n) => n.id === selectedNodeId)?.name ?? selectedNodeId,
    [graph.nodes, selectedNodeId],
  )
  const selectedNodeType = useMemo(
    () => graph.nodes.find((n) => n.id === selectedNodeId)?.node_type,
    [graph.nodes, selectedNodeId],
  )

  const btnCls = isDark
    ? 'w-6 h-6 flex items-center justify-center text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded transition-colors'
    : 'w-6 h-6 flex items-center justify-center text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors'

  const hasTopics = allTopicIds.length > 0

  // -----------------------------------------------------------------------
  // JSX
  // -----------------------------------------------------------------------
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 pt-3 pb-0 flex items-start justify-between gap-2 shrink-0">
        <div className="min-w-0">
          <h2 className={`text-sm font-semibold uppercase tracking-widest ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
            {activeTab === 'full' ? 'Concept Graph' : 'Retrieved Subgraph'}
          </h2>
          <p className={`text-xs mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            {editMode
              ? 'Edit mode — add nodes, connect, delete'
              : activeTab === 'full'
              ? 'Highlighted = retrieval path · drag nodes · scroll to zoom'
              : 'Nodes and relations extracted in the last query'}
          </p>
        </div>

        <div className="flex items-center gap-0.5 shrink-0 mt-0.5">
          <button onClick={zoomOut}   className={btnCls} title="Zoom out"><span className="text-sm font-mono leading-none">−</span></button>
          <button onClick={resetZoom} className={btnCls} title="Reset zoom">
            <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
              <circle cx="5.5" cy="5.5" r="3.5" /><path d="M8.5 8.5l3 3" strokeLinecap="round" />
              <path d="M5.5 3.5v4M3.5 5.5h4" strokeLinecap="round" />
            </svg>
          </button>
          <button onClick={zoomIn}    className={btnCls} title="Zoom in"><span className="text-sm font-mono leading-none">+</span></button>
          <button onClick={captureGraph} className={btnCls} title="Save graph as PNG">
            <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
              <path d="M2 4.5A1.5 1.5 0 013.5 3h.879l.707-1h3.828l.707 1H10.5A1.5 1.5 0 0112 4.5v6A1.5 1.5 0 0110.5 12h-7A1.5 1.5 0 012 10.5v-6z" strokeLinejoin="round"/>
              <circle cx="7" cy="7.5" r="1.75"/>
            </svg>
          </button>

          <div className={`w-px h-4 mx-0.5 ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`} />

          {/* Spacing slider */}
          <div className="flex items-center gap-1 mx-1" title={`Node spacing: ${spacing.toFixed(1)}×`}>
            <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className={`w-3 h-3 shrink-0 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              <circle cx="4" cy="7" r="2" /><circle cx="10" cy="7" r="2" />
              <path d="M1 7h1M6 7h2M12 7h1" strokeLinecap="round" />
            </svg>
            <input
              type="range"
              min="0.3"
              max="3"
              step="0.1"
              value={spacing}
              onChange={(e) => setSpacing(parseFloat(e.target.value))}
              onPointerUp={(e) => setCommittedSpacing(parseFloat((e.target as HTMLInputElement).value))}
              onKeyUp={(e) => setCommittedSpacing(parseFloat((e.target as HTMLInputElement).value))}
              className="w-16 h-1 accent-indigo-500 cursor-pointer"
            />
            <span className={`text-[10px] tabular-nums w-7 text-right ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              {spacing.toFixed(1)}×
            </span>
          </div>

          <div className={`w-px h-4 mx-0.5 ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`} />

          {/* Collapse All / Expand All — only shown when topics exist */}
          {hasTopics && (
            <>
              <button
                onClick={collapseAll}
                className={btnCls}
                title="Collapse all topics"
              >
                <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path d="M3 5l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              <button
                onClick={expandAll}
                className={btnCls}
                title="Expand all topics"
              >
                <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path d="M3 9l4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>

              <div className={`w-px h-4 mx-0.5 ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`} />
            </>
          )}

          <button
            onClick={() => onEditModeChange?.(!editMode)}
            className={`${btnCls} ${editMode ? (isDark ? 'text-amber-400 bg-slate-700/80' : 'text-amber-500 bg-amber-50') : ''}`}
            title={editMode ? 'Exit edit mode' : 'Edit graph'}
          >
            <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
              <path d="M9.5 2.5l2 2L4 12H2v-2L9.5 2.5z" strokeLinejoin="round" />
            </svg>
          </button>

          {editMode && (
            <button
              onClick={() => setShowAddNodeForm(true)}
              className={`${btnCls} text-emerald-400 hover:text-emerald-300`}
              title="Add node"
            >
              <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                <circle cx="7" cy="7" r="5.5" /><path d="M7 4.5v5M4.5 7h5" strokeLinecap="round" />
              </svg>
            </button>
          )}

          {onMaximize && !fullscreen && (
            <button onClick={onMaximize} className={`${btnCls} ml-1`} title={maximized ? 'Restore sidebar' : 'Expand sidebar'}>
              {maximized ? (
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path d="M6 2v4H2M10 2v4h4M6 14v-4H2M10 14v-4h4" strokeLinecap="round" />
                </svg>
              ) : (
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4" strokeLinecap="round" />
                </svg>
              )}
            </button>
          )}

          {onFullscreen && (
            <button
              onClick={onFullscreen}
              className={`${btnCls} ${fullscreen ? (isDark ? 'text-indigo-400 bg-slate-700/80' : 'text-indigo-600 bg-indigo-50') : ''}`}
              title={fullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen graph'}
            >
              {fullscreen ? (
                /* Exit fullscreen — arrows pointing inward */
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path d="M10 2v4h4M2 10h4v4M6 6L2 2M10 6l4-4M6 10l-4 4M10 10l4 4" strokeLinecap="round" />
                </svg>
              ) : (
                /* Enter fullscreen — arrows pointing outward to corners */
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4M2 2l4 4M14 2l-4 4M2 14l4-4M14 14l-4-4" strokeLinecap="round" />
                </svg>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Add node form */}
      {editMode && showAddNodeForm && (
        <div className={`mx-4 mt-2 p-2.5 border rounded-lg flex flex-col gap-2 shrink-0 ${
          isDark ? 'bg-slate-800 border-slate-600' : 'bg-white border-slate-200 shadow-sm'
        }`}>
          <p className={`text-xs font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>New node</p>
          <input
            ref={addNodeInputRef}
            value={addNodeName}
            onChange={(e) => setAddNodeName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') confirmAddNode() }}
            placeholder="Node name…"
            className={`text-xs border rounded px-2 py-1 outline-none focus:border-indigo-500 ${
              isDark ? 'bg-slate-700 border-slate-600 text-slate-200' : 'bg-slate-50 border-slate-200 text-slate-800'
            }`}
          />
          <div className="flex items-center gap-3">
            {(['topic', 'subtopic', 'content'] as const).map((t) => (
              <label key={t} className={`flex items-center gap-1.5 text-xs cursor-pointer ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                <input type="radio" name="nodeType" value={t} checked={addNodeType === t} onChange={() => setAddNodeType(t)} className="accent-indigo-500" />
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </label>
            ))}
            <div className="ml-auto flex gap-1.5">
              <button
                onClick={confirmAddNode}
                disabled={!addNodeName.trim()}
                className="text-xs bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 disabled:cursor-not-allowed text-white px-2 py-0.5 rounded"
              >
                Add
              </button>
              <button
                onClick={() => { setShowAddNodeForm(false); setAddNodeName('') }}
                className={`text-xs px-1 ${isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-400 hover:text-slate-600'}`}
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connecting mode banner */}
      {editMode && connectingFromId && (
        <div className="mx-4 mt-1.5 px-3 py-1.5 bg-amber-900/30 border border-amber-800/50 rounded-lg flex items-center gap-2 shrink-0">
          <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3 text-amber-400 shrink-0">
            <circle cx="3" cy="7" r="1.5" /><circle cx="11" cy="7" r="1.5" />
            <path d="M4.5 7h5" strokeLinecap="round" strokeDasharray="1.5 1.5" />
          </svg>
          <span className="text-xs text-amber-300">
            Connecting from <strong className="text-amber-200">{connectingLabel}</strong> — click a target node
          </span>
          <button onClick={() => setConnectingFromId(null)} className="ml-auto text-xs text-amber-500 hover:text-amber-300">
            Cancel
          </button>
        </div>
      )}

      {/* Tab bar */}
      <div className="px-4 pt-2 pb-0 flex gap-1 shrink-0">
        {(['full', 'subgraph'] as Tab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1 text-xs font-medium rounded-t-md border-b-2 transition-colors ${
              activeTab === tab
                ? isDark ? 'border-indigo-500 text-slate-200 bg-slate-800/60' : 'border-indigo-500 text-slate-800 bg-white'
                : isDark ? 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-800/30' : 'border-transparent text-slate-400 hover:text-slate-600 hover:bg-slate-100/50'
            }`}
          >
            {tab === 'full' ? 'Full Graph' : 'Subgraph'}
            {tab === 'full' && filteredNodes.length > 0 && (
              <span className={`ml-1.5 font-normal ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>{filteredNodes.length}</span>
            )}
            {tab === 'subgraph' && subgraphNodeCount > 0 && (
              <span className={`ml-1.5 px-1.5 py-0.5 rounded-full font-normal ${isDark ? 'bg-indigo-600/30 text-indigo-300' : 'bg-indigo-100 text-indigo-600'}`}>
                {subgraphNodeCount}
              </span>
            )}
            {tab === 'subgraph' && subgraphNodeCount === 0 && (
              <span className={`ml-1.5 font-normal ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>—</span>
            )}
          </button>
        ))}
        <div className={`flex-1 border-b-2 ${isDark ? 'border-slate-700/50' : 'border-slate-200'}`} />
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-hidden px-2 relative">
        {filteredNodes.length === 0 && !editMode ? (
          <div className={`flex items-center justify-center h-full text-xs text-center px-4 ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
            No graph loaded.<br />Run the ingestion pipeline to populate the concept graph.
          </div>
        ) : (
          <>
            {/* Empty subgraph overlay — absolute so it floats above the hidden canvas
                without unmounting CytoscapeComponent (which triggers an expensive cold remount). */}
            {activeTab === 'subgraph' && subgraphNodeCount === 0 && (
              <div className={`absolute inset-0 flex flex-col items-center justify-center text-xs text-center px-6 gap-2 z-10 pointer-events-none ${isDark ? 'text-slate-600' : 'text-slate-400'}`}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8" style={{ color: isDark ? '#334155' : '#c7d2fe' }}>
                  <circle cx="5" cy="12" r="2" /><circle cx="19" cy="6" r="2" /><circle cx="19" cy="18" r="2" />
                  <path d="M7 11.5l10-4M7 12.5l10 4" strokeLinecap="round" />
                </svg>
                No retrieval trace yet.<br />Ask a question to see the extracted subgraph.
              </div>
            )}

            {activeTab === 'subgraph' && subgraphNodeCount > 0 && (
              <div className={`flex items-center gap-3 px-2 py-1 text-xs shrink-0 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                <span><span className="font-medium" style={{ color: isDark ? '#818cf8' : '#6366f1' }}>{subgraphNodeCount}</span> nodes</span>
                <span><span className="font-medium" style={{ color: isDark ? '#818cf8' : '#6366f1' }}>{subEdgeCount}</span> edges</span>
                <span className={`ml-auto italic ${isDark ? 'text-slate-600' : 'text-slate-300'}`}>from last query</span>
              </div>
            )}

            {/* Full graph */}
            <div className={`w-full h-full ${activeTab === 'full' ? 'block' : 'hidden'}`}
                 style={{ cursor: connectingFromId ? 'crosshair' : undefined }}>
              <CytoscapeComponent
                key="full"
                elements={fullElements}
                stylesheet={stylesheet}
                style={{ width: '100%', height: '100%' }}
                cy={(cy) => {
                  if (cyFullRef.current !== cy) {
                    cyFullRef.current = cy
                    wireCy(cy)
                    // Defer initial layout so react-cytoscapejs can patch elements first
                    requestAnimationFrame(() => {
                      if (cy.nodes().length > 0) {
                        prevFullKeyRef.current = fullNodeKey
                        runLayout(cy)
                      }
                    })
                  }
                }}
                userZoomingEnabled
                userPanningEnabled
                autoungrabify={false}
              />
            </div>

            {/* Subgraph */}
            <div className={`w-full h-full ${activeTab === 'subgraph' ? 'block' : 'hidden'}`}>
              <CytoscapeComponent
                key="sub"
                elements={subElements}
                stylesheet={stylesheet}
                style={{ width: '100%', height: '100%' }}
                cy={(cy) => {
                  if (cySubRef.current !== cy) {
                    cySubRef.current = cy
                    wireCy(cy)
                    requestAnimationFrame(() => {
                      if (cy.nodes().length > 0) {
                        prevSubKeyRef.current = subNodeKey
                        runLayout(cy)
                      }
                    })
                  }
                }}
                userZoomingEnabled
                userPanningEnabled
                autoungrabify={false}
              />
            </div>

            {/* Node toolbar — appears when a node is selected */}
            {selectedNodeId && (
              <div className={`absolute top-2 right-3 flex gap-1.5 z-10 border rounded-lg px-2 py-1.5 shadow-lg backdrop-blur ${
                isDark ? 'bg-slate-800/90 border-slate-700' : 'bg-white/90 border-slate-200'
              }`}>
                <span className={`text-xs self-center mr-1 max-w-[120px] truncate font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                  {selectedLabel}
                </span>

                {/* View Details button — always visible when a node is selected */}
                <button
                  onClick={() => onViewDetails?.(selectedNodeId)}
                  className={`text-xs px-2 py-0.5 rounded ${
                    isDark
                      ? 'bg-indigo-700/80 hover:bg-indigo-600 text-white'
                      : 'bg-indigo-500/90 hover:bg-indigo-600 text-white'
                  }`}
                >
                  View Details
                </button>

                {/* Neighbor highlight toggle */}
                <button
                  onClick={() => setNeighborMode((v) => !v)}
                  className={`text-xs px-2 py-0.5 rounded ${
                    neighborMode
                      ? 'bg-violet-600 hover:bg-violet-500 text-white'
                      : isDark
                        ? 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                        : 'bg-slate-200 hover:bg-slate-300 text-slate-700'
                  }`}
                  title="Highlight all nodes connected to this one"
                >
                  Neighbors
                </button>

                {/* Collapse/Expand — shown for topic nodes regardless of edit mode */}
                {selectedNodeType === 'topic' && (
                  <button
                    onClick={() => toggleCollapse(selectedNodeId)}
                    className={`text-xs px-2 py-0.5 rounded text-white ${
                      collapsedTopics.has(selectedNodeId)
                        ? 'bg-teal-700/80 hover:bg-teal-600'
                        : 'bg-teal-700/80 hover:bg-teal-600'
                    }`}
                  >
                    {collapsedTopics.has(selectedNodeId) ? 'Expand' : 'Collapse'}
                  </button>
                )}

                {editMode && (
                  <>
                    <button
                      onClick={() => setConnectingFromId(selectedNodeId)}
                      className="text-xs px-2 py-0.5 rounded bg-amber-700/80 hover:bg-amber-600 text-white"
                    >
                      Connect →
                    </button>
                    <button
                      onClick={() => {
                        const node = graph.nodes.find((n) => n.id === selectedNodeId)
                        if (node) handleDeleteNodeClick(node.id, node.name)
                      }}
                      className="text-xs px-2 py-0.5 rounded bg-red-700/80 hover:bg-red-600 text-white"
                    >
                      Delete
                    </button>
                  </>
                )}
              </div>
            )}

            {/* Relation picker overlay */}
            {pendingEdge && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
                <div className={`border rounded-xl p-4 shadow-2xl pointer-events-auto w-72 ${
                  isDark ? 'bg-slate-800 border-slate-600' : 'bg-white border-slate-200 shadow-xl'
                }`}>
                  <p className={`text-xs mb-2 font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Add edge</p>
                  <div className="flex items-center gap-1.5 text-xs mb-3 flex-wrap">
                    <span className="font-semibold text-indigo-400">{pendingEdge.sourceLabel}</span>
                    <span className={isDark ? 'text-slate-500' : 'text-slate-400'}>→</span>
                    <span className="font-semibold text-indigo-400">{pendingEdge.targetLabel}</span>
                  </div>
                  <label className={`text-xs mb-1 block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Relation</label>
                  <input
                    list="relation-suggestions"
                    value={pendingRelation}
                    onChange={(e) => setPendingRelation(e.target.value.toUpperCase())}
                    placeholder="HAS_SUBTOPIC, HAS_CONTENT, RELATED_TO…"
                    autoFocus
                    className={`w-full text-xs border rounded px-2 py-1.5 outline-none focus:border-indigo-500 mb-3 ${
                      isDark ? 'bg-slate-700 border-slate-600 text-slate-200' : 'bg-slate-50 border-slate-200 text-slate-800'
                    }`}
                  />
                  <datalist id="relation-suggestions">
                    {RELATIONS.map((r) => <option key={r} value={r} />)}
                  </datalist>
                  {pendingRelation.trim().toUpperCase() === 'RELATED_TO' && (
                    <>
                      <label className={`text-xs mb-1 block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Label (free-form verb phrase)
                      </label>
                      <input
                        value={pendingRelationLabel}
                        onChange={(e) => setPendingRelationLabel(e.target.value)}
                        placeholder="uses, generalises, prerequisite for…"
                        className={`w-full text-xs border rounded px-2 py-1.5 outline-none focus:border-indigo-500 mb-3 ${
                          isDark ? 'bg-slate-700 border-slate-600 text-slate-200' : 'bg-slate-50 border-slate-200 text-slate-800'
                        }`}
                      />
                    </>
                  )}
                  <div className="flex gap-2">
                    <button
                      onClick={confirmEdge}
                      disabled={
                        !pendingRelation.trim() ||
                        (pendingRelation.trim().toUpperCase() === 'RELATED_TO' && !pendingRelationLabel.trim())
                      }
                      className="flex-1 text-xs bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white py-1 rounded"
                    >
                      Add Edge
                    </button>
                    <button
                      onClick={() => setPendingEdge(null)}
                      className={`text-xs px-2 ${isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Legend */}
      <div className="px-4 pb-3 flex gap-4 flex-wrap shrink-0">
        {activeTab === 'full' ? (
          <>
            <div className="flex items-center gap-1.5"><div className={`w-3 h-3 rounded-full ${isDark ? 'bg-indigo-900 border border-indigo-500' : 'bg-indigo-100 border border-indigo-400'}`} /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Topic</span></div>
            <div className="flex items-center gap-1.5"><div className={`w-3 h-3 rounded-full ${isDark ? 'bg-purple-900 border border-purple-500' : 'bg-purple-100 border border-purple-400'}`} /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Subtopic</span></div>
            <div className="flex items-center gap-1.5"><div className={`w-3 h-3 rounded-full ${isDark ? 'bg-teal-900 border border-teal-500' : 'bg-teal-100 border border-teal-400'}`} /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Content</span></div>
            <div className="flex items-center gap-1.5"><div className={`w-3 h-3 rounded-full bg-indigo-500`} /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Retrieved</span></div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-amber-500" /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Selected</span></div>
            {editMode && <div className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>· Select a node to Connect or Delete it</div>}
          </>
        ) : (
          <>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-indigo-500" /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Retrieved node</span></div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-amber-500" /><span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Selected</span></div>
          </>
        )}
      </div>
    </div>
  )
}
