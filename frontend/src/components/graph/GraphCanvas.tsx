import { useEffect, useMemo, useRef, useState, Component, type ReactNode, type RefObject } from 'react'
import { SigmaContainer, useSigma, useRegisterEvents } from '@react-sigma/core'
import type Sigma from 'sigma'
import type { Dimensions } from 'sigma/types'
import { MultiDirectedGraph } from 'graphology'
import type { GraphData, GraphNode } from '../../types'
import { graphTopologyKey, selectedNeighborhood } from '../../utils/graph'
import { resizeGraphViewport } from '../../utils/graphViewport'
import { graphOptions } from './graphStyles'
import { syncGraphology, toGraphology } from './toGraphology'
import { useLayout } from './useLayout'
import { useConnectGesture } from './useConnectGesture'
import { useLasso } from './useLasso'
import { applyGraphOps, useGraphStore } from './useGraphStore'
import '@react-sigma/core/lib/style.css'

export interface CanvasMenu { node?: string; left: number; top: number; x: number; y: number }
interface Props {
  data: GraphData; dark: boolean; editMode: boolean; selectedNodeId: string | null; hiddenIds: Set<string>
  neighborMode: boolean; spacing: number; layoutRevision: number; active: boolean
  onNodeClick: (node: GraphNode) => void; onContext: (menu: CanvasMenu) => void
  onConnect: (source: string, target: string) => void; rendererRef: RefObject<Sigma | null>
}

class GraphBoundary extends Component<{ children: ReactNode; onRetry: () => void }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(error: unknown) { return { error: error instanceof Error ? error.message : 'Graph display failed' } }
  render() { return this.state.error ? <div role="alert" className="p-6 text-sm text-slate-500"><p>{/WebGL|rendering context/i.test(this.state.error) ? 'The graph needs WebGL 2. Enable it or try another browser.' : 'The graph could not be displayed. Try loading it again.'} Your knowledge base is saved in this browser.</p><button className="mt-3 text-indigo-400" onClick={() => { this.props.onRetry(); this.setState({ error: null }) }}>Retry graph</button></div> : this.props.children }
}

function GraphController({ data, editMode, selectedNodeId, hiddenIds, neighborMode, spacing, layoutRevision, active, onNodeClick, onContext, onConnect, line, lasso }: Omit<Props, 'dark' | 'rendererRef'> & { line: RefObject<SVGPathElement | null>; lasso: RefObject<SVGPathElement | null> }) {
  const sigma = useSigma()
  const register = useRegisterEvents()
  const selectedIds = useGraphStore((state) => state.selectedIds)
  const topology = graphTopologyKey(data)
  const visibility = JSON.stringify(data.nodes.filter((node) => !hiddenIds.has(node.id)).map((node) => node.id))
  const shouldLayout = data.nodes.some((node) => !Number.isFinite(node.x) || !Number.isFinite(node.y)) || layoutRevision > 0
  useEffect(() => {
    syncGraphology(sigma.getGraph() as MultiDirectedGraph, data)
    const selected = new Set([...selectedIds, ...(selectedNodeId ? [selectedNodeId] : [])])
    const neighbors = selectedNeighborhood(sigma.getGraph() as MultiDirectedGraph, selected)
    const hasPath = data.nodes.some((node) => node.highlighted)
    for (const node of data.nodes) {
      sigma.setNodeState(node.id, { isHidden: hiddenIds.has(node.id), isHighlighted: !!node.highlighted })
      sigma.getGraph().setNodeAttribute(node.id, 'selected', selected.has(node.id))
      sigma.getGraph().setNodeAttribute(node.id, 'dimmed', neighborMode ? !neighbors.has(node.id) : hasPath && !node.highlighted && !selected.has(node.id))
    }
    for (const edge of data.edges) if (sigma.getGraph().hasEdge(edge.id)) {
      sigma.setEdgeState(edge.id, { isHidden: hiddenIds.has(edge.source) || hiddenIds.has(edge.target), isHighlighted: !!edge.highlighted })
      sigma.getGraph().setEdgeAttribute(edge.id, 'dimmed', neighborMode ? !neighbors.has(edge.source) || !neighbors.has(edge.target) : hasPath && !edge.highlighted)
    }
  }, [sigma, data, hiddenIds, selectedIds, selectedNodeId, neighborMode])
  useLayout(`${topology}:${layoutRevision}`, spacing, shouldLayout && active)
  useConnectGesture(editMode, line, onConnect)
  useLasso(editMode, lasso)
  useEffect(() => {
    sigma.setSetting('enableNodeDrag', editMode)
    register({
      clickNode: ({ node, event }) => { if (!(event.original instanceof MouseEvent) || !event.original.shiftKey) { const value = data.nodes.find((candidate) => candidate.id === node); if (value) onNodeClick(value) } },
      rightClickNode: ({ node, event }) => { event.original.preventDefault(); const position = sigma.viewportToGraph(event); onContext({ node, left: event.x, top: event.y, ...position }) },
      rightClickStage: ({ event }) => { event.original.preventDefault(); const position = sigma.viewportToGraph(event); onContext({ left: event.x, top: event.y, ...position }) },
      nodeDragEnd: ({ allDraggedNodes }) => { if (editMode) applyGraphOps([{ op: 'set_positions', positions: allDraggedNodes.map((id) => ({ id, x: sigma.getGraph().getNodeAttribute(id, 'x'), y: sigma.getGraph().getNodeAttribute(id, 'y') })) }]).catch(() => {}) },
    })
  }, [sigma, register, data, editMode, onNodeClick, onContext])
  useEffect(() => {
    let fittedSize: Dimensions | null = null
    const resize = () => { fittedSize = resizeGraphViewport(sigma, fittedSize, active) }
    const observer = new ResizeObserver(resize)
    observer.observe(sigma.getContainer())
    resize()
    return () => observer.disconnect()
  }, [active, sigma, topology, visibility])
  return null
}

export default function GraphCanvas({ rendererRef, ...props }: Props) {
  const [renderGraph, setRenderGraph] = useState(() => toGraphology(props.data))
  const options = useMemo(() => graphOptions(props.dark), [props.dark])
  const line = useRef<SVGPathElement>(null)
  const lasso = useRef<SVGPathElement>(null)
  return <GraphBoundary onRetry={() => setRenderGraph(toGraphology(props.data))}><div className="relative h-full w-full" onContextMenu={(event) => event.preventDefault()}>
    <SigmaContainer graph={renderGraph} options={options} ref={rendererRef} style={{ height: '100%', width: '100%' }}>
      <GraphController {...props} line={line} lasso={lasso} />
    </SigmaContainer>
    <svg className="absolute inset-0 pointer-events-none w-full h-full" aria-hidden="true"><path ref={line} d="" fill="none" stroke="var(--lumin-forest)" strokeWidth="2" strokeDasharray="5 4" /><path ref={lasso} d="" fill="var(--lumin-forest-soft)" fillOpacity="0.4" stroke="var(--lumin-forest)" strokeWidth="1.5" /></svg>
  </div></GraphBoundary>
}
