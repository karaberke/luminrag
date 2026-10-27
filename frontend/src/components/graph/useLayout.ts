import { useEffect } from 'react'
import { useSigma } from '@react-sigma/core'
import ForceAtlas2Worker from 'graphology-layout-forceatlas2/worker'
import { applyGraphOps } from './useGraphStore'
import { fitGraphBounds } from '../../utils/graphViewport'
import { graphLayoutSettings, separateGraphNodes } from '../../utils/graphLayout'

export function useLayout(layoutKey: string, spacing: number, shouldLayout: boolean): void {
  const sigma = useSigma()
  useEffect(() => {
    if (!shouldLayout || !sigma.getGraph().order) return
    const graph = sigma.getGraph()
    const layout = new ForceAtlas2Worker(graph, { settings: graphLayoutSettings(spacing) })
    layout.start()
    const timer = setTimeout(() => {
      layout.stop()
      if (!fitGraphBounds(sigma, { animate: false })) return
      separateGraphNodes(sigma, spacing)
      fitGraphBounds(sigma, { animate: false, preserveScale: true })
      const positions = graph.mapNodes((id, attrs) => ({ id, x: attrs.x, y: attrs.y }))
      // Automatic placement is persisted without displacing the user's semantic edit or redo history.
      applyGraphOps([{ op: 'set_positions', positions }], { recordHistory: false }).catch(() => {})
    }, Math.min(3500, 700 + sigma.getGraph().order / 2))
    return () => { clearTimeout(timer); layout.kill() }
  }, [sigma, layoutKey, spacing, shouldLayout])
}
