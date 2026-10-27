import { useEffect, type RefObject } from 'react'
import { useSigma } from '@react-sigma/core'
import { pointInPolygon, type Point } from '../../utils/graphGestures'
import { useGraphStore } from './useGraphStore'

export function useLasso(enabled: boolean, path: RefObject<SVGPathElement | null>): void {
  const sigma = useSigma()
  useEffect(() => {
    if (!enabled) return
    const overlay = path.current
    const container = sigma.getContainer()
    let points: Point[] = []
    const point = (event: MouseEvent) => { const rect = container.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top } }
    const down = (event: MouseEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.button !== 0) return
      event.preventDefault(); event.stopPropagation()
      points = [point(event)]
    }
    const move = (event: MouseEvent) => {
      if (!points.length) return
      const next = point(event), last = points.at(-1)!
      if (Math.hypot(next.x - last.x, next.y - last.y) < 4) return
      points.push(next)
      overlay?.setAttribute('d', `M ${points.map((p) => `${p.x} ${p.y}`).join(' L ')} Z`)
    }
    const up = () => {
      if (points.length > 2) {
        const selected = sigma.getGraph().filterNodes((id, attrs) => !sigma.getNodeState(id).isHidden && pointInPolygon(sigma.graphToViewport(attrs as { x: number; y: number }), points))
        useGraphStore.getState().setSelection(selected)
      }
      points = []
      overlay?.setAttribute('d', '')
    }
    container.addEventListener('mousedown', down, true)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { container.removeEventListener('mousedown', down, true); window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); overlay?.setAttribute('d', '') }
  }, [sigma, enabled, path])
}
