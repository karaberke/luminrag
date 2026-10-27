import { useEffect, type RefObject } from 'react'
import { useSigma } from '@react-sigma/core'

/** Cancel native drag only for Shift+drag; ordinary movement stays entirely in Sigma. */
export function useConnectGesture(enabled: boolean, line: RefObject<SVGPathElement | null>, onConnect: (source: string, target: string) => void): void {
  const sigma = useSigma()
  useEffect(() => {
    if (!enabled) return
    const overlay = line.current
    let source: string | null = null
    const point = (event: MouseEvent) => { const rect = sigma.getContainer().getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top } }
    const move = (event: MouseEvent) => {
      if (!source || !sigma.getGraph().hasNode(source)) return
      const start = sigma.graphToViewport(sigma.getGraph().getNodeAttributes(source) as { x: number; y: number })
      const end = point(event)
      overlay?.setAttribute('d', `M ${start.x} ${start.y} L ${end.x} ${end.y}`)
    }
    const up = (event: MouseEvent) => {
      if (!source) return
      const from = source
      source = null
      overlay?.setAttribute('d', '')
      const release = point(event)
      let target: string | undefined
      let distance = 24
      sigma.getGraph().forEachNode((id, attrs) => {
        if (id === from || sigma.getNodeState(id).isHidden) return
        const screen = sigma.graphToViewport(attrs as { x: number; y: number })
        const candidate = Math.hypot(screen.x - release.x, screen.y - release.y)
        if (candidate < distance) { target = id; distance = candidate }
      })
      if (target) onConnect(from, target)
    }
    const down: Parameters<typeof sigma.on<'downNode'>>[1] = ({ node, event }) => {
      const original = event.original
      if (!(original instanceof MouseEvent) || !original.shiftKey) return
      event.preventSigmaDefault()
      source = node
      original.preventDefault()
    }
    sigma.on('downNode', down)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { sigma.off('downNode', down); window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); overlay?.setAttribute('d', '') }
  }, [sigma, enabled, line, onConnect])
}
