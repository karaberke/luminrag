import type Sigma from 'sigma'
import type { Dimensions } from 'sigma/types'

/** Freeze a fresh extent only on topology/layout changes or the explicit Fit action. */
export function fitGraphBounds(renderer: Sigma, { animate = true, preserveScale = false }: { animate?: boolean; preserveScale?: boolean } = {}): boolean {
  const container = renderer.getContainer()
  // Sigma substitutes a 1x1 stage for hidden containers. Never freeze a fitted
  // extent against that fallback: wait until the panel has real dimensions.
  if (container.offsetWidth <= 0 || container.offsetHeight <= 0) return false
  if (preserveScale && renderer.getCustomBBox()) {
    const camera = renderer.getCamera()
    camera.cancelAnimation()
    // Process pending position/viewport changes before using public projection.
    renderer.refresh()
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    renderer.getGraph().forEachNode((id, attributes) => {
      if (renderer.getNodeState(id).isHidden || !Number.isFinite(attributes.x) || !Number.isFinite(attributes.y)) return
      const point = renderer.graphToViewport({ x: attributes.x, y: attributes.y })
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return
      const radius = renderer.scaleSize(typeof attributes.size === 'number' ? attributes.size : 0)
      minX = Math.min(minX, point.x - radius); maxX = Math.max(maxX, point.x + radius)
      minY = Math.min(minY, point.y - radius); maxY = Math.max(maxY, point.y + radius)
    })
    if (!Number.isFinite(minX)) return false
    const width = container.offsetWidth, height = container.offsetHeight
    const padding = Math.min(24, width / 4, height / 4)
    const scale = Math.max((maxX - minX) / (width - 2 * padding), (maxY - minY) / (height - 2 * padding))
    const center = renderer.viewportToFramedGraph({ x: (minX + maxX) / 2, y: (minY + maxY) / 2 })
    // The reference ratio is 1. Zooming circles and distances proportionally
    // retains the collision layout while Fit also recovers from manual zoom-out.
    const next = { x: center.x, y: center.y, ratio: camera.getBoundedRatio(Math.max(1, camera.ratio * scale)) }
    if (animate) void camera.animate(next, { duration: 200 })
    else { camera.setState(next); renderer.refresh() }
    return true
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  renderer.getGraph().forEachNode((id, attributes) => {
    if (renderer.getNodeState(id).isHidden || !Number.isFinite(attributes.x) || !Number.isFinite(attributes.y)) return
    minX = Math.min(minX, attributes.x); maxX = Math.max(maxX, attributes.x)
    minY = Math.min(minY, attributes.y); maxY = Math.max(maxY, attributes.y)
  })
  if (!Number.isFinite(minX)) return false
  const margin = Math.max(maxX - minX, maxY - minY, 1) * 0.12
  renderer.setCustomBBox({ x: [minX - margin, maxX + margin], y: [minY - margin, maxY + margin] })
  const camera = renderer.getCamera()
  if (!animate) {
    camera.cancelAnimation()
    camera.setState({ x: 0.5, y: 0.5, ratio: 1, angle: 0 })
  }
  // setCustomBBox only schedules a render in Sigma4; refresh rebuilds the
  // coordinate normalization and label grid for the new extent and viewport.
  renderer.refresh()
  if (animate) void camera.reset({ duration: 200 })
  return true
}

/** Fit once per visible panel size, retaining camera changes between resizes. */
export function resizeGraphViewport(renderer: Sigma, previous: Dimensions | null, active: boolean): Dimensions | null {
  if (!active) return null
  const container = renderer.getContainer()
  const width = container.offsetWidth, height = container.offsetHeight
  if (width <= 0 || height <= 0) return null
  if (previous?.width === width && previous.height === height) return previous
  renderer.resize()
  fitGraphBounds(renderer, { animate: false, preserveScale: true })
  return { width, height }
}
