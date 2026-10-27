import type { Theme } from '../theme'

/** Retain a saved choice; new study nooks begin in warm daylight. */
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem('lumin-theme')
    return saved === 'dark' || saved === 'light' ? saved : 'light'
  } catch { return 'light' }
}

const GRAPH_FALLBACKS: Record<string, string> = {
  topic: '#66884b', subtopic: '#b17a59', content: '#578a78', note: '#a97d2d',
  'path-day': '#a5573e', 'path-night': '#e0ad77', 'label-day': '#3b5033', 'label-night': '#e9e8cf',
  'edge-day': '#7c8964', 'edge-night': '#8b9d76', 'stage-day': '#f7f3e4', 'stage-night': '#24382b',
  'backdrop-day': '#fffaf0', 'backdrop-night': '#314a34',
}

/** Canvas cannot resolve CSS custom properties itself, so resolve our shared tokens once. */
export function graphPalette(dark: boolean): Record<string, string> {
  const styles = typeof document === 'undefined' ? undefined : getComputedStyle(document.documentElement)
  const palette: Record<string, string> = {}
  for (const name of ['topic', 'subtopic', 'content', 'note', 'path', 'label', 'edge', 'stage', 'backdrop']) {
    const token = ['topic', 'subtopic', 'content', 'note'].includes(name) ? name : `${name}-${dark ? 'night' : 'day'}`
    palette[name] = styles?.getPropertyValue(`--lumin-graph-${token}`).trim() || GRAPH_FALLBACKS[token]
  }
  return palette
}
