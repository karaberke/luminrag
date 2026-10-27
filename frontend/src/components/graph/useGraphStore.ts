import { create } from 'zustand'
import { temporal } from 'zundo'
import type { GraphOp, GraphOpsResponse } from '../../types'
import { apiJson } from '../../utils/api'
import { activeWorkspaceId } from '../../storage/db'

interface HistoryEntry { forward: GraphOp[]; inverse: GraphOp[] }
interface GraphEditorState { steps: HistoryEntry[]; busy: boolean; selectedIds: string[]; setSelection: (ids: string[]) => void }
export const useGraphStore = create<GraphEditorState>()(temporal((set) => ({
  steps: [], busy: false, selectedIds: [], setSelection: (selectedIds) => set({ selectedIds }),
}), { partialize: ({ steps }) => ({ steps }), equality: (a, b) => a.steps === b.steps, limit: 100 }))

async function submit(ops: GraphOp[]): Promise<GraphOpsResponse> {
  const workspaceId = activeWorkspaceId()
  const response = await apiJson<GraphOpsResponse>('/api/graph/ops', { method: 'POST', json: { ops }, workspaceId })
  if (activeWorkspaceId() === workspaceId) window.dispatchEvent(new CustomEvent('lumin-graph-changed', { detail: response.graph }))
  return response
}

export async function applyGraphOps(ops: GraphOp[], { recordHistory = true }: { recordHistory?: boolean } = {}): Promise<void> {
  if (!ops.length) return
  if (useGraphStore.getState().busy) throw new Error('Wait for the current graph edit to finish.')
  useGraphStore.setState({ busy: true })
  const workspaceId = activeWorkspaceId()
  try {
    const response = await submit(ops)
    if (recordHistory && workspaceId === activeWorkspaceId()) useGraphStore.setState(({ steps }) => ({ steps: [...steps, { forward: ops, inverse: response.inverses ?? response.inverse ?? [] }] }))
  } finally { useGraphStore.setState({ busy: false }) }
}

export async function undoGraphEdit(): Promise<void> {
  const state = useGraphStore.getState()
  if (state.busy || !state.steps.length) return
  const entry = state.steps.at(-1)!
  if (!entry.inverse.length) return
  useGraphStore.setState({ busy: true })
  const workspaceId = activeWorkspaceId()
  try { await submit(entry.inverse); if (workspaceId === activeWorkspaceId()) useGraphStore.temporal.getState().undo() }
  finally { useGraphStore.setState({ busy: false }) }
}
export async function redoGraphEdit(): Promise<void> {
  const temporalState = useGraphStore.temporal.getState()
  const entry = temporalState.futureStates.at(-1)?.steps?.at(-1)
  if (useGraphStore.getState().busy || !entry) return
  useGraphStore.setState({ busy: true })
  const workspaceId = activeWorkspaceId()
  try { await submit(entry.forward); if (workspaceId === activeWorkspaceId()) temporalState.redo() }
  finally { useGraphStore.setState({ busy: false }) }
}
export function resetGraphHistory(): void {
  useGraphStore.setState({ steps: [], selectedIds: [] })
  useGraphStore.temporal.getState().clear()
}
