import { useEffect, useRef, useState } from 'react'
import { activeWorkspaceId, createWorkspace, db, hasPersistentStorage, requestPersistentStorage, selectWorkspace, type WorkspaceRecord } from '../storage/db'
import { exportWorkspace, importWorkspace, prepareWorkspaceSwitch } from '../utils/workspaces'
import { errorMessage } from '../utils/api'

export default function WorkspaceToolbar({ onSettings, disabled = false }: { onSettings: () => void; disabled?: boolean }) {
  const [workspaces, setWorkspaces] = useState<WorkspaceRecord[]>([])
  const [current, setCurrent] = useState(activeWorkspaceId)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [newName, setNewName] = useState<string | null>(null)
  const [protectedStorage, setProtectedStorage] = useState(hasPersistentStorage)
  const fileRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    const refresh = () => { db.workspaces.toArray().then(setWorkspaces).catch(() => {}); setCurrent(activeWorkspaceId()) }
    const storageStatus = () => setProtectedStorage(hasPersistentStorage())
    refresh()
    window.addEventListener('lumin-workspace-changed', refresh)
    window.addEventListener('lumin-workspaces-updated', refresh)
    window.addEventListener('lumin-storage-status', storageStatus)
    return () => { window.removeEventListener('lumin-workspace-changed', refresh); window.removeEventListener('lumin-workspaces-updated', refresh); window.removeEventListener('lumin-storage-status', storageStatus) }
  }, [])
  const run = async (operation: () => Promise<void>, switching = false) => {
    setBusy(true); setError(null)
    const previousId = activeWorkspaceId()
    if (switching) window.dispatchEvent(new Event('lumin-workspace-changing'))
    try { if (switching) await prepareWorkspaceSwitch(previousId); await operation() } catch (cause) { setError(errorMessage(cause)) }
    finally { setBusy(false); if (switching && previousId === activeWorkspaceId()) window.dispatchEvent(new Event('lumin-workspace-transition-ended')) }
  }
  const button = 'text-xs px-2 py-1.5 rounded-lg border border-slate-500/30 hover:border-indigo-500 disabled:opacity-40'
  return <div className="lumin-workspace-toolbar flex items-center gap-2 px-4 py-2 border-b border-slate-500/20 flex-wrap">
    <select aria-label="Knowledge base" value={current} disabled={busy || disabled} onChange={(event) => { const id = event.target.value; run(() => selectWorkspace(id), true) }} className={`${button} bg-transparent max-w-48`}>
      {workspaces.map((workspace) => <option key={workspace.id} value={workspace.id} className="text-slate-900">{workspace.name}</option>)}
    </select>
    <button className={button} disabled={busy || disabled} onClick={() => setNewName('New knowledge base')}>New knowledge base</button>
    <button className={button} disabled={busy || disabled} onClick={() => { run(exportWorkspace) }}>Export .lumin</button>
    <button className={button} disabled={busy || disabled} onClick={() => fileRef.current?.click()}>Import .lumin</button>
    <input ref={fileRef} hidden type="file" accept=".lumin,.zip" disabled={busy || disabled} onChange={(event) => { const file = event.target.files?.[0]; if (file && !busy && !disabled) run(() => importWorkspace(file), true); event.target.value = '' }} />
    <span className="text-[10px] text-slate-500 ml-auto">{busy ? 'Working…' : protectedStorage ? 'Protected local storage · export a backup' : 'Browser may clear data · export a backup'}</span>
    {!protectedStorage && <button className={button} disabled={busy} onClick={() => { run(async () => { await requestPersistentStorage() }) }}>Protect browser storage</button>}
    <button className={button} onClick={onSettings}>AI settings</button>
    {error && <p role="alert" className="w-full text-xs text-red-500">{error}</p>}
    {newName !== null && <form className="w-full flex gap-2 items-center" onSubmit={(event) => { event.preventDefault(); const name = newName.trim(); if (!name || busy || disabled) return; run(async () => { const workspace = await createWorkspace(name.slice(0, 100)); await selectWorkspace(workspace.id); setNewName(null) }, true) }}><input autoFocus aria-label="Knowledge base name" value={newName} disabled={busy || disabled} maxLength={100} onChange={(event) => setNewName(event.target.value)} className="rounded-lg border border-slate-500/30 px-2 py-1 bg-transparent text-xs" /><button className={button} disabled={busy || disabled || !newName.trim()}>Create</button><button type="button" className={button} onClick={() => setNewName(null)}>Cancel</button></form>}
  </div>
}
