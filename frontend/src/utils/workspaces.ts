import { zipSync, strFromU8, strToU8 } from 'fflate'
import { readBoundedArchive, type ArchiveLimits } from './archive'
import { activeWorkspaceId, cacheAsset, cachePassages, createWorkspace, db, hasPendingOriginals, initializeStorage, preloadSources, selectWorkspace, withOriginalsLock, workspaceGeneration } from '../storage/db'
import { apiJson, apiRequest, registerHydrator } from './api'

const hydration = new Map<string, Promise<void>>()
const saves = new Map<string, ReturnType<typeof setTimeout>>()
const saveQueues = new Map<string, Promise<void>>()
const MAX_BUNDLE_BYTES = 200 * 1024 * 1024
const MAX_EXPANDED_BYTES = 500 * 1024 * 1024
const BUNDLE_ARCHIVE_LIMITS: ArchiveLimits = {
  maxArchiveBytes: MAX_BUNDLE_BYTES, maxExpandedBytes: MAX_EXPANDED_BYTES,
  maxEntryBytes: MAX_BUNDLE_BYTES, maxEntries: 21_001, // Backend entries + 1,000 browser originals + browser metadata.
  limitMessage: 'The expanded backup is too large.', pathMessage: 'The backup contains an unsafe path.',
}
const MIME_TYPES: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', pdf: 'application/pdf', txt: 'text/plain' }

function unpackBundle(blob: Blob): Promise<Record<string, Uint8Array>> {
  if (blob.size > MAX_BUNDLE_BYTES) throw new Error('The backup is larger than 200 MB.')
  return blob.arrayBuffer().then((buffer) => readBoundedArchive(new Uint8Array(buffer), BUNDLE_ARCHIVE_LIMITS))
}

export async function loadBundleAssets(blob: Blob, workspaceId = activeWorkspaceId()): Promise<void> {
  const entries = await unpackBundle(blob)
  cacheBundleAssets(entries, workspaceId)
}
function cacheBundleAssets(entries: Record<string, Uint8Array>, workspaceId: string): void {
  if (workspaceId !== activeWorkspaceId()) return
  cachePassages(entries['chunks.jsonl'] ? strFromU8(entries['chunks.jsonl']) : '')
  for (const [path, bytes] of Object.entries(entries)) {
    if (!path.startsWith('assets/')) continue
    const type = MIME_TYPES[path.split('.').pop()?.toLowerCase() ?? ''] ?? 'application/octet-stream'
    const asset = new Blob([new Uint8Array(bytes)], { type })
    cacheAsset(path, asset)
    cacheAsset(path.replace('assets/', 'static/'), asset)
  }
  window.dispatchEvent(new Event('lumin-assets-updated'))
}

export function hydrateWorkspace(id: string): Promise<void> {
  if (hydration.has(id)) return hydration.get(id)!
  const promise = (async () => {
    const record = await db.workspaces.get(id)
    if (!record) throw new Error('This knowledge base is no longer available on this device.')
    const form = new FormData()
    if (record.bundle) form.append('file', record.bundle, 'workspace.lumin')
    else form.append('empty', 'true')
    const status = await apiJson<{ version: number }>('/api/workspace/import', { method: 'POST', body: form, workspaceId: id, skipHydrate: true })
    if (Number.isSafeInteger(status.version) && status.version >= 0) await db.workspaces.update(id, { version: status.version })
    if (record.bundle && id === activeWorkspaceId()) await loadBundleAssets(record.bundle)
  })().finally(() => hydration.delete(id))
  hydration.set(id, promise)
  return promise
}
registerHydrator(hydrateWorkspace)

export async function restoreSelectedWorkspace(previousWorkspace: string | null, hosted: boolean): Promise<string> {
  // Storage modules can be recreated by HMR while React keeps its old refs.
  // An uninitialized empty activeId must never count as a workspace switch.
  await initializeStorage()
  const id = activeWorkspaceId()
  await ensureWorkspace(!hosted && previousWorkspace !== null && previousWorkspace !== id, hosted)
  return id
}

export async function ensureWorkspace(forceImport = false, hosted = true): Promise<void> {
  await initializeStorage()
  const id = activeWorkspaceId()
  const status = await apiJson<{ loaded: boolean; version: number }>('/api/workspace/status', { skipHydrate: true, workspaceId: id })
  if (id !== activeWorkspaceId()) return
  if (!status.loaded || forceImport) { await hydrateWorkspace(id); await saveWorkspaceBundle(id) }
  else {
    const record = await db.workspaces.get(id)
    if (record && !hosted) {
      // Persistent local data survives a server restart, but its in-memory
      // revision counter does not. Refresh from that authoritative server;
      // never replace it with an older browser bundle on ordinary reconnect.
      await saveWorkspaceBundle(id, { resetFromVersion: record.version })
    }
    else if (record?.bundle) await loadBundleAssets(record.bundle, id)
    else if (record) await saveWorkspaceBundle(record.id)
  }
}

export function saveWorkspaceBundle(id = activeWorkspaceId(), options?: { resetFromVersion: number }): Promise<void> {
  const generation = workspaceGeneration(id)
  const operation = (saveQueues.get(id) ?? Promise.resolve()).catch(() => {}).then(() => writeWorkspaceSnapshot(id, generation, options?.resetFromVersion)).finally(() => {
    if (saveQueues.get(id) === operation) saveQueues.delete(id)
  })
  saveQueues.set(id, operation)
  return operation
}

async function writeWorkspaceSnapshot(id: string, generation: number, resetFromVersion?: number): Promise<void> {
  // A self-hosted server has one local workspace; never export it into an inactive browser workspace.
  if (id !== activeWorkspaceId() || generation !== workspaceGeneration(id)) return
  const durableGeneration = (await db.workspaces.get(id))?.generation ?? 0
  const response = await apiRequest('/api/workspace/export', { workspaceId: id, skipHydrate: true, cache: 'no-store' })
  const bundle = await response.blob()
  const entries = await unpackBundle(bundle)
  if (!entries['manifest.json'] || !entries['graph.json']) throw new Error('The server returned an invalid knowledge base backup.')
  const manifest = JSON.parse(strFromU8(entries['manifest.json']))
  const header = response.headers.get('X-Workspace-Version')
  let version = header !== null ? Number(header) : Number(manifest.workspace_version)
  if (!Number.isSafeInteger(version) || version < 0) version = (await apiJson<{ version: number }>('/api/workspace/status', { workspaceId: id, skipHydrate: true })).version
  if (!Number.isSafeInteger(version) || version < 0) throw new Error('The server returned an invalid backup version.')
  const written = await db.transaction('rw', db.workspaces, async () => {
    const previous = await db.workspaces.get(id)
    if (!previous || id !== activeWorkspaceId() || generation !== workspaceGeneration(id) || (previous.generation ?? 0) !== durableGeneration) return false
    // Only the self-hosted loaded-state restore supplies a reset baseline. If
    // another snapshot changed it meanwhile, retain the normal stale guard.
    if (previous.version > version && previous.version !== resetFromVersion) return false
    await db.workspaces.update(id, { bundle, version, updatedAt: Date.now() })
    return true
  })
  if (!written) return
  cacheBundleAssets(entries, id)
  window.dispatchEvent(new Event('lumin-workspaces-updated'))
}

export function scheduleWorkspaceSave(id = activeWorkspaceId()): void {
  clearTimeout(saves.get(id))
  saves.set(id, setTimeout(() => {
    saves.delete(id)
    saveWorkspaceBundle(id).catch((error) => window.dispatchEvent(new CustomEvent('lumin-storage-error', { detail: error instanceof Error ? error.message : 'Could not save this knowledge base to your browser.' })))
  }, 700))
}

export async function exportWorkspace(): Promise<void> {
  const id = activeWorkspaceId()
  return withOriginalsLock(id, () => exportCommittedWorkspace(id))
}

async function exportCommittedWorkspace(id: string): Promise<void> {
  if (await hasPendingOriginals(id)) throw new Error('Wait for this knowledge base\'s upload to finish before exporting. If an upload was interrupted, reopen Upload and re-upload its files first.')
  await saveWorkspaceBundle(id)
  const record = await db.workspaces.get(id)
  if (!record?.bundle) throw new Error('There is no saved knowledge base to export.')
  const entries = await unpackBundle(record.bundle)
  delete entries['quizzes.json']
  const originals = await db.rawFiles.where('workspaceId').equals(id).toArray()
  const expandedBytes = Object.values(entries).reduce((sum, bytes) => sum + bytes.byteLength, 0) + originals.reduce((sum, original) => sum + original.blob.size, 0)
  if (expandedBytes > MAX_EXPANDED_BYTES) throw new Error('This knowledge base exceeds the 500 MB backup limit. Split the materials into smaller knowledge bases before exporting.')
  const originalManifest: Array<{ path: string; name: string; source: string; type: string }> = []
  for (let index = 0; index < originals.length; index++) {
    const original = originals[index]
    const path = `originals/${index}`
    entries[path] = new Uint8Array(await original.blob.arrayBuffer())
    originalManifest.push({ path, name: original.name, source: original.source, type: original.blob.type })
  }
  entries['browser.json'] = strToU8(JSON.stringify({ name: record.name, originals: originalManifest,
    chats: (await db.chats.get(id))?.messages ?? [] }))
  const archive = zipSync(entries)
  if (Object.values(entries).reduce((sum, bytes) => sum + bytes.byteLength, 0) > MAX_EXPANDED_BYTES || archive.byteLength > MAX_BUNDLE_BYTES) throw new Error('This backup exceeds the 200 MB archive limit. Split the materials into smaller knowledge bases before exporting.')
  const url = URL.createObjectURL(new Blob([new Uint8Array(archive)], { type: 'application/zip' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `${record.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.lumin`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export async function importWorkspace(file: File): Promise<void> {
  const entries = await unpackBundle(file)
  if (!entries['manifest.json'] || !entries['graph.json']) throw new Error('This file is not a LuminRAG backup.')
  const { browserMetadataSchema } = await import('./validation')
  const browser = browserMetadataSchema.parse(entries['browser.json'] ? JSON.parse(strFromU8(entries['browser.json'])) : { name: file.name.replace(/\.lumin$/i, '').slice(0, 100) || 'Imported knowledge base' })
  const record = await createWorkspace(browser.name)
  try {
    // The server bundle only receives compute data; originals and chats remain local.
    const computeEntries = Object.fromEntries(Object.entries(entries).filter(([path]) => path !== 'browser.json' && path !== 'quizzes.json' && !path.startsWith('originals/')))
    const bundle = new Blob([new Uint8Array(zipSync(computeEntries))], { type: 'application/zip' })
    await db.workspaces.update(record.id, { bundle })
    await hydrateWorkspace(record.id)
    for (const original of Array.isArray(browser.originals) ? browser.originals : []) {
      if (typeof original.path !== 'string' || typeof original.name !== 'string' || typeof original.source !== 'string' || !entries[original.path]) continue
      await db.rawFiles.put({ id: `${record.id}:${original.name}`, workspaceId: record.id,
        name: original.name, source: original.source, blob: new Blob([new Uint8Array(entries[original.path])], { type: original.type || 'application/octet-stream' }) })
    }
    if (Array.isArray(browser.chats)) await db.chats.put({ workspaceId: record.id, messages: browser.chats })
    await selectWorkspace(record.id)
    await preloadSources()
    await loadBundleAssets(bundle)
  } catch (error) {
    await db.workspaces.delete(record.id)
    await db.rawFiles.where('workspaceId').equals(record.id).delete()
    throw error
  }
}
