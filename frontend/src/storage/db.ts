import Dexie, { type EntityTable } from 'dexie'
import type { Message } from '../types'

export interface WorkspaceRecord { id: string; name: string; bundle?: Blob; version: number; updatedAt: number; generation?: number }
export interface RawFileRecord { id: string; workspaceId: string; name: string; source: string; blob: Blob }
export interface SettingRecord { id: string; value: unknown }
export interface ChatRecord { workspaceId: string; messages: Message[] }
interface PendingOriginals { workspaceId: string; generation: number; files: RawFileRecord[]; blockedSources?: string[] }
const PENDING_ORIGINALS_PREFIX = 'pending-originals:'

export const db = new Dexie('luminrag') as Dexie & {
  workspaces: EntityTable<WorkspaceRecord, 'id'>
  rawFiles: EntityTable<RawFileRecord, 'id'>
  chats: EntityTable<ChatRecord, 'workspaceId'>
  settings: EntityTable<SettingRecord, 'id'>
}
// Keep the original store layout so existing browser databases need no destructive migration.
db.version(1).stores({
  workspaces: 'id, updatedAt', rawFiles: 'id, workspaceId, [workspaceId+source]',
  chats: 'workspaceId', quizzes: 'workspaceId', settings: 'id',
})

export const ACTIVE_WORKSPACE_KEY = 'lumin-workspace'
let activeId = ''
let initialization: Promise<void> | undefined
const sourceUrls = new Map<string, string>()
const sourceNames = new Map<string, string>()
const assetUrls = new Map<string, string>()
export interface CachedPassage { id: string; text: string; source_id: string; metadata?: Record<string, unknown> }
const passages = new Map<string, CachedPassage>()
let persistentStorage = false
const generations = new Map<string, number>()
export function workspaceGeneration(id: string): number { return generations.get(id) ?? 0 }
export function hasPersistentStorage(): boolean { return persistentStorage }
export async function requestPersistentStorage(): Promise<boolean> {
  persistentStorage = await navigator.storage?.persist?.() ?? false
  window.dispatchEvent(new Event('lumin-storage-status'))
  return persistentStorage
}
const SOURCE_MIME_TYPES: Record<string, string> = {
  pdf: 'application/pdf', png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp',
  txt: 'text/plain', md: 'text/plain', markdown: 'text/plain',
  mp4: 'video/mp4', mkv: 'video/x-matroska', mov: 'video/quicktime', avi: 'video/x-msvideo',
  mp3: 'audio/mpeg', wav: 'audio/wav', m4a: 'audio/mp4', ogg: 'audio/ogg', flac: 'audio/flac',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', epub: 'application/epub+zip',
}
export function safeSourceBlob(name: string, blob: Blob): Blob {
  // An imported archive controls Blob.type: trust only our extension allowlist.
  return new Blob([blob], { type: SOURCE_MIME_TYPES[name.split('.').at(-1)?.toLowerCase() ?? ''] ?? 'text/plain' })
}

export function activeWorkspaceId(): string { return activeId }

export async function initializeStorage(): Promise<void> {
  if (typeof indexedDB === 'undefined') return
  initialization ??= (async () => {
    const savedId = localStorage.getItem(ACTIVE_WORKSPACE_KEY)
    let workspace = savedId ? await db.workspaces.get(savedId) : undefined
    if (!workspace) workspace = await createWorkspace('My knowledge base')
    activeId = workspace.id
    localStorage.setItem(ACTIVE_WORKSPACE_KEY, activeId)
    await preloadSources()
    try { persistentStorage = await navigator.storage?.persisted?.() ?? false } catch { persistentStorage = false }
    window.dispatchEvent(new Event('lumin-storage-status'))
  })()
  return initialization
}

export async function createWorkspace(name: string): Promise<WorkspaceRecord> {
  const workspace = { id: crypto.randomUUID(), name, version: 0, updatedAt: Date.now(), generation: 0 }
  await db.workspaces.put(workspace)
  return workspace
}

export async function selectWorkspace(id: string): Promise<void> {
  if (!await db.workspaces.get(id)) throw new Error('Knowledge base not found on this device.')
  if (id !== activeId) clearUrls()
  activeId = id
  localStorage.setItem(ACTIVE_WORKSPACE_KEY, id)
  await preloadSources()
  window.dispatchEvent(new CustomEvent('lumin-workspace-changed', { detail: id }))
}

function clearSourceUrls(): void {
  for (const url of new Set(sourceUrls.values())) URL.revokeObjectURL(url)
  sourceUrls.clear()
  sourceNames.clear()
}

function clearUrls(): void {
  clearSourceUrls()
  for (const url of assetUrls.values()) URL.revokeObjectURL(url)
  assetUrls.clear()
  passages.clear()
}

export async function preloadSources(workspaceId = activeId): Promise<void> {
  const files = await db.rawFiles.where('workspaceId').equals(workspaceId).toArray()
  if (workspaceId !== activeId) return
  clearSourceUrls()
  for (const record of files) {
    const safeBlob = safeSourceBlob(record.name, record.blob)
    sourceUrls.set(record.source, URL.createObjectURL(safeBlob))
    sourceNames.set(record.source, record.name)
    sourceUrls.set(record.name, sourceUrls.get(record.source)!)
    sourceUrls.set(record.name.replace(/\.[^.]+$/, ''), sourceUrls.get(record.source)!)
    sourceUrls.set(record.source.replace(/[^\w.-]/g, '_'), sourceUrls.get(record.source)!)
  }
}

export function cachedSourceUrl(source: string): string | undefined { return sourceUrls.get(source) }
export function cachedSourceName(source: string): string | undefined { return sourceNames.get(source) }
export async function originalForEvidence(source: string, filename?: string | null): Promise<RawFileRecord | undefined> {
  const workspaceId = activeId
  const original = await db.rawFiles.where('[workspaceId+source]').equals([workspaceId, source]).first()
  return original ?? (filename ? db.rawFiles.get(`${workspaceId}:${filename}`) : undefined)
}
export function cachedPassage(id: string): CachedPassage | undefined { return passages.get(id) }
export function cachePassages(jsonl: string): void {
  passages.clear()
  for (const line of jsonl.split('\n')) {
    if (!line.trim()) continue
    const passage = JSON.parse(line) as CachedPassage
    if (typeof passage.id === 'string' && typeof passage.text === 'string' && typeof passage.source_id === 'string') passages.set(passage.id, passage)
  }
}
export function cachedAssetUrl(path: string): string | undefined { return assetUrls.get(path.replace(/^\/+/, '')) }
export function isCachedBlobUrl(url: string): boolean { return [...sourceUrls.values(), ...assetUrls.values()].includes(url) }
export function cacheAsset(path: string, blob: Blob): void {
  const key = path.replace(/^\/+/, '')
  const previous = assetUrls.get(key)
  if (previous) URL.revokeObjectURL(previous)
  assetUrls.set(key, URL.createObjectURL(blob))
}

async function originalRecords(files: File[], workspaceId: string): Promise<RawFileRecord[]> {
  const identities = await Promise.all(files.map((file) => sourceIdentity(file.name)))
  return files.map((file, index) => ({
    id: `${workspaceId}:${file.name}`, workspaceId, name: file.name,
    source: identities[index], blob: file,
  }))
}

export async function saveOriginals(files: File[], workspaceId = activeId): Promise<void> {
  await db.rawFiles.bulkPut(await originalRecords(files, workspaceId))
  if (workspaceId === activeId) await preloadSources()
}

/** The browser releases this cross-tab admission lease even if its tab closes. */
export async function withOriginalsLock<T>(workspaceId: string, operation: () => Promise<T>, recoverInterrupted = false): Promise<T> {
  if (!navigator.locks) throw new Error('This browser cannot safely coordinate uploads. Open LuminRAG over HTTPS in a current browser.')
  return navigator.locks.request(`lumin-originals:${workspaceId}`, { ifAvailable: true }, async (lock) => {
    if (!lock) throw new Error('Another tab is uploading or exporting this knowledge base. Wait for it to finish and try again.')
    if (recoverInterrupted) {
      const abandoned = await db.settings.where('id').startsWith(`${PENDING_ORIGINALS_PREFIX}${workspaceId}:attempt:`).primaryKeys()
      if (abandoned.length) {
        await db.settings.bulkDelete(abandoned)
        throw new Error('A previous upload was interrupted before its result could be confirmed. Saved originals were kept. Re-upload its files to finish processing; click Upload & Ingest again when ready.')
      }
    }
    return operation()
  })
}

export async function pendingIngestJobs(workspaceId = activeId): Promise<string[]> {
  const prefix = `${PENDING_ORIGINALS_PREFIX}${workspaceId}:job:`
  return (await db.settings.where('id').startsWith(prefix).primaryKeys()).map((id) => id.slice(prefix.length))
}

export async function hasPendingOriginals(workspaceId = activeId): Promise<boolean> {
  return (await db.settings.where('id').startsWith(`${PENDING_ORIGINALS_PREFIX}${workspaceId}:`).count()) > 0
}

/** Keep an upload's originals separate from committed sources until its job succeeds. */
export async function stageOriginals(files: File[], workspaceId = activeId): Promise<string> {
  const workspace = await db.workspaces.get(workspaceId)
  if (!workspace) throw new Error('This knowledge base is no longer available.')
  const generation = workspace.generation ?? 0
  const pending: PendingOriginals = { workspaceId, generation, files: await originalRecords(files, workspaceId) }
  const id = `${PENDING_ORIGINALS_PREFIX}${workspaceId}:attempt:${crypto.randomUUID()}`
  await db.transaction('rw', db.settings, db.workspaces, async () => {
    const current = await db.workspaces.get(workspaceId)
    if (!current || (current.generation ?? 0) !== generation) throw new Error('The knowledge base changed during upload. Please try again.')
    if (await hasPendingOriginals(workspaceId)) throw new Error('This knowledge base already has an upload awaiting completion. Keep it open to finish processing before uploading more files.')
    await db.settings.put({ id, value: pending })
  })
  return id
}

export async function discardStagedOriginals(id: string): Promise<void> {
  if (id.startsWith(PENDING_ORIGINALS_PREFIX)) await db.settings.delete(id)
}

/** Bind durable staging before App begins polling; this also survives a page reload. */
export async function bindOriginalsToJob(id: string, jobId: string): Promise<void> {
  await db.transaction('rw', db.settings, db.workspaces, async () => {
    const record = await db.settings.get(id)
    const pending = record?.value as PendingOriginals | undefined
    const workspace = pending && await db.workspaces.get(pending.workspaceId)
    if (!pending || !workspace || (workspace.generation ?? 0) !== pending.generation) {
      await discardStagedOriginals(id)
      throw new Error('The knowledge base changed during upload. Please try again.')
    }
    await db.settings.put({ id: `${PENDING_ORIGINALS_PREFIX}${pending.workspaceId}:job:${jobId}`, value: pending })
    await db.settings.delete(id)
  })
}

/** Publish only positively committed source IDs in the same, uncleared knowledge base. */
export async function settleIngestOriginals(jobId: string, ingestedSourceIds: readonly string[] | null | undefined, workspaceId = activeId): Promise<'published' | 'discarded' | 'missing' | 'ambiguous'> {
  const id = `${PENDING_ORIGINALS_PREFIX}${workspaceId}:job:${jobId}`
  const result = await db.transaction('rw', db.settings, db.rawFiles, db.workspaces, async () => {
    const pending = (await db.settings.get(id))?.value as PendingOriginals | undefined
    if (!pending) return 'missing' as const // Older jobs have no staged originals.
    const workspace = await db.workspaces.get(workspaceId)
    const valid = pending.workspaceId === workspaceId && workspace && (workspace.generation ?? 0) === pending.generation
    const committed = new Set(ingestedSourceIds ?? [])
    const candidateSources = new Set(pending.files.filter((file) => committed.has(file.source)).map((file) => file.source))
    const blocked = new Set(pending.blockedSources ?? [])
    // Old clients could admit overlapping jobs. Without publication ordering,
    // keep their existing originals rather than publish either ambiguous revision.
    if (valid) for (const record of await db.settings.where('id').startsWith(`${PENDING_ORIGINALS_PREFIX}${workspaceId}:job:`).toArray()) {
      const other = record.value as PendingOriginals
      if (record.id === id || other.generation !== pending.generation) continue
      const overlaps = other.files.filter((file) => candidateSources.has(file.source)).map((file) => file.source)
      if (!overlaps.length) continue
      for (const source of overlaps) blocked.add(source)
      other.blockedSources = [...new Set([...(other.blockedSources ?? []), ...overlaps])]
      await db.settings.put({ ...record, value: other })
    }
    const files = valid ? pending.files.filter((file) => committed.has(file.source) && !blocked.has(file.source)) : []
    if (files.length) await db.rawFiles.bulkPut(files)
    await db.settings.delete(id)
    if (valid && pending.files.some((file) => blocked.has(file.source) && committed.has(file.source))) return 'ambiguous' as const
    return files.length ? 'published' as const : 'discarded' as const
  })
  if ((result === 'published' || result === 'ambiguous') && workspaceId === activeId) {
    await preloadSources()
    window.dispatchEvent(new Event('lumin-assets-updated'))
  }
  return result
}

/** Matches backend.utils.uploads.source_identity; WebCrypto supplies the digest. */
export async function sourceIdentity(filename: string): Promise<string> {
  const original = filename.normalize('NFC').replace(/\\/g, '/').split('/').at(-1)!
  let safe = original.replace(/\p{C}/gu, '').replace(/[^\p{L}\p{N}_.\- ()[\]]/gu, '_').replace(/^[ .]+|[ .]+$/g, '')
  const suffix = /\.[^.]+$/.exec(safe)?.[0] ?? ''
  const characters = Array.from(safe)
  if (characters.length > 180) safe = characters.slice(0, 180 - Array.from(suffix).length).join('') + suffix
  const stem = suffix ? safe.slice(0, -suffix.length) : safe
  const slug = stem.replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'file'
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(original))), (byte) => byte.toString(16).padStart(2, '0')).join('').slice(0, 12)
  return `${slug}_${suffix.slice(1).toLowerCase() || 'bin'}_${digest}`
}

export async function clearWorkspaceData(id = activeId): Promise<void> {
  generations.set(id, workspaceGeneration(id) + 1)
  await db.transaction('rw', db.rawFiles, db.chats, db.workspaces, db.settings, async () => {
    const workspace = await db.workspaces.get(id)
    await db.rawFiles.where('workspaceId').equals(id).delete()
    await db.chats.delete(id)
    await db.settings.filter((record) => record.id.startsWith(`${PENDING_ORIGINALS_PREFIX}${id}:`)).delete()
    await db.workspaces.update(id, { bundle: undefined, version: 0, updatedAt: Date.now(), generation: (workspace?.generation ?? 0) + 1 })
  })
  if (id === activeId) clearUrls()
}
