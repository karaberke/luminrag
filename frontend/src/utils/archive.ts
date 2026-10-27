import { Unzip, UnzipInflate, unzipSync } from 'fflate'

export interface ArchiveLimits {
  maxArchiveBytes: number
  maxExpandedBytes: number
  maxEntryBytes: number
  maxEntries: number
  entryLimit?: (name: string) => number
  limitMessage: string
  pathMessage: string
}

const INPUT_SLICE_BYTES = 1024

/** Decode bounded actual bytes once; callers use these entries rather than the original ZIP again. */
export function readBoundedArchive(bytes: Uint8Array, limits: ArchiveLimits): Record<string, Uint8Array> {
  if (bytes.byteLength > limits.maxArchiveBytes) throw new Error(limits.limitMessage)
  const declared = new Map<string, number>()
  let declaredTotal = 0
  // Read only central-directory metadata here: no entry is expanded by unzipSync.
  unzipSync(bytes, { filter: (entry) => {
    if (entry.name.includes('..') || entry.name.startsWith('/') || entry.name.includes('\\') || /^[A-Za-z]:/.test(entry.name) || /\p{C}/u.test(entry.name) || entry.name.length > 4096) throw new Error(limits.pathMessage)
    if (declared.has(entry.name)) throw new Error('The archive contains duplicate entries.')
    declaredTotal += entry.originalSize
    const entryLimit = Math.min(limits.maxEntryBytes, limits.entryLimit?.(entry.name) ?? limits.maxEntryBytes)
    if (declared.size >= limits.maxEntries || declaredTotal > limits.maxExpandedBytes || entry.originalSize > entryLimit) throw new Error(limits.limitMessage)
    if (entry.compression !== 0 && entry.compression !== 8) throw new Error('The archive uses an unsupported compression method.')
    declared.set(entry.name, entry.originalSize)
    return false
  } })

  let actualTotal = 0
  const visited = new Set<string>()
  const entries: Record<string, Uint8Array> = Object.create(null)
  const stream = new Unzip((file) => {
    if (!declared.has(file.name) || visited.has(file.name)) throw new Error('The archive contains inconsistent entries.')
    visited.add(file.name)
    const expectedSize = declared.get(file.name)!
    const entryLimit = Math.min(limits.maxEntryBytes, limits.entryLimit?.(file.name) ?? limits.maxEntryBytes)
    const chunks: Uint8Array[] = []
    let actualSize = 0
    file.ondata = (error, chunk, final) => {
      if (error) throw error
      actualSize += chunk.length; actualTotal += chunk.length
      if (actualSize > entryLimit || actualTotal > limits.maxExpandedBytes) {
        file.terminate()
        throw new Error(limits.limitMessage)
      }
      // Fail on the first excessive output chunk, even when ZIP headers claim a tiny size.
      if (actualSize > expectedSize || final && actualSize !== expectedSize) {
        file.terminate()
        throw new Error('The archive contains an incorrect expanded size.')
      }
      chunks.push(chunk)
      if (final) {
        const content = new Uint8Array(actualSize)
        let offset = 0
        for (const part of chunks) { content.set(part, offset); offset += part.length }
        chunks.length = 0
        entries[file.name] = content
      }
    }
    file.start()
  })
  stream.register(UnzipInflate)
  // Bound each decoder output allocation as well as the cumulative output retained above.
  for (let offset = 0; offset < bytes.length; offset += INPUT_SLICE_BYTES) stream.push(bytes.subarray(offset, offset + INPUT_SLICE_BYTES), offset + INPUT_SLICE_BYTES >= bytes.length)
  if (Object.keys(entries).length !== declared.size || visited.size !== declared.size) throw new Error('The archive contains inconsistent entries.')
  return entries
}
