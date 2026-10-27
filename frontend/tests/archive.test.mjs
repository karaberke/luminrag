import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createTestServer } from './helpers/vite.mjs'
import { Zip, ZipDeflate, zipSync, strToU8, strFromU8 } from 'fflate'

let server, readBoundedArchive
const limits = { maxArchiveBytes: 2 * 1024 * 1024, maxExpandedBytes: 4 * 1024 * 1024,
  maxEntryBytes: 2 * 1024 * 1024, maxEntries: 20, limitMessage: 'Archive size limit', pathMessage: 'Unsafe archive path' }
before(async () => {
  server = await createTestServer()
  ;({ readBoundedArchive } = await server.ssrLoadModule('/src/utils/archive.ts'))
})
after(async () => { await server?.close() })

function headers(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const central = [], local = []
  let end = -1
  for (let offset = 0; offset < bytes.length - 4; offset++) {
    const signature = view.getUint32(offset, true)
    if (signature === 0x02014b50) central.push(offset)
    if (signature === 0x04034b50) local.push(offset)
    if (signature === 0x06054b50) end = offset
  }
  return { view, central, local, end }
}

function repeatedCentralReferences(duplicateNames = false) {
  const original = zipSync({ 'assets/a0.bin': [new Uint8Array(64 * 1024).fill(65), { level: 0 }] })
  const { central, end } = headers(original)
  const entry = original.slice(central[0], end), copies = 4
  const forged = new Uint8Array(central[0] + entry.length * copies + 22)
  forged.set(original.subarray(0, central[0]))
  for (let index = 0; index < copies; index++) {
    const reference = entry.slice()
    if (!duplicateNames) reference[46 + 'assets/a'.length] = 48 + index
    forged.set(reference, central[0] + index * entry.length)
  }
  forged.set(original.subarray(end), forged.length - 22)
  const tail = new DataView(forged.buffer, forged.length - 22)
  tail.setUint16(8, copies, true); tail.setUint16(10, copies, true); tail.setUint32(12, entry.length * copies, true)
  return forged
}

test('bounded archives preserve stored files, compressed files and valid streamed data descriptors', () => {
  const normal = zipSync({ 'manifest.json': strToU8('{}'), 'assets/plain.txt': [strToU8('Stored passage'), { level: 0 }], 'assets/packed.txt': strToU8('Compressed passage') })
  const entries = readBoundedArchive(normal, limits)
  assert.equal(strFromU8(entries['assets/plain.txt']), 'Stored passage')
  assert.equal(strFromU8(entries['assets/packed.txt']), 'Compressed passage')
  const parts = []
  const stream = new Zip((error, data) => { if (error) throw error; parts.push(data) })
  const file = new ZipDeflate('streamed.txt')
  stream.add(file); file.push(strToU8('A streamed original'), true); stream.end()
  const streamed = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0))
  let offset = 0
  for (const part of parts) { streamed.set(part, offset); offset += part.length }
  assert.equal(strFromU8(readBoundedArchive(streamed, limits)['streamed.txt']), 'A streamed original')
})

test('stored and deflated entries cannot hide actual bytes behind false expanded sizes', () => {
  for (const level of [0, 6]) {
    const bytes = zipSync({ 'assets/payload': [new Uint8Array(64 * 1024).fill(65), { level }] })
    const { view, central, local } = headers(bytes)
    for (const offset of central) view.setUint32(offset + 24, 0, true)
    for (const offset of local) view.setUint32(offset + 22, 0, true)
    assert.throws(() => readBoundedArchive(bytes, limits), /incorrect expanded size/)
    assert.throws(() => readBoundedArchive(bytes, { ...limits, maxEntryBytes: 128, maxExpandedBytes: 128 }), /Archive size limit/)
  }
})

test('repeated central references and duplicate names are rejected without multiplying payload allocations', () => {
  assert.throws(() => readBoundedArchive(repeatedCentralReferences(), limits), /inconsistent entries/)
  assert.throws(() => readBoundedArchive(repeatedCentralReferences(true), limits), /duplicate entries/)
})

test('archive budgets enforce entry count, declared totals, paths and per-entry limits', () => {
  const bytes = zipSync({ first: strToU8('12345'), second: strToU8('67890') })
  for (const policy of [{ maxEntries: 1 }, { maxExpandedBytes: 9 }, { maxEntryBytes: 4 }, { maxArchiveBytes: 1 }]) assert.throws(() => readBoundedArchive(bytes, { ...limits, ...policy }), /Archive size limit/)
  for (const name of ['../escape', '/absolute', 'C:/drive', 'folder\\escape', 'folder/\u0000name']) assert.throws(() => readBoundedArchive(zipSync({ [name]: strToU8('x') }), limits), /Unsafe archive path/)
})

test('central offset discrepancies cannot replace the actual accepted local-file contents', () => {
  const bytes = zipSync({ 'manifest.json': strToU8('Actual local contents') })
  const { view, central } = headers(bytes)
  view.setUint32(central[0] + 42, 1, true)
  assert.equal(strFromU8(readBoundedArchive(bytes, limits)['manifest.json']), 'Actual local contents')
})
