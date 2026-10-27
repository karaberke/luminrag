/** Keep evidence numbering aligned with the API array; invalid citations have no fallback. */
export function citationIndex(number: number, evidenceCount: number): number | null {
  return Number.isInteger(number) && number >= 1 && number <= evidenceCount ? number - 1 : null
}

export function injectCitationLinks(text: string): string {
  return text.replace(/\[(\d+)\](?!\(|\[)/g, '[$1](#citation-$1)')
}

export function citedEvidenceIndices(text: string, evidenceCount: number): Set<number> {
  const indices = new Set<number>()
  for (const match of text.matchAll(/\[(\d+)\](?!\(|\[)/g)) {
    const index = citationIndex(Number(match[1]), evidenceCount)
    if (index !== null) indices.add(index)
  }
  return indices
}
