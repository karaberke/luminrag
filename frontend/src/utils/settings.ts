import { db, initializeStorage } from '../storage/db'

export type LlmProvider = 'openrouter' | 'anthropic' | 'openai' | 'ollama_cloud' | 'ollama'
export interface LlmSettings { provider: LlmProvider; key: string; fastModel: string; smartModel: string }
export const PROVIDER_DETAILS: Record<LlmProvider, { name: string; modelsUrl: string }> = {
  openrouter: { name: 'OpenRouter', modelsUrl: 'https://openrouter.ai/models' },
  anthropic: { name: 'Anthropic', modelsUrl: 'https://platform.claude.com/docs/en/about-claude/models/overview' },
  openai: { name: 'OpenAI', modelsUrl: 'https://developers.openai.com/api/docs/models' },
  ollama_cloud: { name: 'Ollama Cloud', modelsUrl: 'https://docs.ollama.com/cloud' },
  ollama: { name: 'Ollama', modelsUrl: 'https://ollama.com/library' },
}
export const MODEL_DEFAULTS = {
  openrouter: { fastModel: 'anthropic/claude-haiku-4.5', smartModel: 'anthropic/claude-sonnet-4.6' },
  anthropic: { fastModel: 'claude-haiku-4-5', smartModel: 'claude-sonnet-4-6' },
  openai: { fastModel: 'gpt-5.4-mini', smartModel: 'gpt-5.4' },
  ollama_cloud: { fastModel: 'gemma4:31b', smartModel: 'gpt-oss:120b' },
  ollama: { fastModel: 'llama3.2', smartModel: 'llama3.2' },
}
let current: LlmSettings = { provider: 'openrouter', key: '', ...MODEL_DEFAULTS.openrouter }
let loaded: Promise<void> | undefined
function modelPreferences(value: unknown): Omit<LlmSettings, 'key'> {
  const saved = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const provider = saved.provider === 'anthropic' || saved.provider === 'openai' || saved.provider === 'ollama_cloud' || saved.provider === 'ollama' ? saved.provider : 'openrouter'
  const defaults = MODEL_DEFAULTS[provider]
  return {
    provider,
    fastModel: typeof saved.fastModel === 'string' && saved.fastModel.trim() ? saved.fastModel.trim() : defaults.fastModel,
    smartModel: typeof saved.smartModel === 'string' && saved.smartModel.trim() ? saved.smartModel.trim() : defaults.smartModel,
  }
}
export function getLlmSettings(): LlmSettings { return current }
export async function loadLlmSettings(): Promise<void> {
  if (typeof indexedDB === 'undefined') return
  loaded ??= (async () => {
    await initializeStorage()
    const record = await db.settings.get('llm')
    if (record) {
      const preferences = modelPreferences(record.value)
      // Overwrite legacy saved credentials without ever restoring them to this tab.
      await db.settings.put({ id: 'llm', value: preferences })
      current = { ...preferences, key: current.key }
    }
  })()
  return loaded
}
export async function saveLlmSettings(settings: LlmSettings): Promise<void> {
  await loadLlmSettings()
  const preferences = modelPreferences(settings)
  await db.settings.put({ id: 'llm', value: preferences })
  current = { ...preferences, key: typeof settings.key === 'string' ? settings.key : '' }
  window.dispatchEvent(new Event('lumin-settings-changed'))
}
export function llmHeaders(): Record<string, string> {
  if (!current.key && current.provider !== 'ollama') return {}
  return { 'X-LLM-Provider': current.provider, 'X-LLM-Key': current.key,
    'X-LLM-Model-Fast': current.fastModel, 'X-LLM-Model-Smart': current.smartModel }
}

const PKCE_KEY = 'lumin-openrouter-pkce'
const base64url = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
export async function connectOpenRouter(): Promise<void> {
  const verifier = base64url(crypto.getRandomValues(new Uint8Array(32)))
  const challenge = base64url(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))))
  const callback = new URL(window.location.href)
  callback.search = ''
  callback.hash = ''
  sessionStorage.setItem(PKCE_KEY, JSON.stringify({ verifier, createdAt: Date.now(), callback: callback.href }))
  const url = new URL('https://openrouter.ai/auth')
  url.search = new URLSearchParams({ callback_url: callback.href, code_challenge: challenge, code_challenge_method: 'S256' }).toString()
  window.location.assign(url.href)
}
let callbackPromise: Promise<boolean> | undefined
export function finishOpenRouterConnection(): Promise<boolean> {
  if (callbackPromise) return callbackPromise
  if (!new URL(window.location.href).searchParams.has('code')) return Promise.resolve(false)
  callbackPromise = exchangeOpenRouterCode()
  return callbackPromise
}
async function exchangeOpenRouterCode(): Promise<boolean> {
  const url = new URL(window.location.href)
  const code = url.searchParams.get('code')
  if (!code) return false
  const raw = sessionStorage.getItem(PKCE_KEY)
  sessionStorage.removeItem(PKCE_KEY)
  url.searchParams.delete('code')
  window.history.replaceState(null, '', url.href)
  if (!raw) throw new Error('OpenRouter connection expired. Connect again.')
  const saved = JSON.parse(raw) as { verifier: string; createdAt: number; callback: string }
  if (Date.now() - saved.createdAt > 10 * 60_000 || new URL(saved.callback).origin !== url.origin) throw new Error('OpenRouter connection expired. Connect again.')
  const response = await fetch('https://openrouter.ai/api/v1/auth/keys', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, code_verifier: saved.verifier, code_challenge_method: 'S256' }),
  })
  if (!response.ok) throw new Error('OpenRouter could not complete the connection. Try again.')
  const data = await response.json() as { key?: string }
  if (!data.key) throw new Error('OpenRouter did not return an API key.')
  await saveLlmSettings({ provider: 'openrouter', key: data.key, ...MODEL_DEFAULTS.openrouter })
  return true
}
