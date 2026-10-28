import { MODEL_DEFAULTS, type LlmProvider } from './settings'

export interface ModelOption {
  id: string
  name: string
  architecture?: { input_modalities?: string[]; output_modalities?: string[] }
  supported_parameters?: string[]
}

const option = (id: string, name: string, vision?: boolean): ModelOption => ({
  id, name,
  ...(vision === undefined ? {} : { architecture: { input_modalities: vision ? ['text', 'image'] : ['text'], output_modalities: ['text'] } }),
})
export const MODEL_PRESETS: Record<LlmProvider, ModelOption[]> = {
  openrouter: [
    option(MODEL_DEFAULTS.openrouter.fastModel, 'Anthropic: Claude Haiku 4.5'),
    option(MODEL_DEFAULTS.openrouter.smartModel, 'Anthropic: Claude Sonnet 4.6'),
    option('openai/gpt-5.4-mini', 'OpenAI: GPT-5.4 Mini'),
    option('openai/gpt-5.4', 'OpenAI: GPT-5.4'),
    option('deepseek/deepseek-v4-flash', 'DeepSeek: DeepSeek V4 Flash'),
  ],
  openai: [option('gpt-5.4-mini', 'GPT-5.4 Mini'), option('gpt-5.4', 'GPT-5.4')],
  anthropic: [
    option(MODEL_DEFAULTS.anthropic.fastModel, 'Claude Haiku 5.5', true),
    option(MODEL_DEFAULTS.anthropic.smartModel, 'Claude Sonnet 5.5', true),
    option('claude-haiku-4-5', 'Claude Haiku 4.5', true),
    option('claude-sonnet-4-6', 'Claude Sonnet 4.6', true),
  ],
  ollama_cloud: [
    option(MODEL_DEFAULTS.ollama_cloud.fastModel, 'Gemma 4 31B', true),
    option(MODEL_DEFAULTS.ollama_cloud.smartModel, 'GPT-OSS 120B', false),
    option('gpt-oss:20b', 'GPT-OSS 20B', false),
    option('deepseek-v4.1-flash:1017', 'DeepSeek V4.1 Flash', true),
    option('mistral-large-4', 'Mistral Large 4', true),
    option('minimax-m3.1-flash', 'MiniMax M3.1 Flash', true),
  ],
  ollama: [option('llama3.2', 'Llama 3.2')],
}

let catalog: Promise<ModelOption[]> | undefined
/** Share the public catalog across dialog openings; no provider key is sent. */
export function openRouterModels(): Promise<ModelOption[]> {
  if (catalog) return catalog
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 10_000)
  catalog = fetch('https://openrouter.ai/api/v1/models', { signal: controller.signal, credentials: 'omit' })
    .then(async (response) => {
      if (!response.ok) throw new Error('Model catalog unavailable')
      const result: unknown = await response.json()
      const rows = result && typeof result === 'object' && 'data' in result ? result.data : null
      if (!Array.isArray(rows)) throw new Error('Invalid model catalog')
      const models = new Map<string, ModelOption>()
      for (const row of rows.slice(0, 5000)) {
        if (!row || typeof row !== 'object' || typeof row.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$/.test(row.id) || row.id.endsWith(':batch')) continue
        const inputs = Array.isArray(row.architecture?.input_modalities) ? row.architecture.input_modalities.filter((value: unknown) => typeof value === 'string') as string[] : undefined
        const outputs = Array.isArray(row.architecture?.output_modalities) ? row.architecture.output_modalities.filter((value: unknown) => typeof value === 'string') as string[] : undefined
        if ((inputs && !inputs.includes('text')) || (outputs && (outputs.length !== 1 || outputs[0] !== 'text'))) continue
        models.set(row.id, { id: row.id, name: typeof row.name === 'string' ? row.name.slice(0, 200) : row.id,
          architecture: { input_modalities: inputs, output_modalities: outputs },
          supported_parameters: Array.isArray(row.supported_parameters) ? row.supported_parameters.filter((value: unknown) => typeof value === 'string') : undefined })
      }
      if (!models.size) throw new Error('Empty model catalog')
      const preferred = MODEL_PRESETS.openrouter.map((model) => model.id)
      return [...models.values()].sort((a, b) => {
        const ai = preferred.indexOf(a.id), bi = preferred.indexOf(b.id)
        return (ai < 0 ? preferred.length : ai) - (bi < 0 ? preferred.length : bi) || a.name.localeCompare(b.name)
      })
    }).catch((error: unknown) => { catalog = undefined; throw error })
    .finally(() => clearTimeout(timeout))
  return catalog
}
