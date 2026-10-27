import { useEffect, useState } from 'react'
import { useTheme } from '../theme'
import { connectOpenRouter, getLlmSettings, MODEL_DEFAULTS, saveLlmSettings, type LlmProvider } from '../utils/settings'
import { errorMessage } from '../utils/api'

interface Props { onClose: () => void; hosted?: boolean; required?: boolean }
interface Model { id: string; name: string; architecture?: { input_modalities?: string[] }; pricing?: { prompt?: string; completion?: string } }

export default function SettingsModal({ onClose, hosted = true, required = false }: Props) {
  const { theme } = useTheme()
  const dark = theme === 'dark'
  const [settings, setSettings] = useState(getLlmSettings)
  const [models, setModels] = useState<Model[]>([])
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const secure = window.isSecureContext || ['localhost', '127.0.0.1'].includes(window.location.hostname)
  const input = `w-full rounded-lg border p-2 text-sm ${dark ? 'bg-slate-950 border-slate-700' : 'bg-white border-slate-300'}`

  useEffect(() => {
    if (settings.provider !== 'openrouter') return
    const controller = new AbortController()
    fetch('https://openrouter.ai/api/v1/models', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Model list unavailable. You can enter a model ID manually.')
        const result = await response.json() as { data: Model[] }
        setModels(result.data ?? [])
      }).catch(() => { /* Manual model input remains available offline. */ })
    return () => controller.abort()
  }, [settings.provider])

  const chooseProvider = (provider: LlmProvider) => setSettings({ provider, key: '', remember: settings.remember, ...MODEL_DEFAULTS[provider] })
  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      if (!secure) throw new Error('Use HTTPS before entering an API key.')
      if (hosted && (!settings.key.trim() || settings.provider === 'ollama')) throw new Error('Connect OpenRouter or paste an Anthropic key to continue.')
      if (!settings.fastModel.trim() || !settings.smartModel.trim()) throw new Error('Choose a fast model and a smart model.')
      await saveLlmSettings({ ...settings, key: settings.key.trim(), fastModel: settings.fastModel.trim(), smartModel: settings.smartModel.trim() })
      onClose()
    } catch (cause) { setError(errorMessage(cause)) }
    finally { setSaving(false) }
  }
  const lacksVision = settings.provider === 'openrouter' && [settings.fastModel, settings.smartModel].some((id) => {
    const model = models.find((candidate) => candidate.id === id)
    return model && !model.architecture?.input_modalities?.includes('image')
  })

  return <div className="fixed inset-0 z-[70] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-labelledby="settings-title">
    <div className={`lumin-modal-card w-full max-w-lg max-h-[90vh] overflow-auto border rounded-2xl p-6 space-y-5 shadow-2xl ${dark ? 'bg-slate-900 border-slate-700 text-slate-200' : 'bg-white border-slate-200 text-slate-800'}`}>
      <div className="lumin-modal-heading flex justify-between items-center"><h2 id="settings-title" className="font-semibold text-lg">{required ? 'Connect your AI provider' : 'AI settings'}</h2>{!required && <button onClick={onClose} aria-label="Close settings">×</button>}</div>
      <p className="text-sm text-slate-500">Your knowledge bases and original files stay in this browser. The server temporarily processes the materials you use. Your chosen AI provider receives the context needed to answer, using your API key and budget.</p>
      {!secure && <p role="alert" className="text-sm text-red-500">Open this app over HTTPS to connect a provider.</p>}
      <div className="flex gap-2">{(['openrouter', 'anthropic', ...(!hosted ? ['ollama'] as const : [])] as LlmProvider[]).map((provider) => <button key={provider} onClick={() => chooseProvider(provider)} className={`border rounded-lg px-3 py-2 text-sm capitalize ${provider === settings.provider ? 'border-indigo-500 bg-indigo-500/15' : 'border-slate-500/30'}`}>{provider === 'openrouter' ? 'OpenRouter' : provider === 'anthropic' ? 'Anthropic' : 'Ollama'}</button>)}</div>
      {settings.provider === 'openrouter' && <button disabled={!secure} onClick={() => { connectOpenRouter().catch((cause) => setError(errorMessage(cause))) }} className="w-full rounded-xl bg-indigo-600 text-white py-2.5 disabled:opacity-50">Connect OpenRouter</button>}
      {settings.provider !== 'ollama' && <label className="block text-sm space-y-2"><span>{settings.provider === 'anthropic' ? 'Anthropic API key' : 'Or paste an OpenRouter key'}</span><input autoComplete="off" spellCheck={false} type="password" disabled={!secure} value={settings.key} onChange={(event) => setSettings({ ...settings, key: event.target.value })} className={input} placeholder="API key" /></label>}
      <datalist id="lumin-models">{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</datalist>
      <div className="grid gap-4 sm:grid-cols-2">{(['fastModel', 'smartModel'] as const).map((field) => <label key={field} className="block text-sm space-y-2"><span>{field === 'fastModel' ? 'Fast model' : 'Smart model'}</span><input list={settings.provider === 'openrouter' ? 'lumin-models' : undefined} value={settings[field]} onChange={(event) => setSettings({ ...settings, [field]: event.target.value })} className={input} spellCheck={false} /><span className="block text-xs text-slate-500">{field === 'fastModel' ? 'Extraction, routing and reflection' : 'Answers and explanations'}</span></label>)}</div>
      {lacksVision && <p className="text-xs text-amber-500">One of these models does not accept images. Choose a vision model to caption diagrams and image uploads.</p>}
      <label className="flex gap-2 items-start text-sm"><input type="checkbox" checked={settings.remember} onChange={(event) => setSettings({ ...settings, remember: event.target.checked })} className="mt-1 accent-indigo-500" /><span>Remember the key on this device<span className="block text-xs text-slate-500">Stored in this browser. Only enable on a device you trust. Otherwise the key is forgotten when the page closes.</span></span></label>
      {error && <p role="alert" className="text-sm text-red-500">{error}</p>}
      <button onClick={save} disabled={saving || !secure} className="w-full bg-indigo-600 text-white rounded-xl py-2.5 disabled:opacity-40">{saving ? 'Saving…' : 'Save and continue'}</button>
      <p className="text-xs text-slate-500">LuminRAG is open source under AGPL-3.0. <a className="text-indigo-400 underline" href="https://github.com/karaberke/luminrag" target="_blank" rel="noreferrer">View source</a></p>
      {!hosted && required && <button onClick={onClose} className="w-full text-xs text-slate-500">Use self-hosted server settings</button>}
    </div>
  </div>
}
