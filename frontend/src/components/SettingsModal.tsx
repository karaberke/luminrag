import { useEffect, useMemo, useState } from 'react'
import { useTheme } from '../theme'
import { connectOpenRouter, getLlmSettings, MODEL_DEFAULTS, PROVIDER_DETAILS, saveLlmSettings, type LlmProvider } from '../utils/settings'
import { errorMessage } from '../utils/api'
import { MODEL_PRESETS, openRouterModels, type ModelOption } from '../utils/modelCatalog'

interface Props { onClose: () => void; hosted?: boolean; required?: boolean }

export default function SettingsModal({ onClose, hosted = true, required = false }: Props) {
  const { theme } = useTheme()
  const dark = theme === 'dark'
  const [settings, setSettings] = useState(getLlmSettings)
  const [models, setModels] = useState<ModelOption[]>([])
  const [modelSearch, setModelSearch] = useState('')
  const [catalogStatus, setCatalogStatus] = useState<'loading' | 'ready' | 'unavailable'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const secure = window.isSecureContext || ['localhost', '127.0.0.1'].includes(window.location.hostname)
  const input = `w-full rounded-lg border p-2 text-sm ${dark ? 'bg-slate-950 border-slate-700' : 'bg-white border-slate-300'}`

  useEffect(() => {
    if (settings.provider !== 'openrouter') return
    let active = true
    openRouterModels().then((result) => { if (active) { setModels(result); setCatalogStatus('ready') } })
      .catch(() => { if (active) setCatalogStatus('unavailable') })
    return () => { active = false }
  }, [settings.provider])

  const chooseProvider = (provider: LlmProvider) => {
    if (provider !== settings.provider) {
      setSettings({ provider, key: '', ...MODEL_DEFAULTS[provider] })
      setModelSearch('')
      setError(null)
      if (provider === 'openrouter') setCatalogStatus('loading')
    }
  }
  const options = settings.provider === 'openrouter' && models.length ? models : MODEL_PRESETS[settings.provider]
  const matchingModels = useMemo(() => {
    const query = modelSearch.trim().toLowerCase()
    return query ? options.filter((model) => `${model.name} ${model.id}`.toLowerCase().includes(query)) : options
  }, [options, modelSearch])
  // The search covers the entire catalog, while native menus stay quick on phones.
  const shownModels = matchingModels.slice(0, 80)
  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      if (!secure) throw new Error('Use HTTPS before entering an API key.')
      if (settings.provider === 'ollama_cloud' && !settings.key.trim()) throw new Error('Enter your Ollama Cloud API key to continue.')
      if (hosted && (!settings.key.trim() || settings.provider === 'ollama')) throw new Error('Connect OpenRouter, OpenAI, Anthropic or Ollama Cloud to continue.')
      if (!settings.fastModel.trim() || !settings.smartModel.trim()) throw new Error('Choose a fast model and a smart model.')
      await saveLlmSettings({ ...settings, key: settings.key.trim(), fastModel: settings.fastModel.trim(), smartModel: settings.smartModel.trim() })
      onClose()
    } catch (cause) { setError(errorMessage(cause)) }
    finally { setSaving(false) }
  }
  const fastModel = options.find((candidate) => candidate.id === settings.fastModel)
  const lacksVision = fastModel?.architecture?.input_modalities && !fastModel.architecture.input_modalities.includes('image')
  const lacksStructuredOutput = fastModel?.supported_parameters && !fastModel.supported_parameters.some((value) => value === 'structured_outputs' || value === 'response_format')

  return <div className="fixed inset-0 z-[70] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-labelledby="settings-title">
    <div className={`lumin-modal-card w-full max-w-lg max-h-[90vh] overflow-auto border rounded-2xl p-6 space-y-5 shadow-2xl ${dark ? 'bg-slate-900 border-slate-700 text-slate-200' : 'bg-white border-slate-200 text-slate-800'}`}>
      <div className="lumin-modal-heading flex justify-between items-center"><h2 id="settings-title" className="font-semibold text-lg">{required ? 'Connect your AI provider' : 'AI settings'}</h2>{!required && <button onClick={onClose} aria-label="Close settings">×</button>}</div>
      <p className="text-sm text-slate-500">Your knowledge bases and original files stay in this browser. The server temporarily processes the materials you use. Your chosen AI provider receives the context needed to answer, using your API key and budget.</p>
      {!secure && <p role="alert" className="text-sm text-red-500">Open this app over HTTPS to connect a provider.</p>}
      <div className="flex flex-wrap gap-2">{(['openrouter', 'openai', 'anthropic', 'ollama_cloud', ...(!hosted ? ['ollama'] as const : [])] as LlmProvider[]).map((provider) => <button key={provider} onClick={() => chooseProvider(provider)} aria-pressed={provider === settings.provider} className={`border rounded-lg px-3 py-2 text-sm ${provider === settings.provider ? 'border-indigo-500 bg-indigo-500/15' : 'border-slate-500/30'}`}>{PROVIDER_DETAILS[provider].name}</button>)}</div>
      {settings.provider === 'openrouter' && <button disabled={!secure} onClick={() => { connectOpenRouter().catch((cause) => setError(errorMessage(cause))) }} className="w-full rounded-xl bg-indigo-600 text-white py-2.5 disabled:opacity-50">Connect OpenRouter</button>}
      {settings.provider !== 'ollama' && <label className="block text-sm space-y-2"><span>{settings.provider === 'openrouter' ? 'Or paste an OpenRouter key' : `${PROVIDER_DETAILS[settings.provider].name} API key`}</span><input autoComplete="off" spellCheck={false} type="password" disabled={!secure} value={settings.key} onChange={(event) => setSettings({ ...settings, key: event.target.value })} className={input} placeholder="API key" /></label>}
      {settings.provider === 'openai' && <p className="text-xs text-slate-500">Use an API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer" className="underline">OpenAI Platform</a>. API usage is billed separately from a ChatGPT subscription.</p>}
      {settings.provider === 'ollama_cloud' && <p className="text-xs text-slate-500">Use your own key from <a href="https://ollama.com/settings/keys" target="_blank" rel="noreferrer" className="underline">Ollama API keys</a>. Relevant source context is sent to Ollama Cloud and uses your account allowance. No Ollama installation or model download is needed. See the <a href="https://docs.ollama.com/cloud" target="_blank" rel="noreferrer" className="underline">setup guide</a>.</p>}
      <div className="space-y-2"><label className="block text-sm space-y-2"><span>Search models</span><input type="search" aria-label="Search models" placeholder="Search by name or model ID" value={modelSearch} onChange={(event) => setModelSearch(event.target.value)} className={input} /></label>
        <p role="status" className="text-xs text-slate-500">{settings.provider === 'openrouter' && catalogStatus !== 'ready' ? catalogStatus === 'loading' ? 'Loading the OpenRouter catalog. Presets are available now.' : 'The live catalog is unavailable. Use a preset or enter a model ID.' : `${options.length} ${settings.provider === 'openrouter' ? 'text models in the live catalog' : 'presets'}.`}{matchingModels.length > 80 ? ` Showing the first 80 of ${matchingModels.length} matches; search to narrow the list.` : modelSearch && ` ${matchingModels.length} matches.`}</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">{(['fastModel', 'smartModel'] as const).map((field) => <div key={field} className="text-sm space-y-2"><label htmlFor={`lumin-${field}`} className="block">{field === 'fastModel' ? 'Fast model' : 'Smart model'}</label><select aria-label={`Choose ${field === 'fastModel' ? 'fast' : 'smart'} model`} value="" onChange={(event) => { if (event.target.value) setSettings({ ...settings, [field]: event.target.value }) }} className={input}><option value="">Choose from {matchingModels.length} {modelSearch ? 'matches' : 'models'}…</option>{shownModels.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.id}{model.architecture?.input_modalities ? model.architecture.input_modalities.includes('image') ? ' · images' : ' · text only' : ''}</option>)}</select><input id={`lumin-${field}`} aria-label={`${field === 'fastModel' ? 'Fast' : 'Smart'} model ID`} value={settings[field]} onChange={(event) => setSettings({ ...settings, [field]: event.target.value })} className={input} spellCheck={false} /><span className="block text-xs text-slate-500">{field === 'fastModel' ? 'Extraction, routing and reflection' : 'Answers and explanations'}. You can also enter an exact model ID.</span></div>)}</div>
      {lacksVision && <p className="text-xs text-amber-500">This fast model does not accept images. Choose a vision model to caption diagrams and image uploads.</p>}
      {lacksStructuredOutput && <p className="text-xs text-amber-500">This fast model does not advertise structured output. Ingestion may fail; choose a model with JSON output support.</p>}
      {settings.provider === 'ollama_cloud' && <p className="text-xs text-slate-500">Ollama Cloud does not guarantee a fixed response format. LuminRAG checks graph-building responses before saving graph updates.</p>}
      <p className="text-xs text-slate-500">Model access, prices and limits depend on your provider account. A model appearing here does not guarantee access. <a className="underline" href={PROVIDER_DETAILS[settings.provider].modelsUrl} target="_blank" rel="noreferrer">Check models and pricing</a>.</p>
      <p className="text-sm text-slate-500">Your API key stays in this tab’s memory and is sent securely when AI processing is needed. It is never saved. Reconnect after refreshing or reopening the page.</p>
      {error && <p role="alert" className="text-sm text-red-500">{error}</p>}
      <button onClick={save} disabled={saving || !secure} className="w-full bg-indigo-600 text-white rounded-xl py-2.5 disabled:opacity-40">{saving ? 'Applying…' : 'Apply and continue'}</button>
      <p className="text-xs text-slate-500">LuminRAG is open source under AGPL-3.0. <a className="text-indigo-400 underline" href="https://github.com/karaberke/luminrag" target="_blank" rel="noreferrer">View source</a></p>
      {!hosted && required && <button onClick={onClose} className="w-full text-xs text-slate-500">Use self-hosted server settings</button>}
    </div>
  </div>
}
