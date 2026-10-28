import type { ReflectionVerdict } from '../types'
import { useTheme } from '../theme'

interface Props {
  verdict: ReflectionVerdict
}

function Flag({ label, value, isDark }: { label: string; value: boolean; isDark: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-mono ${
        value
          ? isDark
            ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700'
            : 'bg-emerald-50 text-emerald-700 border border-emerald-300'
          : isDark
            ? 'bg-red-900/60 text-red-300 border border-red-700'
            : 'bg-red-50 text-red-600 border border-red-300'
      }`}
    >
      {value ? '✓' : '✗'} {label}
    </span>
  )
}

export default function ReflectionBadge({ verdict }: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  // Older backups used optimistic placeholders when Fast mode skipped reflection.
  const skipped = verdict.verification_status === 'skipped' || /fast mode:\s*reflection skipped/i.test(verdict.reasoning)
  const unavailable = verdict.verification_status === 'unavailable' || /reflection unavailable/i.test(verdict.reasoning)
  const unverified = skipped || unavailable
  const verificationMessage = skipped
    ? /fast mode:/i.test(verdict.reasoning)
      ? 'Verification was not run in Fast mode. Check the cited passages, or use Thorough mode for an AI review.'
      : verdict.reasoning
    : 'AI verification was unavailable. Check the cited passages before relying on this answer.'

  return (
    <div className={`mt-3 rounded-lg border p-3 ${
      isDark ? 'border-slate-700 bg-slate-800/50' : 'border-slate-200 bg-slate-50'
    }`}>
      <p className={`text-xs mb-2 font-semibold uppercase tracking-wider ${
        isDark ? 'text-slate-500' : 'text-slate-400'
      }`}>
        {unverified ? 'Answer verification' : 'Self-RAG Reflection'}
      </p>
      {unverified ? <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>{verificationMessage}</p> : <><div className="flex flex-wrap gap-1.5 mb-2">
        <Flag label="RELEVANT" value={verdict.is_relevant} isDark={isDark} />
        <Flag label="SUPPORTED" value={verdict.is_supported} isDark={isDark} />
        <Flag label="USEFUL" value={verdict.is_useful} isDark={isDark} />
      </div>
      <p className={`text-xs italic ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>{verdict.reasoning}</p></>}
    </div>
  )
}
