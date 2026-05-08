import { useTheme } from '../ThemeContext'

const PAPER_BASE = import.meta.env.VITE_PAPER_PDF ?? '/paper.pdf'
const PAPER_TITLE = import.meta.env.VITE_PAPER_TITLE ?? 'Research Paper'
const PAPER_URL = `${PAPER_BASE}${PAPER_BASE.includes('?') ? '&' : '?'}v=${Date.now()}`

export default function PaperView() {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <div className={`flex flex-col flex-1 min-w-0 ${isDark ? 'bg-slate-950' : 'bg-[#f0f4ff]'}`}>
      <div className={`flex items-center gap-2 px-4 py-2 border-b shrink-0 ${
        isDark ? 'border-slate-800 bg-slate-900/60' : 'border-[#dde5f5] bg-white/80'
      }`}>
        <span className={`text-sm font-medium ${isDark ? 'text-slate-200' : 'text-slate-700'}`}>
          {PAPER_TITLE}
        </span>
        <a
          href={PAPER_URL}
          download
          className={`ml-auto text-xs border px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 ${
            isDark
              ? 'text-slate-400 hover:text-slate-200 border-slate-700 hover:border-indigo-600'
              : 'text-slate-600 hover:text-slate-800 border-slate-300 hover:border-indigo-400 bg-white hover:bg-indigo-50'
          }`}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Download
        </a>
        <a
          href={PAPER_URL}
          target="_blank"
          rel="noopener noreferrer"
          className={`text-xs border px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 ${
            isDark
              ? 'text-slate-400 hover:text-slate-200 border-slate-700 hover:border-indigo-600'
              : 'text-slate-600 hover:text-slate-800 border-slate-300 hover:border-indigo-400 bg-white hover:bg-indigo-50'
          }`}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
            <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Open in new tab
        </a>
      </div>
      <iframe
        src={PAPER_URL}
        title={PAPER_TITLE}
        className="flex-1 w-full border-0"
      />
    </div>
  )
}
