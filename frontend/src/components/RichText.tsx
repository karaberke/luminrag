import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { useTheme } from '../ThemeContext'

interface RichTextProps {
  text: string
  className?: string
  onCitationClick?: (n: number) => void
}

// LLMs frequently emit custom operator names that are not KaTeX built-ins.
// Replace them with \operatorname{...} before parsing so KaTeX can render them.
const MATH_ALIASES: [RegExp, string][] = [
  [/\\softmax\b/g,  '\\operatorname{softmax}'],
  [/\\sigmoid\b/g,  '\\operatorname{sigmoid}'],
  [/\\relu\b/g,     '\\operatorname{ReLU}'],
  [/\\ReLU\b/g,     '\\operatorname{ReLU}'],
  [/\\argmax\b/g,   '\\operatorname{arg\\,max}'],
  [/\\argmin\b/g,   '\\operatorname{arg\\,min}'],
  [/\\diag\b/g,     '\\operatorname{diag}'],
  [/\\tr\b/g,       '\\operatorname{tr}'],
  [/\\prox\b/g,     '\\operatorname{prox}'],
  [/\\conv\b/g,     '\\operatorname{conv}'],
  [/\\norm\b/g,     '\\operatorname{norm}'],
]

function sanitizeMath(text: string): string {
  let out = text

  // LLMs often emit [ \latex ] without backslashes on the brackets.
  // Allow optional leading/trailing whitespace and flexible spacing around brackets.
  // Requires content to start with \ so markdown [text](url) links are never affected.
  out = out.replace(/^[ \t]*\[\s+(\\[^\n]+?)\s+\][ \t]*$/gm, '\\[ $1 \\]')

  for (const [pattern, replacement] of MATH_ALIASES) {
    out = out.replace(pattern, replacement)
  }
  return out
}

export function RichText({ text, className, onCitationClick }: RichTextProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  if (!text) return null
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          a: ({ href, children, ...props }) => {
            const citMatch = href?.match(/^#citation-(\d+)$/)
            if (citMatch && onCitationClick) {
              const n = parseInt(citMatch[1], 10)
              return (
                <button
                  onClick={() => onCitationClick(n)}
                  title={`Jump to source [${n}]`}
                  className={`inline font-mono text-xs px-1 py-0.5 rounded border align-baseline transition-colors cursor-pointer ${
                    isDark
                      ? 'border-indigo-600 text-indigo-400 hover:bg-indigo-900/40 hover:text-indigo-300'
                      : 'border-indigo-300 text-indigo-600 hover:bg-indigo-50 hover:text-indigo-800'
                  }`}
                >
                  [{n}]
                </button>
              )
            }
            return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>
          },
        }}
      >
        {sanitizeMath(text)}
      </ReactMarkdown>
    </div>
  )
}
