import type { ReactNode } from 'react'
export default function ContextMenu({ left, top, dark, children }: { left: number; top: number; dark: boolean; children: ReactNode }) {
  return <div role="menu" onClick={(event) => event.stopPropagation()} style={{ left: Math.max(8, left), top: Math.max(8, top), maxHeight: '80%' }} className={`absolute z-30 border shadow-xl rounded-xl p-1.5 min-w-40 overflow-auto text-xs ${dark ? 'bg-slate-900 border-slate-600 text-slate-200' : 'bg-white border-slate-200 text-slate-700'}`}>{children}</div>
}
