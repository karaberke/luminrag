import { useLayoutEffect, useState } from 'react'
import { ThemeContext, type Theme } from './theme'
import { initialTheme } from './utils/appearance'

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  useLayoutEffect(() => { document.documentElement.dataset.theme = theme }, [theme])

  const toggleTheme = () => {
    setTheme((t) => {
      const next = t === 'dark' ? 'light' : 'dark'
      localStorage.setItem('lumin-theme', next)
      return next
    })
  }

  return <ThemeContext.Provider value={{ theme, toggleTheme }}>{children}</ThemeContext.Provider>
}
