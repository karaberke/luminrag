import { createContext, useContext } from 'react'

export type Theme = 'dark' | 'light'

export interface ThemeCtx {
  theme: Theme
  toggleTheme: () => void
}

export const ThemeContext = createContext<ThemeCtx>({ theme: 'light', toggleTheme: () => {} })

export const useTheme = () => useContext(ThemeContext)
