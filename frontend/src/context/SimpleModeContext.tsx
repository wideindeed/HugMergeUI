import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

type Mode = 'expert' | 'explorer'
const STORAGE_KEY = 'hugmergeui-mode'

interface Ctx {
  mode: Mode
  simple: boolean
  toggle: () => void
}

const SimpleModeContext = createContext<Ctx | null>(null)

export function SimpleModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'explorer' ? 'explorer' : 'expert'
    } catch {
      return 'expert'
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, mode)
    } catch {
      // localStorage unavailable (private mode, etc.), mode just won't persist
    }
  }, [mode])

  const value: Ctx = {
    mode,
    simple: mode === 'explorer',
    toggle: () => setMode((m) => (m === 'expert' ? 'explorer' : 'expert')),
  }

  return <SimpleModeContext.Provider value={value}>{children}</SimpleModeContext.Provider>
}

export function useSimpleMode(): Ctx {
  const ctx = useContext(SimpleModeContext)
  if (!ctx) throw new Error('useSimpleMode must be used within SimpleModeProvider')
  return ctx
}
