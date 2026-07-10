import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { SimpleModeProvider } from './context/SimpleModeContext.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SimpleModeProvider>
      <App />
    </SimpleModeProvider>
  </StrictMode>,
)
