import { useEffect, useMemo, useState } from 'react'
import { getCuratedModels } from '../api/client'
import type { CuratedModel } from '../api/types'

interface Props {
  targetLabel: string
  onSelect: (modelId: string) => void
  onClose: () => void
}

function formatParams(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)}M`
  return String(n)
}

function formatDownloads(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

const FAMILY_LABEL: Record<string, string> = {
  qwen2: 'Qwen2',
  llama: 'Llama',
  stablelm: 'StableLM',
}

export function ModelBrowserModal({ targetLabel, onSelect, onClose }: Props) {
  const [models, setModels] = useState<CuratedModel[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState<string | 'all'>('all')

  useEffect(() => {
    getCuratedModels()
      .then(setModels)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  const families = useMemo(() => {
    if (!models) return []
    return [...new Set(models.map((m) => m.model_type))].sort()
  }, [models])

  const filtered = useMemo(() => {
    if (!models) return []
    const q = query.trim().toLowerCase()
    return models.filter((m) => {
      if (family !== 'all' && m.model_type !== family) return false
      if (q && !m.id.toLowerCase().includes(q)) return false
      return true
    })
  }, [models, query, family])

  return (
    <div className="model-browser-overlay" onClick={onClose}>
      <div className="model-browser-panel" onClick={(e) => e.stopPropagation()}>
        <div className="model-browser-header">
          <div>
            <h3>Browse validated models</h3>
            <p>
              Picking for {targetLabel}. Every model here is in a validated zone: 1-3B for
              qwen2/llama/stablelm, or 0.4-0.65B for qwen2 specifically (VALIDATION.txt, Round
              Sixteen).
            </p>
          </div>
          <button type="button" className="model-browser-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="model-browser-controls">
          <input
            type="text"
            placeholder="Filter by name…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          <div className="model-browser-family-filter">
            <button type="button" className={family === 'all' ? 'active' : ''} onClick={() => setFamily('all')}>
              All
            </button>
            {families.map((f) => (
              <button type="button" key={f} className={family === f ? 'active' : ''} onClick={() => setFamily(f)}>
                {FAMILY_LABEL[f] ?? f}
              </button>
            ))}
          </div>
        </div>

        <div className="model-browser-body">
          {error && <p className="error-banner">{error}</p>}
          {!models && !error && <p className="model-browser-loading">Fetching the validated model pool from the Hub…</p>}
          {models && filtered.length === 0 && <p className="model-browser-loading">No models match that filter.</p>}
          {models && filtered.length > 0 && (
            <div className="model-browser-grid">
              {filtered.map((m) => (
                <button type="button" key={m.id} className="model-card" onClick={() => onSelect(m.id)}>
                  <span className="model-card-id">{m.id}</span>
                  <span className="model-card-stats">
                    <span className={`model-card-badge family-${m.model_type}`}>{FAMILY_LABEL[m.model_type] ?? m.model_type}</span>
                    <span className="model-card-param">{formatParams(m.total_params)}</span>
                    <span className="model-card-downloads">{formatDownloads(m.downloads)} dl/mo</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
