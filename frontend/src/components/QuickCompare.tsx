import { useEffect, useState } from 'react'
import { checkModel } from '../api/client'
import type { ModelCheckResult } from '../api/types'
import { useSimpleMode } from '../context/SimpleModeContext'
import { ModelBrowserModal } from './ModelBrowserModal'
import { ModelSearchInput } from './ModelSearchInput'

type BrowserTarget = 'base' | 'modelA' | 'modelB' | null

interface Props {
  scoring: boolean
  onScore: (baseModel: string, modelA: string, modelB: string, density: number) => void
}

type Check = ModelCheckResult | 'loading'

function zoneBadge(check: Check | undefined) {
  if (!check) return null
  if (check === 'loading') return <span className="zone-badge zone-loading">Checking…</span>
  if (check.error) return <span className="zone-badge zone-error">Couldn't find this model</span>

  switch (check.zone) {
    case 'validated': {
      const isQwenExtended = check.model_type === 'qwen2' && (check.total_params ?? 0) < 1_300_000_000
      const rangeLabel = isQwenExtended ? '0.4-0.65B' : '1-3B'
      return <span className="zone-badge zone-ok">Validated zone ({check.model_type}, {rangeLabel})</span>
    }
    case 'below_range':
      return (
        <span className="zone-badge zone-bad">
          Below the validated range, tested at this size and found not to hold
        </span>
      )
    case 'above_range':
      return <span className="zone-badge zone-warn">Above the validated range, never measured, exploratory</span>
    case 'untested_family':
      return (
        <span className="zone-badge zone-warn">
          Different architecture family ({check.model_type ?? 'unknown'}), never measured, exploratory
        </span>
      )
    default:
      return <span className="zone-badge zone-unknown">Couldn't determine size, treat with caution</span>
  }
}

export function QuickCompare({ scoring, onScore }: Props) {
  const { simple } = useSimpleMode()
  const [baseModel, setBaseModel] = useState('')
  const [modelA, setModelA] = useState('')
  const [modelB, setModelB] = useState('')
  const [density, setDensity] = useState(0.5)
  const [checks, setChecks] = useState<Record<string, Check>>({})
  const [browserTarget, setBrowserTarget] = useState<BrowserTarget>(null)

  function handleBrowserSelect(modelId: string) {
    if (browserTarget === 'base') setBaseModel(modelId)
    else if (browserTarget === 'modelA') setModelA(modelId)
    else if (browserTarget === 'modelB') setModelB(modelId)
    setBrowserTarget(null)
  }

  useEffect(() => {
    const ids = [baseModel, modelA, modelB].filter((id) => id.trim().length > 2)
    const toCheck = ids.filter((id) => !(id in checks))
    if (toCheck.length === 0) return
    const timeout = setTimeout(() => {
      setChecks((prev) => {
        const next = { ...prev }
        for (const id of toCheck) next[id] = 'loading'
        return next
      })
      for (const id of toCheck) {
        checkModel(id)
          .then((result) => setChecks((prev) => ({ ...prev, [id]: result })))
          .catch(() =>
            setChecks((prev) => ({
              ...prev,
              [id]: { model_type: null, total_params: null, validated: null, zone: 'unknown', error: 'lookup failed' },
            })),
          )
      }
    }, 500)
    return () => clearTimeout(timeout)
  }, [baseModel, modelA, modelB, checks])

  function handleScore() {
    onScore(baseModel || modelA, modelA, modelB, density)
  }

  const canScore = modelA.trim().length > 2 && modelB.trim().length > 2 && !scoring

  return (
    <section className="panel" data-tour-id="quick-compare">
      <h2>{simple ? 'Compare any two models' : 'Quick compare'}</h2>
      <p className="simple-intro">
        {simple
          ? "Skip the recipe. Type in any two Hugging Face models and see how they'd clash. Best results are on 1-3B models, that's the range this tool has actually been checked against."
          : 'Type any Hugging Face repo id directly, no merge config needed. drift_magnitude is only validated for 1-3B Qwen2/Llama/StableLM-family models; other pairs will score, but treat the result as exploratory.'}
      </p>
      <div className="quick-compare-fields">
        <div>
          <ModelSearchInput id="qc-base" label="Base / ancestor (optional)" value={baseModel} onChange={setBaseModel} />
          <button type="button" className="browse-button" onClick={() => setBrowserTarget('base')}>
            Browse validated models
          </button>
          {zoneBadge(checks[baseModel])}
        </div>
        <div>
          <ModelSearchInput id="qc-model-a" label="Model A" value={modelA} onChange={setModelA} />
          <button type="button" className="browse-button" onClick={() => setBrowserTarget('modelA')}>
            Browse validated models
          </button>
          {zoneBadge(checks[modelA])}
        </div>
        <div>
          <ModelSearchInput id="qc-model-b" label="Model B" value={modelB} onChange={setModelB} />
          <button type="button" className="browse-button" onClick={() => setBrowserTarget('modelB')}>
            Browse validated models
          </button>
          {zoneBadge(checks[modelB])}
        </div>
      </div>
      <label>
        {simple ? `How much of each model to keep: ${density.toFixed(2)}` : `TIES density: ${density.toFixed(2)}`}
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={density}
          onChange={(e) => setDensity(Number(e.target.value))}
        />
      </label>
      <button type="button" onClick={handleScore} disabled={!canScore}>
        {scoring ? (simple ? 'Checking for clashes…' : 'Scoring…') : simple ? 'Check for clashes' : 'Score conflict'}
      </button>
      {browserTarget && (
        <ModelBrowserModal
          targetLabel={browserTarget === 'base' ? 'Base / ancestor' : browserTarget === 'modelA' ? 'Model A' : 'Model B'}
          onSelect={handleBrowserSelect}
          onClose={() => setBrowserTarget(null)}
        />
      )}
    </section>
  )
}
