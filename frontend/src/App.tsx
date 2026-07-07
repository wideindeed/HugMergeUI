import { useState } from 'react'
import { checkArchitecture, conflictScore, parseConfig } from './api/client'
import type { ArchitectureWarning, ConflictScoreResult } from './api/types'
import { ArchitectureWarnings } from './components/ArchitectureWarnings'
import { ConfigEditor } from './components/ConfigEditor'
import { LayerHeatmap } from './components/LayerHeatmap'
import { ModelPicker } from './components/ModelPicker'
import './App.css'

const DEFAULT_YAML = `merge_method: linear
models:
  - model: Qwen/Qwen2.5-0.5B
    parameters:
      weight: 1.0
  - model: dphn/Dolphin3.0-Qwen2.5-0.5B
    parameters:
      weight: 0.5
  - model: wulli/Qwen2.5-0.5B-sft-capybara
    parameters:
      weight: 0.5
dtype: float32
`

function App() {
  const [yamlText, setYamlText] = useState(DEFAULT_YAML)
  const [analyzing, setAnalyzing] = useState(false)
  const [scoring, setScoring] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [warnings, setWarnings] = useState<ArchitectureWarning[] | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [baseModel, setBaseModel] = useState('')
  const [modelA, setModelA] = useState('')
  const [modelB, setModelB] = useState('')
  const [density, setDensity] = useState(0.5)
  const [scoreResult, setScoreResult] = useState<ConflictScoreResult | null>(null)

  async function handleAnalyze() {
    setAnalyzing(true)
    setError(null)
    setScoreResult(null)
    try {
      const arch = await checkArchitecture(yamlText)
      const parsed = await parseConfig(yamlText, arch.num_layers)
      setWarnings(arch.warnings)
      setModels(parsed.models)
      setBaseModel(parsed.base_model ?? parsed.models[0] ?? '')
      setModelA(parsed.models[0] ?? '')
      setModelB(parsed.models[1] ?? parsed.models[0] ?? '')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setAnalyzing(false)
    }
  }

  async function handleScore() {
    setScoring(true)
    setError(null)
    try {
      const result = await conflictScore(baseModel, modelA, modelB, density)
      setScoreResult(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setScoring(false)
    }
  }

  return (
    <main className="app">
      <h1>HugMergeUI</h1>
      <p className="subtitle">
        Predictive conflict diagnostics for mergekit merges — see sign-conflict
        and redundancy per layer before spending GPU hours on the merge.
      </p>

      {error && <p className="error-banner">{error}</p>}

      <ConfigEditor value={yamlText} onChange={setYamlText} onAnalyze={handleAnalyze} loading={analyzing} />

      {warnings && (
        <section className="panel">
          <h2>Architecture check</h2>
          <ArchitectureWarnings warnings={warnings} />
        </section>
      )}

      {models.length > 0 && (
        <ModelPicker
          models={models}
          baseModel={baseModel}
          modelA={modelA}
          modelB={modelB}
          density={density}
          onBaseModelChange={setBaseModel}
          onModelAChange={setModelA}
          onModelBChange={setModelB}
          onDensityChange={setDensity}
          onScore={handleScore}
          loading={scoring}
        />
      )}

      {scoreResult && <LayerHeatmap result={scoreResult} />}
    </main>
  )
}

export default App
