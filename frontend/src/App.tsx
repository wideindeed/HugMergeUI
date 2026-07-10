import { useRef, useState } from 'react'
import { checkArchitecture, parseConfig, streamConflictScore } from './api/client'
import type { ArchitectureWarning, ConflictScoreResult, ScoreProgressEvent } from './api/types'
import { ArchitectureWarnings } from './components/ArchitectureWarnings'
import { ConfigEditor } from './components/ConfigEditor'
import { GuideTour } from './components/GuideTour'
import { ModelPicker } from './components/ModelPicker'
import { ResultsPanel } from './components/ResultsPanel'
import { useSimpleMode } from './context/SimpleModeContext'
import { EXAMPLE_PAIRS } from './data/examplePairs'
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
  const { simple, toggle } = useSimpleMode()
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
  const [progress, setProgress] = useState<{ percent: number; label: string } | null>(null)
  const [resultsSession, setResultsSession] = useState(0)
  const [tourActive, setTourActive] = useState(false)
  const resolveCount = useRef(0)
  const loadCount = useRef(0)

  async function handleAnalyze(yamlOverride?: string) {
    const text = yamlOverride ?? yamlText
    setAnalyzing(true)
    setError(null)
    setScoreResult(null)
    try {
      const arch = await checkArchitecture(text)
      const parsed = await parseConfig(text, arch.num_layers)
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

  function handleExampleClick(yaml: string) {
    setYamlText(yaml)
    void handleAnalyze(yaml)
  }

  function handleProgress(event: ScoreProgressEvent) {
    if (event.stage === 'resolve') {
      resolveCount.current += 1
      setProgress({ percent: (resolveCount.current / 3) * 15, label: `Resolving ${event.repo}` })
    } else if (event.stage === 'load') {
      loadCount.current += 1
      setProgress({ percent: 15 + (loadCount.current / 3) * 15, label: `Loading weights: ${event.repo}` })
    } else if (event.stage === 'scoring') {
      const pct = 30 + (event.tensor_index / event.tensor_total) * 70
      setProgress({ percent: pct, label: `Scoring tensor ${event.tensor_index}/${event.tensor_total}` })
    }
  }

  async function handleScore() {
    setScoring(true)
    setError(null)
    setResultsSession((s) => s + 1)
    resolveCount.current = 0
    loadCount.current = 0
    setProgress({ percent: 0, label: 'Starting…' })
    try {
      const result = await streamConflictScore(baseModel, modelA, modelB, density, handleProgress)
      setScoreResult(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setScoring(false)
      setProgress(null)
    }
  }

  function scrollToSection(id: string) {
    document.querySelector(`[data-tour-id="${id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>HugMergeUI</h1>
          <p className="subtitle">
            {simple
              ? "A tool that peeks inside two AI models before you mix them, so you're not surprised afterward."
              : 'Predictive conflict diagnostics for mergekit merges.'}
          </p>
        </div>
        <button
          type="button"
          className="mode-toggle"
          onClick={toggle}
          role="switch"
          aria-checked={simple}
          title={simple ? 'Switch to Expert mode' : 'Switch to Explorer mode'}
        >
          <span className={simple ? undefined : 'mode-toggle-active'}>🔬 Expert</span>
          <span className={simple ? 'mode-toggle-active' : undefined}>🪐 Explorer</span>
        </button>
        <nav className="sidebar-nav">
          <button type="button" onClick={() => scrollToSection('examples')}>
            Try an example
          </button>
          <button type="button" onClick={() => scrollToSection('editor')}>
            Merge config
          </button>
          <button type="button" onClick={() => scrollToSection('architecture')} disabled={!warnings}>
            Architecture check
          </button>
          <button type="button" onClick={() => scrollToSection('picker')} disabled={models.length === 0}>
            Score inputs
          </button>
          <button type="button" onClick={() => scrollToSection('results')} disabled={!scoring && !scoreResult}>
            Results
          </button>
        </nav>
        <button type="button" className="guide-button" onClick={() => setTourActive(true)}>
          Guide me
        </button>
      </aside>

      <main className="main-content">
        {error && <p className="error-banner">{error}</p>}

        <div className="workflow-col">
          <section className="panel" data-tour-id="examples">
            <h2>Try an example</h2>
            {simple && (
              <p className="simple-intro">
                Pick a pair below — think of it as picking two ingredients to see if they mix well before you
                commit to the recipe.
              </p>
            )}
            <div className="example-grid">
              {EXAMPLE_PAIRS.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  className="example-card"
                  onClick={() => handleExampleClick(ex.yaml)}
                  disabled={analyzing}
                >
                  <span className="example-label">{ex.label}</span>
                  <span className="example-note">{simple ? ex.simpleNote : ex.note}</span>
                </button>
              ))}
            </div>
          </section>

          <ConfigEditor
            value={yamlText}
            onChange={setYamlText}
            onAnalyze={() => handleAnalyze()}
            loading={analyzing}
          />

          {warnings && (
            <section className="panel" data-tour-id="architecture">
              <h2>{simple ? 'Do these two fit together?' : 'Architecture check'}</h2>
              <ArchitectureWarnings warnings={warnings} />
            </section>
          )}
        </div>

        <div className="insights-col">
          {models.length > 0 && (
            <ModelPicker
              models={Array.from(new Set(models))}
              baseModelOptions={Array.from(new Set(baseModel ? [baseModel, ...models] : models))}
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

          {(scoring || scoreResult) && (
            <ResultsPanel key={resultsSession} scoring={scoring} progress={progress} scoreResult={scoreResult} />
          )}

          {models.length === 0 && !scoring && !scoreResult && (
            <div className="insights-placeholder">
              <p>
                {simple
                  ? 'Pick an example on the left to see what happens when two AI models get mixed together.'
                  : 'Score inputs and results will appear here once you analyze a config.'}
              </p>
            </div>
          )}
        </div>
      </main>

      <GuideTour active={tourActive} onClose={() => setTourActive(false)} />
    </div>
  )
}

export default App
