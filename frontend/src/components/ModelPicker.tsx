import { useSimpleMode } from '../context/SimpleModeContext'

interface Props {
  models: string[]
  baseModelOptions: string[]
  baseModel: string
  modelA: string
  modelB: string
  density: number
  onBaseModelChange: (value: string) => void
  onModelAChange: (value: string) => void
  onModelBChange: (value: string) => void
  onDensityChange: (value: number) => void
  onScore: () => void
  loading: boolean
}

export function ModelPicker({
  models,
  baseModelOptions,
  baseModel,
  modelA,
  modelB,
  density,
  onBaseModelChange,
  onModelAChange,
  onModelBChange,
  onDensityChange,
  onScore,
  loading,
}: Props) {
  const { simple } = useSimpleMode()

  return (
    <section className="panel" data-tour-id="picker">
      <h2>{simple ? 'Set up the comparison' : 'Conflict score inputs'}</h2>
      {simple && (
        <p className="simple-intro">
          The "ancestor" is the shared starting point both models were built from. Model A and Model B are the two
          versions you want to check for clashes.
        </p>
      )}
      <label>
        {simple ? (
          'Ancestor model (the shared starting point)'
        ) : (
          <>
            Base / ancestor model (used as the diff reference — doesn't have to match mergekit's own{' '}
            <code>base_model</code> field)
          </>
        )}
        <select value={baseModel} onChange={(e) => onBaseModelChange(e.target.value)}>
          {baseModelOptions.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>
      <label>
        Model A
        <select value={modelA} onChange={(e) => onModelAChange(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>
      <label>
        Model B
        <select value={modelB} onChange={(e) => onModelBChange(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>
      <label>
        {simple ? `How much of each model to keep: ${density.toFixed(2)}` : `TIES density: ${density.toFixed(2)}`}
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={density}
          onChange={(e) => onDensityChange(Number(e.target.value))}
        />
      </label>
      <button onClick={onScore} disabled={loading}>
        {loading ? (simple ? 'Checking for clashes…' : 'Scoring…') : simple ? 'Check for clashes' : 'Score conflict'}
      </button>
    </section>
  )
}
