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
  return (
    <section className="panel" data-tour-id="picker">
      <h2>Conflict score inputs</h2>
      <label>
        Base / ancestor model (used as the diff reference — doesn't have to
        match mergekit's own <code>base_model</code> field)
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
        TIES density: {density.toFixed(2)}
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
        {loading ? 'Scoring…' : 'Score conflict'}
      </button>
    </section>
  )
}
