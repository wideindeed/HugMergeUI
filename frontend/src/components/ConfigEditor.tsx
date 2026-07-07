interface Props {
  value: string
  onChange: (value: string) => void
  onAnalyze: () => void
  loading: boolean
}

export function ConfigEditor({ value, onChange, onAnalyze, loading }: Props) {
  return (
    <section className="panel">
      <h2>Merge config</h2>
      <textarea
        className="config-editor"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        rows={16}
      />
      <button onClick={onAnalyze} disabled={loading}>
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>
    </section>
  )
}
