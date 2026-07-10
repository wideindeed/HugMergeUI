import { useSimpleMode } from '../context/SimpleModeContext'

interface Props {
  value: string
  onChange: (value: string) => void
  onAnalyze: () => void
  loading: boolean
}

export function ConfigEditor({ value, onChange, onAnalyze, loading }: Props) {
  const { simple } = useSimpleMode()

  return (
    <section className="panel" data-tour-id="editor">
      <h2>{simple ? 'The recipe' : 'Merge config'}</h2>
      {simple && (
        <p className="simple-intro">
          This is the exact recipe for combining the models — which ones, and how much of each. You don't need to
          read it. Pick an example above, then hit Analyze below.
        </p>
      )}
      {simple ? (
        <details className="config-editor-details">
          <summary>Show the technical recipe</summary>
          <textarea
            className="config-editor"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            spellCheck={false}
            rows={16}
          />
        </details>
      ) : (
        <textarea
          className="config-editor"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          rows={16}
        />
      )}
      <button onClick={onAnalyze} disabled={loading}>
        {loading ? (simple ? 'Checking…' : 'Analyzing…') : simple ? 'Check this pairing' : 'Analyze'}
      </button>
    </section>
  )
}
