import type { ConflictScoreResult, LayerScore, OtherScore } from '../api/types'

// drift_magnitude is validated (VALIDATION.txt Rounds 4-8) as the metric
// that actually tracks merge quality; conflict is shown as a secondary
// stat only. See the same ceiling rationale in ConflictScene.tsx.
const DRIFT_RISK_CEILING = 0.9

function driftColor(drift: number): string {
  // 0 -> green (low drift risk), ceiling+ -> red (high)
  const shaped = Math.pow(Math.min(Math.max(drift, 0), DRIFT_RISK_CEILING) / DRIFT_RISK_CEILING, 0.6)
  const hue = 120 * (1 - shaped)
  return `hsl(${hue}, 70%, 45%)`
}

function LayerCell({ layer }: { layer: LayerScore }) {
  return (
    <div
      className="layer-cell"
      style={{ backgroundColor: driftColor(layer.drift_magnitude) }}
      title={
        `layer ${layer.layer}\n` +
        `drift magnitude: ${layer.drift_magnitude.toFixed(3)}\n` +
        `conflict: ${layer.conflict.toFixed(3)}\n` +
        `redundancy A: ${layer.redundancy_a.toFixed(3)}\n` +
        `redundancy B: ${layer.redundancy_b.toFixed(3)}\n` +
        `tensors: ${layer.tensor_count}`
      }
    >
      <span className="layer-index">{layer.layer}</span>
    </div>
  )
}

function OtherCard({ other }: { other: OtherScore }) {
  return (
    <div className="other-card" style={{ borderColor: driftColor(other.drift_magnitude) }}>
      <strong>Non-layer tensors</strong> (embeddings, norms, lm_head, …)
      <div>drift magnitude: {other.drift_magnitude.toFixed(3)}</div>
      <div>conflict: {other.conflict.toFixed(3)}</div>
      <div>redundancy A: {other.redundancy_a.toFixed(3)}</div>
      <div>redundancy B: {other.redundancy_b.toFixed(3)}</div>
      <div>tensors: {other.tensor_count}</div>
    </div>
  )
}

export function LayerHeatmap({ result }: { result: ConflictScoreResult }) {
  return (
    <section className="panel">
      <h2>Per-layer drift-risk heatmap</h2>
      <div className="legend">
        <span>low risk</span>
        <div className="legend-gradient" />
        <span>high risk</span>
      </div>
      <div className="layer-strip">
        {result.layers.map((layer) => (
          <LayerCell key={layer.layer} layer={layer} />
        ))}
      </div>
      {result.other && <OtherCard other={result.other} />}
    </section>
  )
}
