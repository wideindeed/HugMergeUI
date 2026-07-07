import type { ConflictScoreResult, LayerScore, OtherScore } from '../api/types'

function conflictColor(value: number): string {
  // 0 -> green (agreement), 1 -> red (conflict)
  const hue = 120 * (1 - value)
  return `hsl(${hue}, 70%, 45%)`
}

function LayerCell({ layer }: { layer: LayerScore }) {
  return (
    <div
      className="layer-cell"
      style={{ backgroundColor: conflictColor(layer.conflict) }}
      title={
        `layer ${layer.layer}\n` +
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
    <div className="other-card" style={{ borderColor: conflictColor(other.conflict) }}>
      <strong>Non-layer tensors</strong> (embeddings, norms, lm_head, …)
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
      <h2>Per-layer conflict heatmap</h2>
      <div className="legend">
        <span>agreement</span>
        <div className="legend-gradient" />
        <span>conflict</span>
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
