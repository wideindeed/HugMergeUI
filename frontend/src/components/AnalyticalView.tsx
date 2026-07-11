import { useState } from 'react'
import type { ConflictScoreResult, LayerScore } from '../api/types'

// Same validated risk framing as ConflictScene.tsx / LayerHeatmap.tsx, this
// view exists to show the exact numbers behind that color, not a new metric.
const DRIFT_RISK_CEILING = 0.9

function driftColor(drift: number): string {
  const shaped = Math.pow(Math.min(Math.max(drift, 0), DRIFT_RISK_CEILING) / DRIFT_RISK_CEILING, 0.6)
  const hue = 120 * (1 - shaped)
  return `hsl(${hue}, 70%, 50%)`
}

const CHART_HEIGHT = 220
const CHART_PAD_LEFT = 42
const CHART_PAD_BOTTOM = 22
const CHART_PAD_TOP = 12

export function AnalyticalView({ result }: { result: ConflictScoreResult }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const { layers, other } = result

  if (layers.length === 0) {
    return (
      <section className="panel analytical-panel">
        <h2>Layer-by-layer breakdown</h2>
        <p className="analytical-empty">No per-layer data in this result.</p>
      </section>
    )
  }

  const maxDrift = Math.max(DRIFT_RISK_CEILING, ...layers.map((l) => l.drift_magnitude), other?.drift_magnitude ?? 0)
  const yMax = maxDrift * 1.08
  const chartWidth = Math.max(layers.length * 26, 480)
  const plotWidth = chartWidth - CHART_PAD_LEFT - 8
  const plotHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM
  const barWidth = Math.min(20, (plotWidth / layers.length) * 0.7)

  function x(i: number): number {
    return CHART_PAD_LEFT + (i + 0.5) * (plotWidth / layers.length)
  }
  function y(v: number): number {
    return CHART_PAD_TOP + plotHeight * (1 - Math.min(v, yMax) / yMax)
  }

  const ceilingY = y(DRIFT_RISK_CEILING)
  const conflictPath = layers
    .map((l, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(l.conflict * yMax).toFixed(1)}`)
    .join(' ')

  const gridLines = 4
  const ticks = Array.from({ length: gridLines + 1 }, (_, i) => (yMax / gridLines) * i)

  const hovered = hoverIdx !== null ? layers[hoverIdx] : null

  return (
    <section className="panel analytical-panel">
      <h2>Layer-by-layer breakdown</h2>
      <p className="analytical-intro">
        Bars are drift magnitude (the validated risk signal) per layer, exact values, no shaping beyond color.
        The dashed line is the sign-conflict fraction, right-aligned to the same 0–1 span. The red dotted line marks
        the {DRIFT_RISK_CEILING} risk ceiling used for coloring elsewhere in this app.
      </p>

      <div className="analytical-chart-scroll">
        <svg
          width={chartWidth}
          height={CHART_HEIGHT}
          className="analytical-chart"
          onMouseLeave={() => setHoverIdx(null)}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={CHART_PAD_LEFT}
                x2={chartWidth - 8}
                y1={y(t)}
                y2={y(t)}
                stroke="#2a2a2a"
                strokeWidth={1}
              />
              <text x={CHART_PAD_LEFT - 6} y={y(t) + 3} textAnchor="end" className="analytical-axis-label">
                {t.toFixed(2)}
              </text>
            </g>
          ))}

          <line
            x1={CHART_PAD_LEFT}
            x2={chartWidth - 8}
            y1={ceilingY}
            y2={ceilingY}
            stroke="#ff5c5c"
            strokeWidth={1}
            strokeDasharray="4 3"
          />

          {layers.map((l, i) => {
            const barY = y(l.drift_magnitude)
            return (
              <rect
                key={l.layer}
                x={x(i) - barWidth / 2}
                y={barY}
                width={barWidth}
                height={Math.max(CHART_PAD_TOP + plotHeight - barY, 0.5)}
                fill={driftColor(l.drift_magnitude)}
                opacity={hoverIdx === null || hoverIdx === i ? 1 : 0.35}
                onMouseEnter={() => setHoverIdx(i)}
              />
            )
          })}

          <path d={conflictPath} fill="none" stroke="#8fbfff" strokeWidth={1.5} strokeDasharray="3 3" />

          {layers.map((l, i) => (
            <text
              key={l.layer}
              x={x(i)}
              y={CHART_HEIGHT - 6}
              textAnchor="middle"
              className="analytical-axis-label"
              opacity={layers.length > 40 && i % Math.ceil(layers.length / 40) !== 0 ? 0 : 1}
            >
              {l.layer}
            </text>
          ))}
        </svg>
      </div>

      <div className="analytical-hover-readout">
        {hovered ? (
          <span>
            layer {hovered.layer}, drift {hovered.drift_magnitude.toFixed(4)}, conflict{' '}
            {hovered.conflict.toFixed(4)}, redundancy A {hovered.redundancy_a.toFixed(4)}, redundancy B{' '}
            {hovered.redundancy_b.toFixed(4)}, {hovered.tensor_count} tensor
            {hovered.tensor_count === 1 ? '' : 's'}
          </span>
        ) : (
          <span className="analytical-hover-hint">Hover a bar for exact values.</span>
        )}
      </div>

      <div className="analytical-table-scroll">
        <table className="analytical-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Tensors</th>
              <th>Drift</th>
              <th>Conflict</th>
              <th>Redund. A</th>
              <th>Redund. B</th>
            </tr>
          </thead>
          <tbody>
            {layers.map((l: LayerScore, i) => (
              <tr
                key={l.layer}
                className={hoverIdx === i ? 'analytical-row-hover' : undefined}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx(null)}
              >
                <td>{l.layer}</td>
                <td>{l.tensor_count}</td>
                <td style={{ color: driftColor(l.drift_magnitude) }}>{l.drift_magnitude.toFixed(4)}</td>
                <td>{l.conflict.toFixed(4)}</td>
                <td>{l.redundancy_a.toFixed(4)}</td>
                <td>{l.redundancy_b.toFixed(4)}</td>
              </tr>
            ))}
            {other && (
              <tr className="analytical-other-row">
                <td>other</td>
                <td>{other.tensor_count}</td>
                <td style={{ color: driftColor(other.drift_magnitude) }}>{other.drift_magnitude.toFixed(4)}</td>
                <td>{other.conflict.toFixed(4)}</td>
                <td>{other.redundancy_a.toFixed(4)}</td>
                <td>{other.redundancy_b.toFixed(4)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}
