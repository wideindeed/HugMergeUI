import { useMemo, useState } from 'react'
import type { ConflictScoreResult, LayerScore } from '../api/types'
import { DRIFT_RISK_CEILING, driftColor, weightedAverage } from '../lib/risk'

type SortKey = keyof Pick<
  LayerScore,
  'layer' | 'tensor_count' | 'drift_magnitude' | 'conflict' | 'conflict_weighted' | 'redundancy_a' | 'redundancy_b'
>

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'layer', label: 'Layer' },
  { key: 'tensor_count', label: 'Tensors' },
  { key: 'drift_magnitude', label: 'Drift' },
  { key: 'conflict', label: 'Conflict' },
  { key: 'conflict_weighted', label: 'Conflict (wt)' },
  { key: 'redundancy_a', label: 'Redund. A' },
  { key: 'redundancy_b', label: 'Redund. B' },
]

function verdictTier(drift: number): { label: string; tone: string } {
  if (drift < 0.15) return { label: 'Low drift', tone: 'good' }
  if (drift < 0.45) return { label: 'Moderate drift', tone: 'ok' }
  if (drift < DRIFT_RISK_CEILING) return { label: 'Elevated drift', tone: 'warn' }
  return { label: 'High drift', tone: 'high' }
}

function toCsv(layers: LayerScore[], other: ConflictScoreResult['other']): string {
  const header = ['layer', ...COLUMNS.slice(1).map((c) => c.key)].join(',')
  const rows = layers.map((l) => COLUMNS.map((c) => l[c.key]).join(','))
  if (other) {
    rows.push(
      ['other', other.tensor_count, other.drift_magnitude, other.conflict, other.conflict_weighted, other.redundancy_a, other.redundancy_b].join(','),
    )
  }
  return [header, ...rows].join('\n')
}

function downloadCsv(csv: string) {
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'hugmergeui-layer-scores.csv'
  a.click()
  URL.revokeObjectURL(url)
}

export function DataView({ result }: { result: ConflictScoreResult }) {
  const { layers, other } = result
  const [sortKey, setSortKey] = useState<SortKey>('layer')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const overallDrift = useMemo(() => weightedAverage(layers, 'drift_magnitude'), [layers])
  const verdict = verdictTier(overallDrift)

  const sortedLayers = useMemo(() => {
    const dir = sortDir === 'asc' ? 1 : -1
    return [...layers].sort((a, b) => (a[sortKey] - b[sortKey]) * dir)
  }, [layers, sortKey, sortDir])

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  if (layers.length === 0) {
    return (
      <section className="panel data-view-panel">
        <h2>Risk &amp; layer data</h2>
        <p className="analytical-empty">No per-layer data in this result.</p>
      </section>
    )
  }

  return (
    <section className="panel data-view-panel">
      <h2>Risk &amp; layer data</h2>

      <div className={`verdict-banner verdict-${verdict.tone}`}>
        <div className="verdict-headline">
          <span className="verdict-tag">{verdict.label}</span>
          <span className="verdict-number" style={{ color: driftColor(overallDrift) }}>
            drift_magnitude {overallDrift.toFixed(4)}
          </span>
        </div>
        <p className="verdict-body">
          This is a tensor-count-weighted average across layers, the same aggregation validated against real merge
          damage (VALIDATION.txt, n=117 merges: pearson r=0.56, spearman rho=0.59, both p&lt;1e-9). Treat it as a
          smoke detector, not a fine-grained score: almost all of that predictive power comes from catching total
          blowups. Among merges that stayed in a reasonably healthy range, drift_magnitude only weakly ranks them
          (pearson r=0.18, not significant; spearman rho=0.32, significant but rank-level only). A low number here
          is a genuinely good sign. A high number is a real warning. Small differences between two already-low
          numbers are not meaningful.
        </p>
      </div>

      <div className="data-view-toolbar">
        <span className="analytical-hover-hint">Click a column header to sort.</span>
        <button type="button" className="csv-export-button" onClick={() => downloadCsv(toCsv(sortedLayers, other))}>
          Export CSV
        </button>
      </div>

      <div className="analytical-table-scroll">
        <table className="analytical-table">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th key={col.key}>
                  <button type="button" className="sortable-header" onClick={() => toggleSort(col.key)}>
                    {col.label}
                    {sortKey === col.key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedLayers.map((l) => (
              <tr key={l.layer}>
                <td>{l.layer}</td>
                <td>{l.tensor_count}</td>
                <td style={{ color: driftColor(l.drift_magnitude) }}>{l.drift_magnitude.toFixed(4)}</td>
                <td>{l.conflict.toFixed(4)}</td>
                <td>{l.conflict_weighted.toFixed(4)}</td>
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
                <td>{other.conflict_weighted.toFixed(4)}</td>
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
