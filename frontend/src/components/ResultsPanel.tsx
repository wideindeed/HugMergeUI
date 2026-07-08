import { useState } from 'react'
import type { ConflictScoreResult } from '../api/types'
import { ConflictScene } from './ConflictScene'
import { LayerHeatmap } from './LayerHeatmap'
import { ScoreProgress } from './ScoreProgress'

interface Props {
  scoring: boolean
  progress: { percent: number; label: string } | null
  scoreResult: ConflictScoreResult | null
}

export function ResultsPanel({ scoring, progress, scoreResult }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <section className="panel results-panel" data-tour-id="results">
      <div className="results-header">
        <h2>Results</h2>
        <button type="button" className="collapse-button" onClick={() => setCollapsed((c) => !c)}>
          {collapsed ? 'Expand ▾' : 'Minimize ▴'}
        </button>
      </div>
      <div className={`results-collapse${collapsed ? ' collapsed' : ''}`}>
        <div className="results-collapse-inner">
          {scoring && progress && <ScoreProgress percent={progress.percent} label={progress.label} />}
          {scoreResult && (
            <>
              <ConflictScene layers={scoreResult.layers} other={scoreResult.other} />
              <LayerHeatmap result={scoreResult} />
              <p className="caveat">
                Heuristic score, not a validated quality predictor — correlation
                against real merge quality (perplexity, MMLU/GSM8K) failed to
                reach significance across 28 tested pairs. See the Phase 5/6
                write-up in the README for the full validation history.
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  )
}
