import { useState } from 'react'
import type { ConflictScoreResult } from '../api/types'
import { useSimpleMode } from '../context/SimpleModeContext'
import { explainResult } from '../lib/plainEnglish'
import { AnalyticalView } from './AnalyticalView'
import { ConflictScene } from './ConflictScene'
import { LayerHeatmap } from './LayerHeatmap'
import { ScoreProgress } from './ScoreProgress'

interface Props {
  scoring: boolean
  progress: { percent: number; label: string } | null
  scoreResult: ConflictScoreResult | null
}

type ViewMode = 'orbit' | 'analytical'

export function ResultsPanel({ scoring, progress, scoreResult }: Props) {
  const { simple } = useSimpleMode()
  const [collapsed, setCollapsed] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('orbit')

  return (
    <section className="panel results-panel" data-tour-id="results">
      <div className="results-header">
        <h2>{simple ? 'What we found' : 'Results'}</h2>
        <button type="button" className="collapse-button" onClick={() => setCollapsed((c) => !c)}>
          {collapsed ? 'Expand ▾' : 'Minimize ▴'}
        </button>
      </div>
      <div className={`results-collapse${collapsed ? ' collapsed' : ''}`}>
        <div className="results-collapse-inner">
          {scoring && progress && <ScoreProgress percent={progress.percent} label={progress.label} />}
          {scoreResult && (
            <>
              {!simple && (
                <div className="view-toggle" role="tablist" aria-label="Merge visual mode">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={viewMode === 'orbit'}
                    className={viewMode === 'orbit' ? 'view-toggle-active' : undefined}
                    onClick={() => setViewMode('orbit')}
                  >
                    Orbit view
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={viewMode === 'analytical'}
                    className={viewMode === 'analytical' ? 'view-toggle-active' : undefined}
                    onClick={() => setViewMode('analytical')}
                  >
                    Analytical view
                  </button>
                </div>
              )}
              {simple || viewMode === 'orbit' ? (
                <ConflictScene layers={scoreResult.layers} other={scoreResult.other} />
              ) : (
                <AnalyticalView result={scoreResult} />
              )}
              {simple ? (
                <PlainEnglishExplanation result={scoreResult} />
              ) : (
                <>
                  <LayerHeatmap result={scoreResult} />
                  <p className="caveat">
                    drift_magnitude (the color/size signal above) is validated against real merge
                    quality — significant at 1.5B+ params (Round Four/Five: n=29, spearman
                    ρ=0.96), but not yet shown to hold at the 0.5B/360M scale most example
                    presets use here. conflict/conflict_weighted, shown as secondary stats,
                    did not reach significance in the original Phase 5/6 sweep. See VALIDATION.txt
                    for the full round-by-round history.
                  </p>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  )
}

function PlainEnglishExplanation({ result }: { result: ConflictScoreResult }) {
  const { headline, paragraphs } = explainResult(result)
  return (
    <div className="plain-explanation">
      <h3>{headline}</h3>
      {paragraphs.map((p, i) => (
        <p key={i}>{p}</p>
      ))}
    </div>
  )
}
