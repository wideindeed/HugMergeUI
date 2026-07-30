import type { LayerScore } from '../api/types'

// drift_magnitude is the validated risk signal (VALIDATION.txt Rounds 4-8);
// conflict/conflict_weighted are shown for comparison only. 0.9 is the
// ceiling used for color/verdict shaping across every view in this app.
export const DRIFT_RISK_CEILING = 0.9

export function driftColor(drift: number, lightness = 50): string {
  const shaped = Math.pow(Math.min(Math.max(drift, 0), DRIFT_RISK_CEILING) / DRIFT_RISK_CEILING, 0.6)
  const hue = 120 * (1 - shaped)
  return `hsl(${hue}, 70%, ${lightness}%)`
}

// Tensor-count-weighted mean over layers only (excludes `other`), matching
// the exact aggregation that produced the merge-level drift_magnitude
// validated against real perplexity in eval/big_validation.py (avg_metric)
// and pooled in eval/phase_pooled_meta.py.
export function weightedAverage(layers: LayerScore[], field: keyof LayerScore): number {
  const totalWeight = layers.reduce((sum, l) => sum + l.tensor_count, 0)
  if (totalWeight === 0) return 0
  return layers.reduce((sum, l) => sum + l.tensor_count * (l[field] as number), 0) / totalWeight
}
