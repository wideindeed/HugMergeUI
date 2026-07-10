import type { ConflictScoreResult } from '../api/types'

const RISK_CEILING = 0.9

export interface PlainExplanation {
  headline: string
  paragraphs: string[]
}

function mean(nums: number[]): number {
  return nums.length === 0 ? 0 : nums.reduce((a, b) => a + b, 0) / nums.length
}

export function explainResult(result: ConflictScoreResult): PlainExplanation {
  const { layers, other } = result
  const avgLayerDrift = mean(layers.map((l) => l.drift_magnitude))

  let headline: string
  if (avgLayerDrift < 0.15) {
    headline = 'These two models get along well.'
  } else if (avgLayerDrift < 0.45) {
    headline = 'These two models mostly agree, with some friction.'
  } else if (avgLayerDrift < RISK_CEILING) {
    headline = 'These two models pull in noticeably different directions.'
  } else {
    headline = "These two models are fighting hard — expect real damage if you merge them."
  }

  const paragraphs: string[] = [
    'Think of each model as a set of learned habits. When you merge two models, you\'re averaging their habits together — and averaging works fine when they agree, but gets messy when they don\'t.',
  ]

  if (other) {
    const otherDrift = other.drift_magnitude
    const diff = otherDrift - avgLayerDrift
    const meaningfulGap = Math.abs(diff) > 0.15 && otherDrift > avgLayerDrift * 1.5
    if (meaningfulGap) {
      paragraphs.push(
        "One thing worth noticing: the \"vocabulary and output\" section (labeled other) shows a lot more risk " +
          "than the typical layer above. That usually means one of the two models was pushed hard into a narrow " +
          "specialty — like math, code, or a specific writing style — which reshapes how it turns its thinking " +
          'into words, even when the general reasoning underneath barely changed. Different parts of a merge can ' +
          'genuinely disagree like this; it\'s not a bug, it\'s two independent readings.',
      )
    } else {
      paragraphs.push(
        'The "vocabulary and output" section (labeled other) is roughly in line with the typical layer — no ' +
          'single part is carrying a wildly different amount of risk here.',
      )
    }
  }

  paragraphs.push(
    'The color scale is driven by how much each model\'s training actually changed its weights (not just whether ' +
      'they disagree) — a signal that has been tested against real merge quality. It\'s solidly proven at larger ' +
      'model sizes (1.5B+ parameters); at the small scale most examples here use, treat it as a helpful hint ' +
      'rather than a guarantee.',
  )

  return { headline, paragraphs }
}
