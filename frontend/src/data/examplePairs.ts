export interface ExamplePair {
  id: string
  label: string
  note: string
  simpleNote: string
  yaml: string
}

function tiesYaml(base: string, modelA: string, modelB: string): string {
  return `merge_method: ties
base_model: ${base}
models:
  - model: ${modelA}
    parameters:
      weight: 0.5
      density: 0.5
  - model: ${modelB}
    parameters:
      weight: 0.5
      density: 0.5
parameters:
  normalize: true
dtype: float32
`
}

// All presets below sit at 1-1.7B params, the scale where drift_magnitude
// is actually validated (VALIDATION.txt Rounds Four/Five/Nine — spearman
// rho up to 0.96 against real perplexity across 29 pairs). Earlier presets
// were all 0.5B/360M, exactly the scale where the metric is proven dead —
// replaced outright rather than kept as a "known bad" example, since a
// diagnostic tool's whole pitch is only real at sizes worth diagnosing.
export const EXAMPLE_PAIRS: ExamplePair[] = [
  {
    id: 'qwen1.5b-self-merge',
    label: 'Qwen2.5-1.5B: self-merge baseline',
    note: 'Same model on both sides — measured drift 0.000, perplexity 7.12 (== the base model alone). Sanity-checks that the score reads zero when there is nothing to disagree about.',
    simpleNote: "The same model merged with itself. Nothing should clash — a good sanity check that the tool isn't lying to you.",
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B'),
  },
  {
    id: 'qwen1.5b-instruct-capybara',
    label: 'Qwen2.5-1.5B: Instruct + Capybara SFT',
    note: 'Two independent fine-tunes, the real target scenario — measured drift 0.021, perplexity 7.50 (barely above the 7.12 baseline). Validated healthy: low drift correctly predicts a low-damage merge.',
    simpleNote: "Two independently fine-tuned versions of the same model, mixed together. Real measurements: they got along fine — barely any damage.",
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 'lewtun/Qwen2.5-1.5B-SFT-Capybara-No-Packing'),
  },
  {
    id: 'smollm2-1.7b-instruct-smoltulu',
    label: 'SmolLM2-1.7B: Instruct + SmolTulu',
    note: 'Moderate case — measured drift 0.254, perplexity 17.36 (vs. 7.22 baseline), a real, noticeable quality hit without being catastrophic. Sits mid-scale on the validated drift↔perplexity curve.',
    simpleNote: 'A middle-of-the-road merge. Real measurements show a genuine, noticeable dip in quality — not a disaster, but not free either.',
    yaml: tiesYaml('HuggingFaceTB/SmolLM2-1.7B', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'SultanR/SmolTulu-1.7b-Instruct'),
  },
  {
    id: 'llama3.2-1b-instruct-capybara',
    label: 'Llama-3.2-1B: Instruct + Capybara SFT',
    note: 'A third, architecturally distinct family (validated in Round Five alongside Qwen/SmolLM2) — measured drift 0.176, perplexity 11.66 (vs. 9.14 baseline). Confirms the metric isn\'t a Qwen-specific fluke.',
    simpleNote: 'Same idea, different model family (Meta\'s Llama instead of Qwen or SmolLM2) — proof this isn\'t a one-model-family fluke.',
    yaml: tiesYaml('unsloth/Llama-3.2-1B', 'unsloth/Llama-3.2-1B-Instruct', 'lewtun/Llama-3.2-1B-SFT-Capybara-No-Packing-Llama'),
  },
  {
    id: 'qwen1.5b-coder-math-catastrophic',
    label: 'Qwen2.5: Coder-1.5B + Math-1.5B (catastrophic)',
    note: 'Two disjoint specialist fine-tunes — measured drift 2.475 (near the top of the observed range), perplexity 6,915,101 vs. a healthy ~7-9. This is what the score is actually for: catching a merge that would otherwise burn real GPU time to produce a broken model.',
    simpleNote: "The big one. Two specialists pulled so far apart (coding vs. math) that merging them essentially destroys the model — perplexity rockets from a healthy ~7 to almost 7 million. This is the exact disaster this tool exists to catch before you waste time on it.",
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-Coder-1.5B', 'Qwen/Qwen2.5-Math-1.5B'),
  },
]
