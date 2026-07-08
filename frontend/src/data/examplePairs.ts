export interface ExamplePair {
  id: string
  label: string
  note: string
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

export const EXAMPLE_PAIRS: ExamplePair[] = [
  {
    id: 'qwen-instruct',
    label: 'Qwen2.5-0.5B: base + Instruct',
    note: 'Smallest/fastest pair — good first run.',
    yaml: tiesYaml('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct'),
  },
  {
    id: 'qwen-dolphin-capybara',
    label: 'Qwen2.5-0.5B: Dolphin + Capybara SFT',
    note: 'Two independent fine-tunes on unrelated datasets — the real target scenario.',
    yaml: tiesYaml('Qwen/Qwen2.5-0.5B', 'dphn/Dolphin3.0-Qwen2.5-0.5B', 'wulli/Qwen2.5-0.5B-sft-capybara'),
  },
  {
    id: 'smollm2-self-merge',
    label: 'SmolLM2-360M: self-merge baseline',
    note: 'Same model on both sides — expect ~0 conflict, sanity-checks the score.',
    yaml: tiesYaml('HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-360M'),
  },
  {
    id: 'qwen-instruct-qgallouedec',
    label: 'Qwen2.5-0.5B: Instruct + SFT (qgallouedec)',
    note: 'Typical case — conflict score saturates near ~0.5, as most non-trivial pairs do.',
    yaml: tiesYaml('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct', 'qgallouedec/Qwen2.5-0.5B-SFT'),
  },
  {
    id: 'smollm2-instruct-wikitext',
    label: 'SmolLM2-360M: Instruct + wikitext-finetune (anomaly)',
    note: 'Near-zero conflict despite real weight drift — an open, unexplained outlier from validation. Worth poking at.',
    yaml: tiesYaml(
      'HuggingFaceTB/SmolLM2-360M',
      'HuggingFaceTB/SmolLM2-360M-Instruct',
      'Michaelj1/INSTRUCT_smolLM2-360M-finetuned-wikitext2-raw-v1',
    ),
  },
]
