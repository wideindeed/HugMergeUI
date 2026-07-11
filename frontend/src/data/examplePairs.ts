export type ExampleTier = 'self-merge' | 'healthy' | 'parent-child' | 'moderate' | 'edge-case' | 'catastrophic'

export interface ExamplePair {
  id: string
  label: string
  tier: ExampleTier
  note: string
  simpleNote: string
  yaml: string
}

export const TIER_INFO: Record<ExampleTier, { label: string; simpleLabel: string }> = {
  'self-merge': { label: 'Sanity checks (self-merge)', simpleLabel: 'Sanity checks' },
  healthy: { label: 'Healthy merges', simpleLabel: 'These went fine' },
  'parent-child': { label: 'Parent/child fine-tunes', simpleLabel: 'Same lineage, not a real clash' },
  moderate: { label: 'Moderate damage', simpleLabel: 'Noticeable but survivable' },
  'edge-case': { label: 'Edge cases, the metrics disagree', simpleLabel: "Where the tool's own signals conflict" },
  catastrophic: { label: 'Catastrophic domain clashes', simpleLabel: 'This is what breaks' },
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

// Every entry below is a real pair measured in VALIDATION.txt (Rounds
// Four/Five/Six/Nine/Eleven/Twelve), all at 1.5B+ params - the scale
// where drift_magnitude is actually validated (spearman rho up to 0.96
// against real perplexity). Earlier presets sat at 0.5B/360M, exactly
// the scale where the metric is proven dead; they were replaced rather
// than kept as "known bad" examples. This list spans 1B-3B, two
// architecture families (Qwen2.5, Llama 3.2) plus SmolLM2, and every
// risk tier the tool is meant to distinguish between.
export const EXAMPLE_PAIRS: ExamplePair[] = [
  // --- self-merge sanity checks ---
  {
    id: 'qwen1.5b-self-merge',
    label: 'Qwen2.5-1.5B: self-merge baseline',
    tier: 'self-merge',
    note: 'Same model on both sides, measured drift 0.000, perplexity 7.12 (== the base model alone). Sanity-checks that the score reads zero when there is nothing to disagree about.',
    simpleNote: "The same model merged with itself. Nothing should clash, a good sanity check that the tool isn't lying to you.",
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B'),
  },
  {
    id: 'qwen3b-self-merge',
    label: 'Qwen2.5-3B: self-merge baseline',
    tier: 'self-merge',
    note: 'Same check, one scale up, measured drift 0.000, perplexity 6.68. Confirms the zero-drift baseline holds at 3B, not just 1.5B.',
    simpleNote: 'Same sanity check, bigger model. Still nothing should clash.',
    yaml: tiesYaml('Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B'),
  },
  {
    id: 'llama3.2-3b-self-merge',
    label: 'Llama-3.2-3B: self-merge baseline',
    tier: 'self-merge',
    note: 'Same check on the second 3B family, measured drift 0.000, perplexity 6.97.',
    simpleNote: 'Same sanity check, different model family.',
    yaml: tiesYaml('unsloth/Llama-3.2-3B', 'unsloth/Llama-3.2-3B', 'unsloth/Llama-3.2-3B'),
  },
  {
    id: 'stablelm2-self-merge',
    label: 'StableLM-2-1.6B: self-merge baseline',
    tier: 'self-merge',
    note: 'Third architecture family entirely (Stability AI, not Qwen or Llama), drift 0.000, perplexity 9.03. Confirms the zero-drift baseline holds outside the two main families.',
    simpleNote: 'Same sanity check, a third, unrelated model family (Stability AI).',
    yaml: tiesYaml('stabilityai/stablelm-2-1_6b', 'stabilityai/stablelm-2-1_6b', 'stabilityai/stablelm-2-1_6b'),
  },

  // --- healthy merges ---
  {
    id: 'qwen1.5b-instruct-capybara',
    label: 'Qwen2.5-1.5B: Instruct + Capybara SFT',
    tier: 'healthy',
    note: 'Two independent fine-tunes, the real target scenario, measured drift 0.021, perplexity 7.50 (barely above the 7.12 baseline). Validated healthy: low drift correctly predicts a low-damage merge.',
    simpleNote: "Two independently fine-tuned versions of the same model, mixed together. Real measurements: they got along fine, barely any damage.",
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 'lewtun/Qwen2.5-1.5B-SFT-Capybara-No-Packing'),
  },
  {
    id: 'llama3.2-1b-sft-capybara',
    label: 'Llama-3.2-1B: SFT + Capybara SFT',
    tier: 'healthy',
    note: 'Two general instruction-tuning runs on the same base, measured drift 0.038, perplexity 10.63 (vs. 9.14 baseline). Another clean, low-drift/low-damage healthy case.',
    simpleNote: 'Two general-purpose fine-tunes on the same base model, almost no measurable damage.',
    yaml: tiesYaml('unsloth/Llama-3.2-1B', 'RLHFlow/LLaMA3.2-1B-SFT', 'lewtun/Llama-3.2-1B-SFT-Capybara-No-Packing-Llama'),
  },
  {
    id: 'smollm2-1.7b-tulu3-code',
    label: 'SmolLM2-1.7B: Tulu3 SFT + code SFT',
    tier: 'healthy',
    note: 'The lowest measured drift of any non-self-merge pair on record, drift 0.015, perplexity 7.53 (vs. 7.22 baseline). Shows the metric doesn\'t just flag "any two fine-tunes" as risky.',
    simpleNote: 'About as safe a real-world merge as this project has measured.',
    yaml: tiesYaml('HuggingFaceTB/SmolLM2-1.7B', 'ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated', 'CodeAtCMU/SmolLM2-1.7B_full_sft_code_data_120K'),
  },
  {
    id: 'smollm2-1.7b-instruct-sftonly',
    label: 'SmolLM2-1.7B: Instruct + SFT-only',
    tier: 'healthy',
    note: 'Two close variants of the same instruction-tuning recipe, drift 0.126, perplexity 7.61 (vs. 7.22 baseline). Mild, healthy.',
    simpleNote: 'Two close cousins of the same fine-tune, mild, healthy result.',
    yaml: tiesYaml('HuggingFaceTB/SmolLM2-1.7B', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'HuggingFaceTB/SmolLM2-1.7B-sft-only'),
  },
  {
    id: 'stablelm2-zephyr-dpo',
    label: 'StableLM-2-1.6B: Zephyr + DPO',
    tier: 'healthy',
    note: 'Two independently-trained instruct variants on the third architecture family, drift 0.088, perplexity 10.31 (vs. 9.03 baseline). Healthy, consistent with the pattern on Qwen/Llama.',
    simpleNote: 'Two instruct variants of a third, unrelated model family, healthy result again.',
    yaml: tiesYaml('stabilityai/stablelm-2-1_6b', 'stabilityai/stablelm-2-zephyr-1_6b', 'nnheui/stablelm-2-1_6b-dpo-full-ultrafeedback_generated'),
  },
  {
    id: 'stablelm2-chat-dpo',
    label: 'StableLM-2-1.6B: Chat + DPO',
    tier: 'healthy',
    note: 'Drift 0.110, perplexity 9.72 (vs. 9.03 baseline), the healthiest of the three StableLM-2 pairs, despite having the highest measured conflict (0.426) of the three. A real case where conflict alone would have ranked this the riskiest pair; drift_magnitude ranks it correctly as the safest.',
    simpleNote: 'Actually the safest of this trio, but the "conflict" signal alone would have called it the riskiest one. A concrete example of why this tool leans on drift, not conflict, as the primary signal.',
    yaml: tiesYaml('stabilityai/stablelm-2-1_6b', 'stabilityai/stablelm-2-1_6b-chat', 'nnheui/stablelm-2-1_6b-dpo-full-ultrafeedback_generated'),
  },

  // --- parent/child (the confound-close finding) ---
  {
    id: 'qwen3b-instruct-abliterated',
    label: 'Qwen2.5-3B-Instruct + its abliterated child',
    tier: 'parent-child',
    note: 'An abliteration fine-tune built directly on top of Instruct, same lineage, not domain-divergent. Drift 0.032, conflict 0.077, perplexity 9.11 (vs. 6.68 baseline): low across the board, exactly as expected. Closes a methodological question (VALIDATION.txt Rounds Nine and Eleven): parent/child structure alone doesn\'t fool the metric, it\'s domain divergence that does the damage, not shared ancestry.',
    simpleNote: 'One model is a direct descendant of the other (same training lineage, no real subject-matter change). Low risk across the board, proof the tool isn\'t just flagging "these came from different fine-tuning runs," it\'s flagging real subject-matter conflict.',
    yaml: tiesYaml('Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct', 'huihui-ai/Qwen2.5-3B-Instruct-abliterated'),
  },
  {
    id: 'qwen1.5b-instruct-abliterated',
    label: 'Qwen2.5-1.5B-Instruct + its abliterated child',
    tier: 'parent-child',
    note: 'The same parent/child confound-close, one scale down from the 3B version above, drift 0.034, conflict 0.086, perplexity 8.25 (healthy). This is the original Round Nine result the 3B version was built to replicate.',
    simpleNote: 'The smaller-scale version of the "same lineage" example above, same clean, low-risk result.',
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 'huihui-ai/Qwen2.5-1.5B-Instruct-abliterated'),
  },

  // --- moderate damage ---
  {
    id: 'smollm2-1.7b-instruct-smoltulu',
    label: 'SmolLM2-1.7B: Instruct + SmolTulu',
    tier: 'moderate',
    note: 'Moderate case, measured drift 0.254, perplexity 17.36 (vs. 7.22 baseline), a real, noticeable quality hit without being catastrophic. Sits mid-scale on the validated drift↔perplexity curve.',
    simpleNote: 'A middle-of-the-road merge. Real measurements show a genuine, noticeable dip in quality, not a disaster, but not free either.',
    yaml: tiesYaml('HuggingFaceTB/SmolLM2-1.7B', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'SultanR/SmolTulu-1.7b-Instruct'),
  },
  {
    id: 'llama3.2-1b-instruct-capybara',
    label: 'Llama-3.2-1B: Instruct + Capybara SFT',
    tier: 'moderate',
    note: 'A third, architecturally distinct family (validated in Round Five alongside Qwen/SmolLM2), measured drift 0.176, perplexity 11.66 (vs. 9.14 baseline). Confirms the metric isn\'t a Qwen-specific fluke.',
    simpleNote: 'Same idea, different model family (Meta\'s Llama instead of Qwen or SmolLM2), proof this isn\'t a one-model-family fluke.',
    yaml: tiesYaml('unsloth/Llama-3.2-1B', 'unsloth/Llama-3.2-1B-Instruct', 'lewtun/Llama-3.2-1B-SFT-Capybara-No-Packing-Llama'),
  },
  {
    id: 'llama3.2-1b-dolphin-instruct',
    label: 'Llama-3.2-1B: Dolphin + Instruct',
    tier: 'moderate',
    note: 'Two general-purpose chat fine-tunes with different data mixes, drift 0.244, perplexity 11.07 (vs. 9.14 baseline). Real but modest damage.',
    simpleNote: 'Two different chat-style fine-tunes, a real but modest quality dip.',
    yaml: tiesYaml('unsloth/Llama-3.2-1B', 'dphn/Dolphin3.0-Llama3.2-1B', 'unsloth/Llama-3.2-1B-Instruct'),
  },
  {
    id: 'llama3.2-3b-instruct-coder-nearmiss',
    label: 'Llama-3.2-3B: Instruct + Agent007-Coder (near-miss)',
    tier: 'moderate',
    note: 'A genuine code-domain fine-tune, chosen to try to reproduce Qwen\'s catastrophic coder/math pattern on a second architecture, it didn\'t. Drift 0.324, perplexity 8.56 (vs. 6.97 baseline): a real but mild bump, not a blowup. Documented honestly in VALIDATION.txt Round Twelve as inconclusive, not a confirmation, the fine-tune wasn\'t domain-shifted enough to force the same failure mode.',
    simpleNote: 'An attempt to reproduce the "coding models don\'t mix with math models" disaster on a different model family. It came back mild instead, an honest negative result, not hidden here.',
    yaml: tiesYaml('unsloth/Llama-3.2-3B', 'unsloth/Llama-3.2-3B-Instruct', 'EpistemeAI/Llama-3.2-3B-Agent007-Coder'),
  },
  {
    id: 'stablelm2-zephyr-chat',
    label: 'StableLM-2-1.6B: Zephyr + Chat',
    tier: 'moderate',
    note: 'Third architecture family, real degradation, drift 0.171, perplexity 12.64 (vs. 9.03 baseline). Confirms the healthy/moderate/catastrophic pattern isn\'t Qwen- or Llama-specific.',
    simpleNote: 'A real, moderate quality dip on a third, unrelated model family, same pattern seen elsewhere.',
    yaml: tiesYaml('stabilityai/stablelm-2-1_6b', 'stabilityai/stablelm-2-zephyr-1_6b', 'stabilityai/stablelm-2-1_6b-chat'),
  },

  // --- edge cases: conflict and drift_magnitude disagree ---
  {
    id: 'qwen3b-coder-coder-instruct-edgecase',
    label: 'Qwen2.5-3B: Coder + Coder-Instruct (drift overstates risk)',
    tier: 'edge-case',
    note: 'Coder-3B-Instruct is instruction-tuned directly FROM Coder-3B, not an independent sibling, a parent/child pair, same code domain throughout. Conflict is near-zero (0.012, the lowest non-trivial score on record) and correctly reads this as low-risk. But drift_magnitude is the highest ever measured (1.845) despite this producing the best perplexity (18.20) of any non-trivial 3B pair. A large, well-aligned update apparently isn\'t damaging here, a genuine, documented gap in drift_magnitude as currently formulated (VALIDATION.txt Round Six/Seven): it measures update size, not whether the two updates agree.',
    simpleNote: 'One of these two signals gets fooled here: "drift" reads this as the riskiest pair on record, but it actually merged the best out of any comparable pair. A real, documented blind spot, not swept under the rug.',
    yaml: tiesYaml('Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-Coder-3B', 'Qwen/Qwen2.5-Coder-3B-Instruct'),
  },
  {
    id: 'qwen1.5b-coder-coder-instruct-edgecase',
    label: 'Qwen2.5-1.5B: Coder + Coder-Instruct (conflict misses it)',
    tier: 'edge-case',
    note: 'The flip side of the 3B case above, at 1.5B: another parent/child pair, but this time genuinely catastrophic, perplexity 414.5 (vs. ~7 healthy). Conflict reads it as almost risk-free (0.003, near the self-merge floor) and would have completely missed this. drift_magnitude correctly flags it (1.83, elevated). Documents why this project treats drift_magnitude, not conflict, as the primary signal.',
    simpleNote: 'Here "conflict" is the one that gets fooled, it reads almost no risk on a merge that actually failed badly. "Drift" catches it. The reason this tool leans on drift as the main number.',
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-Coder-1.5B', 'Qwen/Qwen2.5-Coder-1.5B-Instruct'),
  },
  {
    id: 'qwen1.5b-math-math-instruct-edgecase',
    label: 'Qwen2.5-1.5B: Math + Math-Instruct (conflict misses it)',
    tier: 'edge-case',
    note: 'Same pattern as the coder/coder-instruct case, independently replicated on the math specialist family, perplexity 893.1, conflict near-zero (0.036), drift_magnitude elevated and correct (3.01, the highest measured value on record at any scale). Two independent confirmations that conflict alone can miss a real catastrophic parent/child merge.',
    simpleNote: 'The same "conflict misses it, drift catches it" story, confirmed a second time on a different specialist pair.',
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct'),
  },

  // --- catastrophic domain clashes ---
  {
    id: 'qwen1.5b-coder-math-catastrophic',
    label: 'Qwen2.5: Coder-1.5B + Math-1.5B (catastrophic)',
    tier: 'catastrophic',
    note: 'Two disjoint specialist fine-tunes, measured drift 2.475 (near the top of the observed range), perplexity 6,915,101 vs. a healthy ~7-9. This is what the score is actually for: catching a merge that would otherwise burn real GPU time to produce a broken model.',
    simpleNote: "The big one. Two specialists pulled so far apart (coding vs. math) that merging them essentially destroys the model, perplexity rockets from a healthy ~7 to almost 7 million. This is the exact disaster this tool exists to catch before you waste time on it.",
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-Coder-1.5B', 'Qwen/Qwen2.5-Math-1.5B'),
  },
  {
    id: 'qwen1.5b-coder-capybara-catastrophic',
    label: 'Qwen2.5-1.5B: Coder + Capybara SFT (catastrophic)',
    tier: 'catastrophic',
    note: 'A code specialist against a general chat fine-tune, drift 0.924, perplexity 76,070 vs. ~7 healthy. High drift correctly flags this before the merge is ever run.',
    simpleNote: 'A code specialist mixed with a general chat model, badly broken, and flagged well in advance.',
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-Coder-1.5B', 'lewtun/Qwen2.5-1.5B-SFT-Capybara-No-Packing'),
  },
  {
    id: 'qwen1.5b-dolphin-coder-catastrophic',
    label: 'Qwen2.5-1.5B: Dolphin + Coder (catastrophic)',
    tier: 'catastrophic',
    note: 'General chat fine-tune against a code specialist, drift 0.954, perplexity 22,865 vs. ~7 healthy.',
    simpleNote: 'A chat model mixed with a code specialist, badly broken.',
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'dphn/Dolphin3.0-Qwen2.5-1.5B', 'Qwen/Qwen2.5-Coder-1.5B'),
  },
  {
    id: 'qwen1.5b-instruct-math-catastrophic',
    label: 'Qwen2.5-1.5B: Instruct + Math (catastrophic)',
    tier: 'catastrophic',
    note: 'General instruct against a math specialist, drift 1.571, perplexity 629.5 vs. ~7 healthy. Less extreme than coder/math but still a clear failure the score catches.',
    simpleNote: 'A general assistant mixed with a math specialist, a real failure, correctly flagged.',
    yaml: tiesYaml('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-Math-1.5B'),
  },
  {
    id: 'qwen3b-instruct-coder-catastrophic',
    label: 'Qwen2.5-3B: Instruct + Coder (catastrophic, replicated at 3B)',
    tier: 'catastrophic',
    note: 'The same coder/instruct clash that was catastrophic at 1.5B, replicated at 3B scale, drift 0.935, perplexity 70.9 vs. 6.68 baseline. Confirms the failure mode isn\'t a small-model artifact.',
    simpleNote: 'The same "coding model doesn\'t mix with a general assistant" disaster, confirmed again at a larger model size.',
    yaml: tiesYaml('Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-Coder-3B'),
  },
  {
    id: 'qwen3b-coder-abliterated-catastrophic',
    label: 'Qwen2.5-3B: Coder + abliterated Instruct (catastrophic)',
    tier: 'catastrophic',
    note: 'Pairs the code specialist against the (otherwise low-risk) abliterated child from the parent/child case above, drift 0.942, perplexity 103.0 vs. 6.68 baseline. Shows the same abliterated model that scored safe against its own parent scores catastrophic against a domain-divergent specialist: it\'s the pairing, not the model alone, that the tool has to evaluate.',
    simpleNote: 'The same "safe" model from the parent/child example above turns risky the moment it\'s paired with a genuinely different specialist, proof the risk lives in the pairing, not any one model.',
    yaml: tiesYaml('Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-Coder-3B', 'huihui-ai/Qwen2.5-3B-Instruct-abliterated'),
  },
]
