# HugMergeUI

Predictive conflict diagnostics for mergekit model merges: sign-conflict and
redundancy metrics per layer, computed from weight diffs, surfaced as a
heatmap over a merge-config diagram, before you spend hours running the
merge on a GPU.

## Structure

- `backend/` : FastAPI service. YAML config parsing, Hugging Face
  architecture sanity checks, conflict-score engine (safetensors weight
  diffs, TIES-style sign-conflict/redundancy math).
- `frontend/` : Vite + React + TypeScript. Stacked layer diagram, heatmap
  overlay, config editor.

## Backend setup

```
.venv\Scripts\activate
pip install -r backend/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126
uvicorn backend.app.main:app --port 8010 --reload
```

Port 8010, not 8000: on Windows, `8000` frequently falls inside the OS's
dynamic port exclusion range (Hyper-V/WSL reserve chunks of it), which
fails with a permissions error on bind. The frontend's Vite dev proxy is
already pointed at `8010` to match.

Verify CUDA is actually being used (plain `pip install torch` silently grabs
the CPU-only build):

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Frontend setup

```
cd frontend
npm install
npm run dev
```

## Test model pairs

Used for backend development/testing throughout Phases 1-3 (no GPU needed
until Phase 5):

- `Qwen/Qwen2.5-0.5B` + `Qwen/Qwen2.5-0.5B-Instruct` : smallest pair (~1GB
  each), fastest iteration loop
- `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` +
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — same-arch baseline (~2.2GB each)
- `dphn/Dolphin3.0-Qwen2.5-0.5B` + `wulli/Qwen2.5-0.5B-sft-capybara`, both
  SFT'd from `Qwen/Qwen2.5-0.5B` by unrelated authors on different
  datasets — the actual target scenario. Confirmed architecturally
  compatible via `/check-architecture`, and confirmed non-trivial: real
  weight diffs (not near-zero noise), average sign conflict ~0.48 across
  layers (near chance, as expected for independently-drifted tunes) vs.
  0.0 for a model merged with itself

## Phase 5: does the conflict score actually predict merge quality?

`eval/` holds a real test of the core claim: run mergekit for real, on real
weights, and check whether a higher sign-conflict score predicts a worse
merged model (measured as perplexity on a small fixed held-out text sample,
`eval/measure_perplexity.py`). All three merges use `ties`, density 0.5,
equal weight, against `Qwen/Qwen2.5-0.5B` as base:

| variant | models merged | conflict score | perplexity |
|---|---|---|---|
| `anchor-self` | base merged with itself | 0.0 | 8.23 (identical to base) |
| `validated-medium` | Dolphin3.0 + Capybara-sft | 0.482 | 9.23 |
| `candidate-high` | Dolphin3.0 + Qwen2.5-0.5B-Instruct | 0.468 | 10.33 (worst) |

Two honest findings, not massaged to look cleaner than they are:

1. **The binary signal holds perfectly.** A conflict score of exactly 0.0
   predicts exactly zero quality loss (the self-merge reproduces the base
   model's perplexity to the decimal). Any independently-trained pair shows
   real, measurable degradation. This is the claim the whole project rests
   on, and it checks out.
2. **The metric does not finely rank-order between two non-trivial pairs.**
   `candidate-high` has a *lower* conflict score than `validated-medium` but
   a *worse* merge. Likely explanation: sign-conflict-rate on raw weight
   diffs saturates near chance level (~0.47-0.48) for almost any two
   independently-updated models, regardless of how differently they were
   trained — it doesn't capture that `Qwen2.5-0.5B-Instruct` went through
   much heavier alignment/RLHF-style tuning than a plain SFT run, which
   apparently matters more for merge quality than the sign-conflict rate
   does.

**Attempted fix, and why it failed:** the natural hypothesis is that plain
sign-conflict-rate is *magnitude-blind* — for two independent updates it
converges toward ~50% regardless of whether the disagreements are tiny
noise or large, load-bearing changes. `magnitude_weighted_conflict_rate`
(`backend/app/conflict/sign_conflict.py`) weights each element by
`min(|diff_a|, |diff_b|)` — the amount of update actually contested — instead
of counting elements uniformly, so a handful of huge disagreements can
dominate a sea of small ones. Re-running the same three triples:

| variant | conflict (raw) | conflict_weighted | perplexity |
|---|---|---|---|
| `anchor-self` | 0.0 | 0.0 | 8.23 |
| `validated-medium` | 0.482 | 0.471 | 9.23 |
| `candidate-high` | 0.468 | 0.444 | 10.33 (worst) |

The weighted metric moved both numbers down slightly but **preserved the
same wrong ordering** — `candidate-high` still scores as less conflicted
despite the worse merge. So the magnitude-blindness theory was wrong, or at
least incomplete: weighting conflicts by contested magnitude isn't the fix.
Current best guess is that the real missing variable is each model's
*overall drift magnitude from base*, independent of sign conflict —
`Qwen2.5-0.5B-Instruct` likely underwent far heavier training than a plain
SFT run, making its update disproportionately large in an absolute sense,
which neither `conflict` nor `conflict_weighted` normalizes against.

**Second fix, and this one works.** `drift_magnitude`
(`backend/app/conflict/drift.py`) drops the conflict framing entirely and
measures relative update size instead: `(rms(diff_a) + rms(diff_b)) /
rms(base)` per tensor, aggregated the same way as the other metrics. It
doesn't ask whether the two updates disagree, only how much total change is
being forced into the merge relative to the weight's own scale. Re-running
the same three triples through the real engine (`score_model_pair`,
tensor-count-weighted average across layers):

| variant | conflict | conflict_weighted | drift_magnitude | perplexity |
|---|---|---|---|---|
| `anchor-self` | 0.0 | 0.0 | 0.0 | 8.23 |
| `validated-medium` | 0.482 | 0.471 | 0.1046 | 9.23 |
| `candidate-high` | 0.468 | 0.444 | 0.1564 | 10.33 (worst) |

`drift_magnitude` rank-orders all three correctly — exactly matching the
perplexity ordering, where both conflict metrics inverted the middle two.
It converges with the earlier whole-model observation that
`Qwen2.5-0.5B-Instruct`'s diff norm (36.2) plus Dolphin3.0's (34.05) sums to
a much larger combined update than Dolphin3.0 + Capybara-sft's (34.05 +
16.07), and that combined size — not sign agreement — is what tracked
perplexity degradation. Still only 3 real data points, so this isn't
statistically bulletproof, but it's a real, verified result: the conflict
score's blind spot (magnitude of total drift) has a working second metric
now, computed by the actual engine and covered by tests
(`backend/tests/test_drift.py`).

Reproduce: `eval/configs/*.yaml` are the mergekit configs,
`mergekit-yaml eval/configs/<name>.yaml eval/merged/<name> --cuda` runs the
merge (needs `pip install mergekit`; pin `transformers==4.46.3` — newer
4.x/5.x releases break mergekit 0.1.4's pydantic schema build with a
misleading `torch is not defined` error), then
`python eval/measure_perplexity.py`.

## Build order

Phase 0: setup, Phase 1: config parser, Phase 2: architecture sanity checks,
Phase 3: conflict score engine, Phase 4: visual diagram, Phase 5: proxy
evaluation against real merges (done — see above).
