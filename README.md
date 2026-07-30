# HugMergeUI

[![CI](https://github.com/wideindeed/HugMergeUI/actions/workflows/ci.yml/badge.svg)](https://github.com/wideindeed/HugMergeUI/actions/workflows/ci.yml)

Checks whether two fine-tuned models are going to fight each other before you burn GPU hours merging them for real.

Point it at a mergekit config. It pulls the real weights, diffs them layer by layer, and shows where the two models agree and where they clash. The result renders as a 3D scene you can fly around and click into.

## Status: early beta

Solo side project, not a product. It's a diagnostic tool, not a merge engine: it doesn't run mergekit for you, it tells you if a merge is worth attempting. The app works end to end: config editor, architecture checks, model picker, 3D visualization, all wired up to real Hugging Face models.

`drift_magnitude` correlates with merge damage across every scale tested so far (0.36B to 7B params, nine model families, pearson r=0.56, spearman rho=0.59, p<1e-9, n=117 non-trivial merges, holds up when dropping any single family). Full methodology and every round, including the negative results and dead ends, is in `VALIDATION.txt`.

Splitting that correlation apart tells a more useful story than the headline number alone. Almost all of its predictive power comes from catching total blowups, merges that go on to have perplexity in the thousands or millions. On those, `drift_magnitude` works like a smoke detector: it reliably flags the fire before you light it. Set the disasters aside and look only at merges that came out fine, and the signal gets a lot weaker. Still real (spearman rho=0.32, p=0.001), but not strong enough to say one decent config will beat another (pearson r=0.18, not significant). Read it as "will this merge blow up," not "which of these two is slightly better."

`conflict` and `conflict_weighted` are shown in the app for comparison but haven't shown a real signal in any round tested. `drift_magnitude` is the number worth trusting.

## Quick start

### Backend

```
.venv\Scripts\activate
pip install -r backend/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126
uvicorn backend.app.main:app --port 8010 --reload
```

Windows note: use port 8010, not 8000. Port 8000 often falls inside Windows' reserved dynamic port range (Hyper-V/WSL) and fails to bind. The frontend dev proxy already points at 8010.

Check you actually got the CUDA build, not the CPU-only one:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Frontend

```
cd frontend
npm install
npm run dev
```

## Try it

Click "Try an example" in the app for a few ready-made model pairs, no config writing required. Or drop your own mergekit YAML into the editor.

## How it's built

- `backend/`: FastAPI. Parses mergekit configs, checks architecture compatibility, pulls safetensors weights from Hugging Face, scores sign-conflict and redundancy per layer.
- `frontend/`: React, TypeScript, Three.js. Config editor, architecture warnings, model picker, and a 3D layer-by-layer scene you can fly around and click into.

## Testing

`backend/tests/` (pytest) covers the scoring engine, architecture checks, and config resolution against real Hugging Face models and real weight tensors. No mocking, same philosophy as the validation work in `VALIDATION.txt`. `backend/` is also linted with `ruff`. `frontend/` is type-checked and linted (`tsc`, `oxlint`) in CI; it has no dedicated unit tests yet. Every push and PR to `main` runs all of this through GitHub Actions (`.github/workflows/ci.yml`), see the badge above.

## Does the score actually predict merge quality?

Yes, pooled across every scale and family tested so far (0.36B-7B params, nine model families, n=117 non-trivial merges), `drift_magnitude` holds pearson r=0.56 (r^2=0.32) and spearman rho=0.59, both p<1e-9, and that holds up when dropping any single family. Individual rounds at 1.5B-1.7B scale saw spearman rho as high as 0.80-0.96. Smaller scale (0.5B/360M) is mixed: one family (Qwen2.5) clears significance, another (SmolLM2) doesn't. `conflict` and `conflict_weighted` show no signal at any scale tested.

Worth being precise about what that pooled number means, though. Most of it comes from correctly flagging catastrophic merges, not from finely ranking merges that are all reasonably okay (see the split above). The full investigation, including what was tried and what didn't pan out, is in `VALIDATION.txt`.
