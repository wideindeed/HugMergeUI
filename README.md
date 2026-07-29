# HugMergeUI

See whether two fine-tuned models are going to fight each other before you burn GPU hours merging them for real.

Point it at a mergekit config. It pulls the real weights, diffs them layer by layer, and shows where the two models agree and where they clash. The result renders as a 3D orbiting system you can fly around, click into, and inspect layer by layer.

## Status: early beta

This is a solo side project, not a product, and it's a diagnostic tool, not a merge engine — it doesn't run mergekit for you, it tells you whether a merge is worth attempting before you do. The app works end to end: config editor, architecture checks, model picker, 3D visualization, all wired up against real Hugging Face models.

`drift_magnitude` correlates with merge damage across the full range tested so far, 0.36B-7B params, pooled across nine model families (pearson r=0.56, spearman rho=0.59, p<1e-9, n=117 non-trivial merges, robust to leave-one-family-out). Full methodology, every individual round including the negative results and dead ends, and the pooled meta-analysis are in `VALIDATION.txt`. There is a modest, not-yet-fully-confirmed signal that larger models absorb merge conflict better than the pooled trend alone predicts (see Round Seventeen); this is an active area, not settled. `conflict` and `conflict_weighted` are shown in the app for comparison but have not shown a real signal in any round tested — `drift_magnitude` is the number worth trusting.

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

- `backend/` — FastAPI. Parses mergekit configs, checks architecture compatibility, pulls safetensors weights from Hugging Face, scores sign-conflict and redundancy per layer.
- `frontend/` — React, TypeScript, Three.js. Config editor, architecture warnings, model picker, and a 3D layer-by-layer scene you can fly around and click into.

## Does the score actually predict merge quality?

Yes — pooled across every scale and family tested so far (0.36B-7B params, nine model families, n=117 non-trivial merges), `drift_magnitude` holds a pearson r=0.56 (r^2=0.32) and spearman rho=0.59, both p<1e-9, robust to dropping any single family. Individual rounds at 1.5B-1.7B scale saw spearman rho as high as 0.80-0.96; smaller scale (0.5B/360M) is mixed — one family (Qwen2.5) clears significance, another (SmolLM2) doesn't. `conflict` and `conflict_weighted` show no signal at any scale tested. The full investigation, including what was tried and what didn't pan out, is in `VALIDATION.txt`.
