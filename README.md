# HugMergeUI

See whether two fine-tuned models are going to fight each other before you burn GPU hours merging them for real.

Point it at a mergekit config. It pulls the real weights, diffs them layer by layer, and shows where the two models agree and where they clash. The result renders as a 3D orbiting system you can fly around, click into, and inspect layer by layer.

## Status: early beta

This is a solo side project, not a product, and it's a diagnostic tool, not a merge engine — it doesn't run mergekit for you, it tells you whether a merge is worth attempting before you do. The app works end to end: config editor, architecture checks, model picker, 3D visualization, all wired up against real Hugging Face models.

The `drift_magnitude` score is validated at 1.5B+ params — it correlates strongly with real post-merge perplexity (spearman rho up to 0.96 across 29 pairs, three model families, several rounds of replication). It is **not** validated at smaller scale (0.5B/360M), where the same tests found no signal at all. The example presets in the app are now all 1.5B+ for exactly this reason — full details, including what was tried and rejected, in `VALIDATION.txt`.

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

Yes, at 1.5B+ params — tested against real merges and real perplexity across 29 model pairs spanning three architecture families (Qwen2.5, SmolLM2, Llama 3.2), with `drift_magnitude` holding a spearman rank correlation of 0.80-0.96 depending on the cut. It does not hold at 0.5B/360M scale (tested separately, no significant signal there). The full investigation, including what was tried and what didn't pan out, is in `VALIDATION.txt`.
