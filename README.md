# HugMergeUI

See whether two fine-tuned models are going to fight each other before you burn GPU hours merging them for real.

Point it at a mergekit config. It pulls the real weights, diffs them layer by layer, and shows where the two models agree and where they clash. The result renders as a 3D orbiting system you can fly around, click into, and inspect layer by layer.

## Status: early beta

This is a solo side project, not a product. The app works end to end: config editor, architecture checks, model picker, 3D visualization, all wired up against real Hugging Face models. The conflict score itself is still a heuristic, not a validated quality predictor, full details in `VALIDATION.txt`. Expect rough edges, and better scoring methods are on the way.

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

Tested against real merges, real perplexity, and real task accuracy (MMLU, GSM8K), across 28 model pairs. Short answer: not reliably yet, at least not at small model scale. The full investigation, including what was tried and what didn't pan out, is in `VALIDATION.txt`.
