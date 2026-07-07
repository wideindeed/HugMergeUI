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
uvicorn backend.app.main:app --reload
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
- Two independent fine-tunes of the same base model (TBD which pair, pick
  when Phase 3 needs a realistic sign-conflict test case) — the actual
  target scenario, since independently-drifted fine-tunes are where
  conflict is expected to show up, unlike a base/instruct pair from the
  same lineage

## Build order

Phase 0: setup, Phase 1: config parser, Phase 2: architecture sanity checks,
Phase 3: conflict score engine, Phase 4: visual diagram, Phase 5: proxy
evaluation (stretch goal).
