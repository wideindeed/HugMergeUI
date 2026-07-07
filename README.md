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
uvicorn backend.app.main:app --reload
```

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

## Build order

Phase 0: setup, Phase 1: config parser, Phase 2: architecture sanity checks,
Phase 3: conflict score engine, Phase 4: visual diagram, Phase 5: proxy
evaluation (stretch goal).
