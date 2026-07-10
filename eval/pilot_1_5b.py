"""Timing pilot for a single merge+score at 1.5B scale (3x the 0.5B models
used in Phase 5/6), to get real wall-clock numbers on this hardware before
committing to a full n=22-style pass. Not a validation result by itself -
n=1, just a timing and feasibility check (does fp32 fit in 8GB VRAM at
this size, roughly how long does each stage take).
"""

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_texts import EVAL_TEXTS

BASE = "Qwen/Qwen2.5-1.5B"
MODEL_A = "dphn/Dolphin3.0-Qwen2.5-1.5B"
MODEL_B = "Qwen/Qwen2.5-1.5B-Instruct"

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "pilot_1_5b.yaml"
OUT_PATH = Path(__file__).resolve().parent / "merged" / "pilot_1_5b"


def mean_perplexity(model_path: str, device: str = "cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for text in EVAL_TEXTS:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = model(input_ids, labels=input_ids)
            losses.append(out.loss.item())

    del model
    torch.cuda.empty_cache()
    return math.exp(sum(losses) / len(losses))


def main() -> None:
    timings = {}
    t_total_start = time.time()

    config = {
        "merge_method": "ties",
        "base_model": BASE,
        "models": [
            {"model": MODEL_A, "parameters": {"weight": 0.5, "density": 0.5}},
            {"model": MODEL_B, "parameters": {"weight": 0.5, "density": 0.5}},
        ],
        "parameters": {"normalize": True},
        "dtype": "float32",
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

    print("[merge] starting...", flush=True)
    t0 = time.time()
    subprocess.run(["mergekit-yaml", str(CONFIG_PATH), str(OUT_PATH), "--cuda"], check=True)
    timings["merge_seconds"] = time.time() - t0
    print(f"[merge] done in {timings['merge_seconds']:.1f}s", flush=True)

    print("[perplexity] starting...", flush=True)
    t0 = time.time()
    ppl = mean_perplexity(str(OUT_PATH.resolve()))
    timings["perplexity_seconds"] = time.time() - t0
    print(f"[perplexity] done in {timings['perplexity_seconds']:.1f}s, ppl={ppl:.4f}", flush=True)

    print("[conflict_score] starting...", flush=True)
    t0 = time.time()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.conflict.engine import score_model_pair  # noqa: E402

    scored = score_model_pair(BASE, MODEL_A, MODEL_B, density=0.5)
    layers = scored["layers"]
    total_weight = sum(layer["tensor_count"] for layer in layers)
    conflict = sum(layer["tensor_count"] * layer["conflict"] for layer in layers) / total_weight
    timings["conflict_score_seconds"] = time.time() - t0
    print(f"[conflict_score] done in {timings['conflict_score_seconds']:.1f}s, conflict={conflict:.4f}", flush=True)

    shutil.rmtree(OUT_PATH, ignore_errors=True)

    timings["total_seconds"] = time.time() - t_total_start
    result = {
        "base": BASE,
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "perplexity": ppl,
        "conflict": conflict,
        "timings": timings,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))

    (Path(__file__).resolve().parent / "pilot_1_5b_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
