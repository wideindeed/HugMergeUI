"""Timing/feasibility pilot for a single merge+score at 3B scale, using
float16 merge dtype instead of the float32 used at 0.5B-1.7B so far. 3B in
fp32 is right at the edge of what fits comfortably in 8GB VRAM alongside a
base model for scoring; fp16 halves that footprint. Not a validation
result by itself - n=1, just confirms fp16-at-3B fits and gets real
wall-clock numbers before committing to a full family pass.
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

BASE = "Qwen/Qwen2.5-3B"
MODEL_A = "Qwen/Qwen2.5-3B-Instruct"
MODEL_B = "Qwen/Qwen2.5-Coder-3B"

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "pilot_3b.yaml"
OUT_PATH = Path(__file__).resolve().parent / "merged" / "pilot_3b"


def mean_perplexity(model_path: str, device: str = "cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
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
        "dtype": "float16",
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
    drift = sum(layer["tensor_count"] * layer["drift_magnitude"] for layer in layers) / total_weight
    timings["conflict_score_seconds"] = time.time() - t0
    print(
        f"[conflict_score] done in {timings['conflict_score_seconds']:.1f}s, "
        f"conflict={conflict:.4f}, drift_magnitude={drift:.4f}",
        flush=True,
    )

    shutil.rmtree(OUT_PATH, ignore_errors=True)

    timings["total_seconds"] = time.time() - t_total_start
    result = {
        "base": BASE,
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "perplexity": ppl,
        "conflict": conflict,
        "drift_magnitude": drift,
        "timings": timings,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))

    (Path(__file__).resolve().parent / "pilot_3b_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
