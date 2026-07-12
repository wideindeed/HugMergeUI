"""7B validation round: mistralai/Mistral-7B-v0.1 family, run on a rented
24GB-VRAM cloud GPU (RTX 4090, Vast.ai).

Mirrors the llama3.2_3b methodology: a shared base with multiple
community fine-tunes, paired light-to-heavy to see whether
drift_magnitude/conflict track real perplexity damage at 7B scale the
same way they did at 3B.

Fine-tunes (all built from mistralai/Mistral-7B-v0.1):
  - openhermes: teknium/OpenHermes-2.5-Mistral-7B (instruction chat)
  - dolphin: cognitivecomputations/dolphin-2.1-mistral-7b (instruction chat)
  - metamath: meta-math/MetaMath-Mistral-7B (math continued fine-tune)
  - code: ajibawa-2023/Code-Mistral-7B (code continued fine-tune)

Pairs chosen to span light (chat+chat) to heavy (math+code) domain
shift, same reasoning as Round Eleven/Twelve on llama3.2_3b.

fp16 throughout. Runs directly (no subprocess isolation - single rented
box, single one-shot budget run; a crash just stops the loop, resumable
via the results JSON on restart).
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from app.conflict.engine import score_model_pair  # noqa: E402

from eval_texts import EVAL_TEXTS  # noqa: E402

BASE_REPO = "mistralai/Mistral-7B-v0.1"
FINETUNES = {
    "openhermes": "teknium/OpenHermes-2.5-Mistral-7B",
    "dolphin": "cognitivecomputations/dolphin-2.1-mistral-7b",
    "metamath": "abacusai/Fewshot-Metamath-OrcaVicuna-Mistral",
    "code": "ajibawa-2023/Code-Mistral-7B",
}

PAIRS = [
    ("openhermes", "dolphin"),
    ("openhermes", "metamath"),
    ("dolphin", "code"),
    ("metamath", "code"),
]

RESULTS_PATH = Path(__file__).resolve().parent / "phase_7b_mistral_results.json"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "phase_7b_mistral"
MERGED_DIR = Path(__file__).resolve().parent / "merged"


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


def avg_metric(layers: list[dict], field: str) -> float:
    total_weight = sum(layer["tensor_count"] for layer in layers)
    return sum(layer["tensor_count"] * layer[field] for layer in layers) / total_weight


def run_pair(name_a: str, name_b: str) -> dict:
    repo_a, repo_b = FINETUNES[name_a], FINETUNES[name_b]
    pair_name = f"mistral7b__{name_a}+{name_b}"

    config = {
        "merge_method": "ties",
        "base_model": BASE_REPO,
        "models": [
            {"model": repo_a, "parameters": {"weight": 0.5, "density": 0.5}},
            {"model": repo_b, "parameters": {"weight": 0.5, "density": 0.5}},
        ],
        "parameters": {"normalize": True},
        "dtype": "float16",
    }
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{pair_name}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    out_path = MERGED_DIR / pair_name
    print(f"[{pair_name}] merging...", flush=True)
    subprocess.run(["mergekit-yaml", str(config_path), str(out_path), "--cuda"], check=True)

    print(f"[{pair_name}] measuring perplexity...", flush=True)
    ppl = mean_perplexity(str(out_path.resolve()))

    print(f"[{pair_name}] scoring conflict metrics...", flush=True)
    scored = score_model_pair(BASE_REPO, repo_a, repo_b, density=0.5)
    layers = scored["layers"]

    result = {
        "family": "mistral7b",
        "pair": f"{name_a}+{name_b}",
        "model_a": repo_a,
        "model_b": repo_b,
        "conflict": avg_metric(layers, "conflict"),
        "conflict_weighted": avg_metric(layers, "conflict_weighted"),
        "drift_magnitude": avg_metric(layers, "drift_magnitude"),
        "perplexity": ppl,
    }
    print(f"[{pair_name}] done: {result}", flush=True)

    shutil.rmtree(out_path, ignore_errors=True)
    return result


def main() -> None:
    results: list[dict] = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"resuming: {len(results)} pair(s) already completed", flush=True)

    done_keys = {r["pair"] for r in results}

    for name_a, name_b in PAIRS:
        pair = f"{name_a}+{name_b}"
        if pair in done_keys:
            continue
        try:
            result = run_pair(name_a, name_b)
            results.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
        except Exception as e:
            print(f"[{pair}] FAILED: {e}", flush=True)
            shutil.rmtree(MERGED_DIR / f"mistral7b__{pair}", ignore_errors=True)
            continue

    print("\n=== ALL RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
