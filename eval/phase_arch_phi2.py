"""Broadening axis (b): every family tested so far (Qwen2.5, SmolLM2,
Llama3.2) is Llama-style transformer architecture. This is the first test
on a genuinely different architecture: microsoft/phi-2 (2.7B, MIT
license, ungated, parallel attention+MLP blocks rather than the
sequential Llama-style block). Checks whether drift_magnitude's signal
is architecture-specific or generalizes.

3 independently-trained community fine-tunes off the same base (not a
parent/child chain - avoiding Round Six/Seven's confound on purpose):
  - dolphin: cognitivecomputations/dolphin-2_6-phi-2 (general SFT)
  - instruct: venkycs/phi-2-instruct (general SFT)
  - coder: mrm8488/phi-2-coder (code-domain SFT - the "domain-divergent"
    category from Round Four/Five, included to see if the catastrophic
    perplexity pattern also generalizes across architectures)

3 non-trivial pairs + 1 self-anchor, 4 points. fp16 throughout (merge +
perplexity), same as phase_scale_3b.py, since phi-2 at fp32 doesn't
comfortably fit two-model-loaded scoring on an 8GB card. Same
subprocess-per-pair + resume-from-checkpoint + clean-retry design as
phase_scale_3b.py / phase_hierarchy.py.
"""

import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.conflict.engine import score_model_pair  # noqa: E402

from eval_texts import EVAL_TEXTS  # noqa: E402

BASE = "microsoft/phi-2"
FINETUNES = {
    "dolphin": "cognitivecomputations/dolphin-2_6-phi-2",
    "instruct": "venkycs/phi-2-instruct",
    "coder": "mrm8488/phi-2-coder",
}
FAMILY_NAME = "phi2"

RESULTS_PATH = Path(__file__).resolve().parent / "phase_arch_phi2_results.json"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "arch_phi2"
MERGED_DIR = Path(__file__).resolve().parent / "merged"


def mean_perplexity(model_path: str, device: str = "cuda") -> float:
    # The pre-2024 community phi-2 fine-tunes (dolphin/coder/instruct) ship
    # configs with an auto_map pointing at microsoft/phi-2's own legacy
    # modeling_phi.py (pre-native-support era) - manually reviewed
    # (Copyright Microsoft/Tri Dao, MIT/BSD-3, plain PyTorch, no network or
    # subprocess calls) before enabling this.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
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


def run_pair(family_name: str, base_repo: str, name_a: str, repo_a: str, name_b: str, repo_b: str) -> dict:
    pair_name = f"{family_name}__{name_a}+{name_b}"
    config = {
        "merge_method": "ties",
        "base_model": base_repo,
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
    subprocess.run(
        ["mergekit-yaml", str(config_path), str(out_path), "--cuda", "--trust-remote-code"], check=True
    )

    print(f"[{pair_name}] measuring perplexity...", flush=True)
    ppl = mean_perplexity(str(out_path.resolve()))

    print(f"[{pair_name}] scoring conflict metrics...", flush=True)
    scored = score_model_pair(base_repo, repo_a, repo_b, density=0.5)
    layers = scored["layers"]

    result = {
        "family": family_name,
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


def all_pair_specs() -> list[dict]:
    specs = [
        dict(family_name=FAMILY_NAME, base_repo=BASE, name_a="base", repo_a=BASE, name_b="base2", repo_b=BASE)
    ]
    for (name_a, repo_a), (name_b, repo_b) in itertools.combinations(FINETUNES.items(), 2):
        specs.append(
            dict(family_name=FAMILY_NAME, base_repo=BASE, name_a=name_a, repo_a=repo_a, name_b=name_b, repo_b=repo_b)
        )
    return specs


def run_pair_isolated(spec: dict, timeout_s: int = 1800, retries: int = 1) -> dict | None:
    pair_name = f"{spec['family_name']}__{spec['name_a']}+{spec['name_b']}"

    for attempt in range(1, retries + 2):
        shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            worker_spec = {**spec, "out_path": str(out_path)}
            worker_script = Path(__file__).resolve().parent / "phase_arch_phi2_worker.py"
            print(f"[{pair_name}] launching worker (attempt {attempt})...", flush=True)
            try:
                proc = subprocess.run(
                    [sys.executable, str(worker_script), json.dumps(worker_spec)],
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                print(f"[{pair_name}] TIMED OUT after {timeout_s}s, treating as failed", flush=True)
                continue

            if proc.returncode != 0:
                print(f"[{pair_name}] worker crashed (exit {proc.returncode})", flush=True)
                time.sleep(10)
                continue

            if not out_path.exists():
                print(f"[{pair_name}] worker exited 0 but wrote no result, treating as failed", flush=True)
                continue

            result = json.loads(out_path.read_text())
            print(f"[{pair_name}] done: {result}", flush=True)
            shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
            return result

    print(f"[{pair_name}] giving up after {retries + 1} attempt(s)", flush=True)
    shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
    return None


def main() -> None:
    results: list[dict] = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"resuming: {len(results)} pair(s) already completed", flush=True)

    done_keys = {(r["family"], r["pair"]) for r in results}

    for spec in all_pair_specs():
        pair = f"{spec['name_a']}+{spec['name_b']}"
        if (spec["family_name"], pair) in done_keys:
            continue

        result = run_pair_isolated(spec)
        if result is not None:
            results.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== ALL RESULTS ===")
    for r in results:
        print(r)

    perplexities = [r["perplexity"] for r in results]
    print(f"\n=== CORRELATIONS (n={len(results)}) ===")
    for metric in ["conflict", "conflict_weighted", "drift_magnitude"]:
        values = [r[metric] for r in results]
        pearson_r, pearson_p = stats.pearsonr(values, perplexities)
        spearman_r, spearman_p = stats.spearmanr(values, perplexities)
        print(
            f"{metric}: pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
            f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
        )


if __name__ == "__main__":
    main()
