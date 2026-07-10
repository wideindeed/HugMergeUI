"""Scale-up retest of the raw-weight-diff conflict metric family (conflict,
conflict_weighted, drift_magnitude - all abandoned at 0.5B/360M per
VALIDATION.txt) at ~3x model size, to check whether the saturation problem
is scale-dependent or not.

Same design as big_validation.py (merge every pair within a family with
mergekit, measure real perplexity, score with the real engine, correlate),
scoped down from 5 fine-tunes/family to 4 to keep runtime reasonable at
this size: 6 non-trivial pairs + 1 self-anchor per family, 2 families, 14
points total (vs. 22 in the original pass). All fine-tune repos below are
official (Qwen team, HuggingFaceTB team) or well-known community SFT/Tulu
recipes, verified to exist via the HF API before this run.
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

FAMILIES = {
    "qwen1.5b": {
        "base": "Qwen/Qwen2.5-1.5B",
        "finetunes": {
            "dolphin": "dphn/Dolphin3.0-Qwen2.5-1.5B",
            "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "coder": "Qwen/Qwen2.5-Coder-1.5B",
            "math": "Qwen/Qwen2.5-Math-1.5B",
            # added in the round-four scale-up pass: a second "healthy" (non
            # domain-divergent) fine-tune, to shore up the thin healthy subset
            "capybara": "lewtun/Qwen2.5-1.5B-SFT-Capybara-No-Packing",
        },
    },
    "smollm2_1.7b": {
        "base": "HuggingFaceTB/SmolLM2-1.7B",
        "finetunes": {
            "instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "sft_only": "HuggingFaceTB/SmolLM2-1.7B-sft-only",
            "smoltulu": "SultanR/SmolTulu-1.7b-Instruct",
            "tulu3": "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
            # added in the round-four scale-up pass: a code-domain SFT, to see
            # whether the catastrophic-perplexity domain-divergence effect
            # seen in qwen1.5b (coder/math pairs) replicates on this family
            "code": "CodeAtCMU/SmolLM2-1.7B_full_sft_code_data_120K",
        },
    },
    "llama3.2_1b": {
        # unsloth mirror used as base: meta-llama/Llama-3.2-1B is gated and
        # requires an accepted-license HF token we don't have configured;
        # unsloth's re-upload is bit-identical, ungated, and used as the base
        # for several of the fine-tunes below anyway.
        "base": "unsloth/Llama-3.2-1B",
        "finetunes": {
            "dolphin": "dphn/Dolphin3.0-Llama3.2-1B",
            "instruct": "unsloth/Llama-3.2-1B-Instruct",  # ungated mirror of meta-llama's official instruct
            "sft": "RLHFlow/LLaMA3.2-1B-SFT",
            "capybara": "lewtun/Llama-3.2-1B-SFT-Capybara-No-Packing-Llama",
        },
    },
}

RESULTS_PATH = Path(__file__).resolve().parent / "phase_scale_1_5b_results.json"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "scale_1_5b"
MERGED_DIR = Path(__file__).resolve().parent / "merged"


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
        "dtype": "float32",
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
    """Enumerate every (family, pair) job the full run consists of, in order."""
    specs = []
    for family_name, spec in FAMILIES.items():
        base_repo = spec["base"]
        finetunes = spec["finetunes"]

        specs.append(
            dict(family_name=family_name, base_repo=base_repo, name_a="base", repo_a=base_repo,
                 name_b="base2", repo_b=base_repo)
        )
        for (name_a, repo_a), (name_b, repo_b) in itertools.combinations(finetunes.items(), 2):
            specs.append(
                dict(family_name=family_name, base_repo=base_repo, name_a=name_a, repo_a=repo_a,
                     name_b=name_b, repo_b=repo_b)
            )
    return specs


def run_pair_isolated(spec: dict, timeout_s: int = 1800, retries: int = 1) -> dict | None:
    """Run one pair in its own subprocess so a hard crash (OOM, driver crash,
    mergekit segfault) can't take down the driver's CUDA context or leak RAM
    into the next pair. Returns None (after logging) if the pair still fails
    after retries, so the overall run can keep going.
    """
    pair_name = f"{spec['family_name']}__{spec['name_a']}+{spec['name_b']}"

    for attempt in range(1, retries + 2):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            worker_spec = {**spec, "out_path": str(out_path)}
            worker_script = Path(__file__).resolve().parent / "phase_scale_1_5b_worker.py"
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
                # give the OS/driver a moment to fully reclaim GPU memory before retrying
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
