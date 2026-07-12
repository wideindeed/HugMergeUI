"""Large-scale Phase 5 validation: every unique pair of fine-tunes within
two unrelated model families (Qwen2.5-0.5B, SmolLM2-360M) is actually
merged with mergekit and measured for real perplexity, then correlated
against our own conflict / conflict_weighted / drift_magnitude scores for
that same (base, model_a, model_b) triple.

Runs merges sequentially (one GPU) and checkpoints results to
big_validation_results.json after every pair, so a partial run still
leaves usable data. Merged model directories are deleted right after
perplexity is measured - eval/merged/ is gitignored and fully
reproducible from eval/configs/big/*.yaml.
"""

import itertools
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.conflict.engine import score_model_pair  # noqa: E402

from eval_texts import EVAL_TEXTS  # noqa: E402

FAMILIES = {
    "qwen": {
        "base": "Qwen/Qwen2.5-0.5B",
        "finetunes": {
            "dolphin": "dphn/Dolphin3.0-Qwen2.5-0.5B",
            "capybara": "wulli/Qwen2.5-0.5B-sft-capybara",
            "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "jayhyeon": "JayHyeon/Qwen2.5-0.5B-SFT",
            "qgallouedec": "qgallouedec/Qwen2.5-0.5B-SFT",
            "nuextract": "numind/NuExtract-1.5-tiny",
            "gsm8k": "OhhMoo/qwen05b-gsm8k-sft-instruct",
            "bcarr92": "BCarr92/Qwen2.5-0.5B-SFT",
        },
    },
    "smollm2": {
        "base": "HuggingFaceTB/SmolLM2-360M",
        "finetunes": {
            "instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "michaelj1": "Michaelj1/finetune-smolLM2-360M",
            "cot": "prithivMLmods/SmolLM2-CoT-360M",
            "rickified": "Masorian06/SmolLM2-360M-Rickified",
            "michaelj1_wikitext": "Michaelj1/INSTRUCT_smolLM2-360M-finetuned-wikitext2-raw-v1",
            "gsm8k": "qiuyu8290/SmolLM2-360M-sft-math-gsm8k",
            "instructmath": "srmty/smolLM2-360M-instruct-math-v1",
            "everyday": "lewtun/SmolLM2-360M-SFT-everyday-conversations",
        },
    },
}

RESULTS_PATH = Path(__file__).resolve().parent / "big_validation_results.json"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "big"
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


def main() -> None:
    results: list[dict] = []

    for family_name, spec in FAMILIES.items():
        base_repo = spec["base"]
        finetunes = spec["finetunes"]

        results.append(run_pair(family_name, base_repo, "base", base_repo, "base2", base_repo))
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

        for (name_a, repo_a), (name_b, repo_b) in itertools.combinations(finetunes.items(), 2):
            results.append(run_pair(family_name, base_repo, name_a, repo_a, name_b, repo_b))
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
