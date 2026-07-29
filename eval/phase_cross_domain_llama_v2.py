"""Round Fourteen's second attempt: Round Fourteen's first cross-domain
pair on llama3.2_3b (MergeBench coding+math) turned out to be a
weak-fine-tune confound, quantitatively confirmed by
phase_cross_domain_llama_selfcheck.py (both checkpoints moved the
weights ~10-15x less than every other domain fine-tune tested on this
family). This pair replaces both candidates with the heaviest options
available on this base at this scale, verified via
phase_cross_domain_llama_v2_selfcheck.py BEFORE committing to this run:

  - LLM360/MegaMath-Llama-3.2-3B: a genuine continued-pretraining
    checkpoint (300B+ token MegaMath corpus, not an instruction SFT).
    Solo drift_magnitude vs base = 0.2574 - the heaviest single-model
    domain shift measured on this family so far (previous max was
    EpistemeAI's coder fine-tune at 0.1752; dolphin/home/korean were
    ~0.11-0.15; the MergeBench pair that motivated this follow-up was
    0.0064/0.0158).
  - EpistemeAI/Llama-3.2-3B-Agent007-Coder: Round Twelve's code-domain
    pick, kept for direct comparability with that round's numbers.

Same subprocess-isolation design and fp16 methodology (this project's
8GB-card precedent for 3B pairs) as phase_cross_domain_llama.py.

Originally ran from a D:\ drive copy (VALIDATION.txt "ROUND FOURTEEN,
CONTINUED") with family_name "llama3.2_3b_cross_domain_v2"; renamed to
"llama3.2_3b" here to pool with the rest of that family's data in
phase_scale_3b_results.json, phase_trend_llama_results.json, and
phase_pooled_meta.py.
"""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.conflict.engine import score_model_pair  # noqa: E402

from eval_texts import EVAL_TEXTS  # noqa: E402

PAIRS = [
    dict(
        family_name="llama3.2_3b",
        base_repo="unsloth/Llama-3.2-3B",
        name_a="megamath",
        repo_a="LLM360/MegaMath-Llama-3.2-3B",
        name_b="episteme_coder",
        repo_b="EpistemeAI/Llama-3.2-3B-Agent007-Coder",
    ),
]

RESULTS_PATH = Path(__file__).resolve().parent / "phase_cross_domain_llama_v2_results.json"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "cross_domain_llama_v2"
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


def run_pair_isolated(spec: dict, timeout_s: int = 1800, retries: int = 1) -> dict | None:
    pair_name = f"{spec['family_name']}__{spec['name_a']}+{spec['name_b']}"

    for attempt in range(1, retries + 2):
        shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            worker_spec = {**spec, "out_path": str(out_path)}
            worker_script = Path(__file__).resolve().parent / "phase_cross_domain_llama_v2_worker.py"
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

    for spec in PAIRS:
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


if __name__ == "__main__":
    main()
