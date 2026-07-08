"""Phase 6, Phase A: does task-accuracy (MMLU/GSM8K) correlate with our
conflict metrics where perplexity didn't? Tests whether perplexity itself
was the wrong ruler (aligned/instruct models often show *higher*
perplexity on generic text than their base despite being functionally
better - "alignment tax").

Reuses the mergekit configs already written to eval/configs/big/*.yaml by
big_validation.py / phase_b_domain_divergent.py - no remerging logic
duplicated. Reuses conflict/conflict_weighted/drift_magnitude/perplexity
already computed in big_validation_results.json and phase_b_results.json,
joined by (family, pair) - only the MMLU/GSM8K accuracy is new work here.

This is a PARTIAL run (PAIRS below), not the full 22+6 set - local
hardware makes the full set impractically slow (see PHASE6_STATUS.txt).
It's meant to give an early directional signal before committing to
rented-GPU compute for the full run. Checkpoints to
phase_a_results.json after every pair.
"""

import json
import subprocess
import sys
from pathlib import Path

import lm_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from big_validation import CONFIGS_DIR, MERGED_DIR  # noqa: E402
import shutil  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "phase_a_results.json"

# Small, diverse subset: spans both families, spans the full conflict
# range (0 -> 0.50), spans the perplexity range, includes one
# domain-divergent pair. Picked from big_validation_results.json /
# phase_b_results.json - see chat log for the selection table.
PAIRS = [
    ("qwen", "base+base2"),
    ("qwen", "capybara+qgallouedec"),
    ("qwen", "dolphin+instruct"),
    ("smollm2", "michaelj1+cot"),
    ("qwen", "math+chat"),
]

MMLU_NUM_FEWSHOT = 5
MMLU_LIMIT = 3  # per-subject question cap, keeps full 57-subject MMLU tractable
GSM8K_NUM_FEWSHOT = 8
GSM8K_LIMIT = 15


def load_existing_metrics() -> dict[tuple[str, str], dict]:
    by_key = {}
    for path in [
        Path(__file__).resolve().parent / "big_validation_results.json",
        Path(__file__).resolve().parent / "phase_b_results.json",
    ]:
        if path.exists():
            for r in json.loads(path.read_text()):
                by_key[(r["family"], r["pair"])] = r
    return by_key


def run_task_eval(family: str, pair: str) -> dict:
    pair_name = f"{family}__{pair}"
    config_path = CONFIGS_DIR / f"{pair_name}.yaml"
    out_path = MERGED_DIR / pair_name

    print(f"[{pair_name}] merging...", flush=True)
    subprocess.run(["mergekit-yaml", str(config_path), str(out_path), "--cuda"], check=True)

    model_args = f"pretrained={out_path.resolve()},dtype=float32"

    print(f"[{pair_name}] running MMLU (limit={MMLU_LIMIT}/subject, {MMLU_NUM_FEWSHOT}-shot)...", flush=True)
    mmlu_res = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["mmlu"],
        num_fewshot=MMLU_NUM_FEWSHOT,
        limit=MMLU_LIMIT,
        batch_size="auto",
        device="cuda",
        log_samples=False,
    )
    mmlu_acc = mmlu_res["results"]["mmlu"]["acc,none"]

    print(f"[{pair_name}] running GSM8K (limit={GSM8K_LIMIT}, {GSM8K_NUM_FEWSHOT}-shot)...", flush=True)
    gsm8k_res = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["gsm8k"],
        num_fewshot=GSM8K_NUM_FEWSHOT,
        limit=GSM8K_LIMIT,
        batch_size="auto",
        device="cuda",
        log_samples=False,
    )
    gsm8k_key = "exact_match,strict-match" if "exact_match,strict-match" in gsm8k_res["results"]["gsm8k"] else "exact_match,flexible-extract"
    gsm8k_acc = gsm8k_res["results"]["gsm8k"][gsm8k_key]

    shutil.rmtree(out_path, ignore_errors=True)

    result = {"family": family, "pair": pair, "mmlu_acc": mmlu_acc, "gsm8k_acc": gsm8k_acc}
    print(f"[{pair_name}] done: {result}", flush=True)
    return result


def main() -> None:
    existing = load_existing_metrics()
    results: list[dict] = []

    for family, pair in PAIRS:
        task_result = run_task_eval(family, pair)
        base = existing.get((family, pair), {})
        merged = {
            **task_result,
            "conflict": base.get("conflict"),
            "conflict_weighted": base.get("conflict_weighted"),
            "drift_magnitude": base.get("drift_magnitude"),
            "perplexity": base.get("perplexity"),
        }
        results.append(merged)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== PHASE A RESULTS (partial, n={}) ===".format(len(results)))
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
