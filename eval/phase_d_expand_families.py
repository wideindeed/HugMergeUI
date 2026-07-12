"""Follow-up to phase_c_solo_baseline.py: the corrected (degradation vs. solo
baseline) per-family correlations were directionally right - qwen r=0.53,
smollm2 r=-0.36 after removing the component-quality confound - but neither
reaches significance at n=10 pairs per family. This isn't evidence the
metric fails, it's an underpowered test. Fix: add 3 more real community
fine-tunes to each family (found via HF Hub base_model:finetune search,
filtered to plain causal-LM SFT checkpoints, no quantized/format-converted
duplicates) and merge every new pair, growing each family from 5 fine-tunes
(10 pairs) to 8 fine-tunes (28 pairs).

Resumable: reads big_validation_results.json and phase_c_solo_results.json,
only runs pairs/solo-baselines not already present, appends and
checkpoints after each. Reuses run_pair()/mean_perplexity() from
big_validation.py and FAMILIES now includes the new fine-tunes.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from big_validation import FAMILIES, RESULTS_PATH, run_pair, mean_perplexity  # noqa: E402
from scipy import stats  # noqa: E402

SOLO_PATH = Path(__file__).resolve().parent / "phase_c_solo_results.json"


def main() -> None:
    results: list[dict] = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"resuming: {len(results)} pair(s) already banked", flush=True)
    done_pairs = {(r["family"], r["pair"]) for r in results}

    solo: dict[str, float] = {}
    if SOLO_PATH.exists():
        solo = json.loads(SOLO_PATH.read_text())
        print(f"resuming: {len(solo)} solo repo(s) already measured", flush=True)

    # solo baselines for the new fine-tunes (skips ones already banked)
    for family_name, spec in FAMILIES.items():
        base_repo = spec["base"]
        for repo in spec["finetunes"].values():
            if repo in solo:
                continue
            print(f"[solo] {repo}...", flush=True)
            try:
                solo[repo] = mean_perplexity(repo)
            except OSError:
                print(f"  (no tokenizer shipped, falling back to {base_repo})", flush=True)
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch, math
                tokenizer = AutoTokenizer.from_pretrained(base_repo)
                model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to("cuda")
                model.eval()
                from eval_texts import EVAL_TEXTS
                losses = []
                with torch.no_grad():
                    for text in EVAL_TEXTS:
                        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
                        out = model(input_ids, labels=input_ids)
                        losses.append(out.loss.item())
                del model
                torch.cuda.empty_cache()
                solo[repo] = math.exp(sum(losses) / len(losses))
            SOLO_PATH.write_text(json.dumps(solo, indent=2))
            print(f"[solo] {repo}: {solo[repo]}", flush=True)

    # every pair not already banked
    for family_name, spec in FAMILIES.items():
        base_repo = spec["base"]
        finetunes = spec["finetunes"]
        for (name_a, repo_a), (name_b, repo_b) in itertools.combinations(finetunes.items(), 2):
            pair = f"{name_a}+{name_b}"
            if (family_name, pair) in done_pairs:
                continue
            try:
                result = run_pair(family_name, base_repo, name_a, repo_a, name_b, repo_b)
                results.append(result)
                RESULTS_PATH.write_text(json.dumps(results, indent=2))
            except Exception as e:
                print(f"[{family_name}/{pair}] FAILED: {e}", flush=True)
                continue

    print("\n=== PER-FAMILY CORRELATIONS: degradation vs. best solo baseline (expanded) ===")
    for fam in FAMILIES:
        rows = [r for r in results if r["family"] == fam and r["pair"] != "base+base2"]
        degradation = [r["perplexity"] - max(solo[r["model_a"]], solo[r["model_b"]]) for r in rows]

        print(f"\n{fam} (n={len(rows)})")
        for metric in ["conflict", "conflict_weighted", "drift_magnitude"]:
            vals = [r[metric] for r in rows]
            pr, pp = stats.pearsonr(vals, degradation)
            sr, sp = stats.spearmanr(vals, degradation)
            print(
                f"{metric}: pearson r={pr:.4f} (p={pp:.4g}) | "
                f"spearman rho={sr:.4f} (p={sp:.4g})"
            )
        for r, d in zip(rows, degradation):
            print(f"  {r['pair']:30s} drift={r['drift_magnitude']:.4f} degradation={d:+.3f}")


if __name__ == "__main__":
    main()
