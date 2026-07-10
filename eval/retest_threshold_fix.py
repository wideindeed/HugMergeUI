"""Cheap retest of the thresholded sign-conflict fix proposed in
VALIDATION.txt, against the 28 model pairs already validated in Phase 5/6
(big_validation_results.json + phase_b_results.json).

No re-merging and no re-measuring perplexity: those numbers are already on
disk and were expensive to produce. This only re-downloads/re-scores the
raw weight tensors with thresholded_sign_conflict_rate at a handful of
threshold_frac values and re-checks the correlation against the existing
perplexity column.
"""

import json
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.conflict.engine import _load_tensors, _resolve_tensor_files  # noqa: E402
from app.conflict.sign_conflict import thresholded_sign_conflict_rate  # noqa: E402

FAMILY_BASE = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "smollm2": "HuggingFaceTB/SmolLM2-360M",
}

THRESHOLD_FRACS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

RESULTS_PATH = Path(__file__).resolve().parent / "retest_threshold_fix_results.json"


def score_pair(base_repo: str, repo_a: str, repo_b: str, threshold_fracs: list[float]) -> dict[float, float]:
    base_files = _resolve_tensor_files(base_repo)
    a_files = _resolve_tensor_files(repo_a)
    b_files = _resolve_tensor_files(repo_b)
    common = set(base_files) & set(a_files) & set(b_files)

    base = _load_tensors(base_files, common)
    model_a = _load_tensors(a_files, common)
    model_b = _load_tensors(b_files, common)

    totals = {frac: [0.0, 0] for frac in threshold_fracs}  # frac -> [weighted_sum, weight]
    for name in common:
        base_t, a_t, b_t = base[name], model_a[name], model_b[name]
        if base_t.shape != a_t.shape or base_t.shape != b_t.shape:
            continue
        diff_a = (a_t - base_t).float()
        diff_b = (b_t - base_t).float()
        weight = diff_a.numel()
        for frac in threshold_fracs:
            score = thresholded_sign_conflict_rate(diff_a, diff_b, threshold_frac=frac)
            totals[frac][0] += weight * score
            totals[frac][1] += weight

    return {frac: (num / den if den else 0.0) for frac, (num, den) in totals.items()}


def main() -> None:
    pairs = json.loads((Path(__file__).resolve().parent / "big_validation_results.json").read_text())
    pairs += json.loads((Path(__file__).resolve().parent / "phase_b_results.json").read_text())

    results = []
    for r in pairs:
        base_repo = FAMILY_BASE[r["family"]]
        print(f"[{r['family']}/{r['pair']}] scoring...", flush=True)
        scores = score_pair(base_repo, r["model_a"], r["model_b"], THRESHOLD_FRACS)
        entry = {
            "family": r["family"],
            "pair": r["pair"],
            "perplexity": r["perplexity"],
            "old_conflict": r["conflict"],
            **{f"thresholded_{frac}": score for frac, score in scores.items()},
        }
        results.append(entry)
        print(f"[{r['family']}/{r['pair']}] {entry}", flush=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    perplexities = [r["perplexity"] for r in results]
    print(f"\n=== CORRELATIONS (n={len(results)}) ===")

    old_values = [r["old_conflict"] for r in results]
    pearson_r, pearson_p = stats.pearsonr(old_values, perplexities)
    spearman_r, spearman_p = stats.spearmanr(old_values, perplexities)
    print(
        f"old conflict (untresholded): pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
        f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
    )

    for frac in THRESHOLD_FRACS:
        values = [r[f"thresholded_{frac}"] for r in results]
        pearson_r, pearson_p = stats.pearsonr(values, perplexities)
        spearman_r, spearman_p = stats.spearmanr(values, perplexities)
        print(
            f"thresholded (frac={frac}): pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
            f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
        )


if __name__ == "__main__":
    main()
