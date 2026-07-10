"""Re-run the phase_scale_1_5b correlation analysis on the 'healthy' subset,
excluding domain-divergent specialist pairs (anything involving the coder
or math fine-tunes) that produced catastrophic perplexity blowups (hundreds
to millions). Those 5 outlier points dominate the full n=14 correlation;
this checks whether drift_magnitude's rho=0.96 (p=6e-08) survives on the
remaining 9 pairs, which cluster much more tightly (perplexity ~7-17).

No new merges/scoring - reuses phase_scale_1_5b_results.json.
"""

import json
from pathlib import Path

from scipy import stats

RESULTS_PATH = Path(__file__).resolve().parent / "phase_scale_1_5b_results.json"
DOMAIN_DIVERGENT_NAMES = {"coder", "math", "code"}


def is_domain_divergent(pair: str) -> bool:
    name_a, name_b = pair.split("+")
    return name_a in DOMAIN_DIVERGENT_NAMES or name_b in DOMAIN_DIVERGENT_NAMES


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    healthy = [r for r in results if not is_domain_divergent(r["pair"])]
    excluded = [r for r in results if is_domain_divergent(r["pair"])]

    print(f"excluded {len(excluded)} domain-divergent pair(s):")
    for r in excluded:
        print(f"  {r['family']} {r['pair']}: perplexity={r['perplexity']:.2f}")

    print(f"\nhealthy subset (n={len(healthy)}):")
    for r in healthy:
        print(f"  {r['family']} {r['pair']}: conflict={r['conflict']:.4f} "
              f"drift={r['drift_magnitude']:.4f} perplexity={r['perplexity']:.4f}")

    perplexities = [r["perplexity"] for r in healthy]
    print(f"\n=== CORRELATIONS on healthy subset (n={len(healthy)}) ===")
    for metric in ["conflict", "conflict_weighted", "drift_magnitude"]:
        values = [r[metric] for r in healthy]
        pearson_r, pearson_p = stats.pearsonr(values, perplexities)
        spearman_r, spearman_p = stats.spearmanr(values, perplexities)
        print(
            f"{metric}: pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
            f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
        )

    # Also drop the two trivial self-anchor pairs (conflict=0 by construction,
    # base model vs itself) since they're not real merges and could be doing
    # a lot of the rank-order lifting at the low end.
    non_trivial = [r for r in healthy if r["pair"] not in ("base+base2",)]
    perplexities_nt = [r["perplexity"] for r in non_trivial]
    print(f"\n=== CORRELATIONS on healthy, non-trivial subset (n={len(non_trivial)}) ===")
    for metric in ["conflict", "conflict_weighted", "drift_magnitude"]:
        values = [r[metric] for r in non_trivial]
        pearson_r, pearson_p = stats.pearsonr(values, perplexities_nt)
        spearman_r, spearman_p = stats.spearmanr(values, perplexities_nt)
        print(
            f"{metric}: pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
            f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
        )


if __name__ == "__main__":
    main()
