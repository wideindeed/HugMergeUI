"""
phase_pooled_meta.py - pooled meta-analysis across every committed results file.

Zero GPU, zero downloads, runs anywhere including the GTX 1650 laptop.
Reads the JSONs already in eval/ and asks the one question fifteen rounds
of individual runs could not: is parameter count actually the variable, or
is it drift range?

Method notes:
  - perplexity is normalized to each family's own self-merge anchor
    (damage_ratio = merged_ppl / anchor_ppl). Raw perplexity is NOT
    comparable across families: a 7B model has a lower baseline than a
    360M one regardless of merge quality. Skipping this step produces a
    fake size effect.
  - both axes are log-transformed. Damage ratios in this corpus span
    ~1x to ~10^5x; untransformed Pearson is dominated by single points.
  - self-merge anchors (drift == 0) are excluded from correlations. They
    are trivially perfect and inflate every metric that includes them.

Run: python eval/phase_pooled_meta.py
"""
import json
import os
import numpy as np
from scipy import stats

EVAL = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "big_validation_results.json",        # 0.5B/360M, expanded to 8 finetunes/family
    "phase_b_results.json",               # domain-divergent small scale
    "phase_scale_1_5b_results.json",      # 1.5B/1.7B/1B, three families
    "phase_scale_3b_results.json",        # 3B, qwen + llama
    "phase_arch_stablelm_results.json",   # StableLM-2 1.6B
    "phase_trend_llama_results.json",     # llama3.2_3b heavy-domain trend
    "phase_7b_mistral_results.json",      # 7B Mistral
    "phase_cross_domain_results.json",         # qwen1.5b coder_instruct x math_instruct (Round 13)
    "phase_cross_domain_llama_v2_results.json",  # llama3.2_3b megamath x episteme_coder (Round 14 cont.)
]

# approximate params in billions, per family key
SIZE = {
    "qwen": 0.5, "smollm2": 0.36, "llama3.2_1b": 1.0, "qwen1.5b": 1.5,
    "smollm2_1.7b": 1.7, "stablelm2_1.6b": 1.6, "qwen3b": 3.0,
    "llama3.2_3b": 3.0, "mistral7b": 7.0,
}

# the 7B run has no self-merge anchor; use its cleanest observed pair as a
# baseline proxy. Flagged explicitly because it is an assumption, not data.
ANCHOR_FALLBACK = {"mistral7b": 4.417}


def load():
    rows, seen = [], set()
    for fname in FILES:
        path = os.path.join(EVAL, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        payload = json.load(open(path))
        for rec in (payload if isinstance(payload, list) else [payload]):
            if "drift_magnitude" not in rec or "perplexity" not in rec:
                continue
            if rec.get("family") not in SIZE:
                continue
            key = (rec["family"], rec.get("pair"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
    return rows


def main():
    rows = load()
    anchors = {r["family"]: r["perplexity"] for r in rows if r["drift_magnitude"] == 0}
    for k, v in ANCHOR_FALLBACK.items():
        anchors.setdefault(k, v)

    data = [r for r in rows if r["drift_magnitude"] > 0 and r["family"] in anchors]
    drift = np.array([r["drift_magnitude"] for r in data])
    size = np.array([SIZE[r["family"]] for r in data], float)
    fam = np.array([r["family"] for r in data])
    damage = np.array([r["perplexity"] / anchors[r["family"]] for r in data])

    x, y = np.log(drift), np.log(damage)
    print(f"\npooled non-trivial merges: n={len(data)} across {len(set(fam))} families, "
          f"{size.min()}B - {size.max()}B\n")

    slope, icept, r, p, _ = stats.linregress(x, y)
    rho, rp = stats.spearmanr(drift, damage)
    print("log(damage ratio) ~ log(drift_magnitude), all scales pooled")
    print(f"  pearson r={r:.4f} (r^2={r**2:.3f}) p={p:.3g}   spearman rho={rho:.4f} p={rp:.3g}")

    resid = y - (slope * x + icept)
    pr, pp = stats.pearsonr(np.log(size), resid)
    sr, sp = stats.spearmanr(size, resid)
    print("\nresidual vs parameter count (does size add anything beyond drift?)")
    print(f"  pearson r={pr:+.4f} p={pp:.3g}   spearman rho={sr:+.4f} p={sp:.3g}")

    print("\nmean residual by family (positive = worse than the curve predicts)")
    for f in sorted(set(fam)):
        m = fam == f
        print(f"  {f:16s} n={m.sum():3d} {size[m][0]:4.2f}B  "
              f"resid={resid[m].mean():+.3f} +/- {resid[m].std():.3f}")

    print("\nleave-one-family-out (is any single family carrying the result?)")
    for f in sorted(set(fam)):
        m = fam != f
        rr, _ = stats.pearsonr(x[m], y[m])
        print(f"  drop {f:16s} -> r={rr:.4f}  n={m.sum()}")

    print("\nhealthy-vs-catastrophic split (catastrophic = log(damage ratio) >= 1, "
          "i.e. damage >= ~2.72x)")
    healthy = y < 1
    cat_fam = fam[~healthy]
    print(f"  catastrophic tail: n={(~healthy).sum()}  families: "
          f"{', '.join(f'{f}x{(cat_fam == f).sum()}' for f in sorted(set(cat_fam)))}")
    hr, hp = stats.pearsonr(x[healthy], y[healthy])
    hrho, hrp = stats.spearmanr(drift[healthy], damage[healthy])
    print(f"  healthy-merge-only subset: n={healthy.sum()}")
    print(f"    pearson r={hr:.4f} (r^2={hr**2:.3f}) p={hp:.3g}   "
          f"spearman rho={hrho:.4f} p={hrp:.3g}")

    print("\nobserved drift range per family (detectability check)")
    for f in sorted(set(fam), key=lambda k: SIZE[k]):
        m = fam == f
        print(f"  {f:16s} {size[m][0]:4.2f}B  drift {drift[m].min():.4f}-{drift[m].max():.4f}"
              f"   damage {damage[m].min():.2f}x-{damage[m].max():.2f}x")


if __name__ == "__main__":
    main()
