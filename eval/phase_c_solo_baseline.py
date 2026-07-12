"""Follow-up to Round One/Two's 0.5B/360M null result (VALIDATION.txt).

Two things were left dangling from that round: (1) the n=22 correlation
pooled two architecturally-unrelated families into one Pearson/Spearman
figure despite an explicit caveat that perplexity scale differs by family
and tokenizer, and per-family correlations were recommended but never run;
(2) several of the smollm2 fine-tunes are ungated hobbyist checkpoints of
unknown individual quality (Michaelj1/*, prithivMLmods/SmolLM2-CoT-360M,
Masorian06/SmolLM2-360M-Rickified) - if one of them is just a weak
standalone model, any pair including it looks "damaged" regardless of
real merge conflict, contaminating the correlation the same way a bad
component model would confound any A/B test.

This measures each individual fine-tune's own solo perplexity (no merge)
on the same 5 held-out passages used throughout, then reruns the
correlation using degradation = merged_ppl - max(solo_ppl_a, solo_ppl_b)
against drift_magnitude/conflict, instead of raw merged perplexity - the
same "compare against solo baseline" fix that worked for the domain-eval
follow-up at 7B (phase_7b_domain.py). Reuses big_validation_results.json,
no re-merging needed.
"""

import json
import math
import sys
from pathlib import Path

import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_texts import EVAL_TEXTS  # noqa: E402
from big_validation import FAMILIES  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "big_validation_results.json"
SOLO_PATH = Path(__file__).resolve().parent / "phase_c_solo_results.json"


def mean_perplexity(model_path: str, tokenizer_fallback: str, device: str = "cuda") -> float:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"  (no tokenizer shipped, falling back to {tokenizer_fallback})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_fallback)
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


def main() -> None:
    solo: dict[str, float] = {}
    if SOLO_PATH.exists():
        solo = json.loads(SOLO_PATH.read_text())
        print(f"resuming: {len(solo)} solo repo(s) already measured", flush=True)

    for family_name, spec in FAMILIES.items():
        base_repo = spec["base"]
        repos = [base_repo] + list(spec["finetunes"].values())
        for repo in repos:
            if repo in solo:
                continue
            print(f"[solo] {repo}...", flush=True)
            solo[repo] = mean_perplexity(repo, tokenizer_fallback=base_repo)
            SOLO_PATH.write_text(json.dumps(solo, indent=2))
            print(f"[solo] {repo}: {solo[repo]}", flush=True)

    results = json.loads(RESULTS_PATH.read_text())

    print("\n=== PER-FAMILY CORRELATIONS: degradation vs. best solo baseline ===")
    for fam in FAMILIES:
        rows = [r for r in results if r["family"] == fam and r["pair"] != "base+base2"]
        degradation = []
        for r in rows:
            solo_a = solo[r["model_a"]]
            solo_b = solo[r["model_b"]]
            degradation.append(r["perplexity"] - max(solo_a, solo_b))

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
