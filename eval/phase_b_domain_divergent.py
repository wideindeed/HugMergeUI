"""Phase 6, Phase B: does the same-domain-only test set explain Phase 5's
rank-order failure? All 20 non-trivial pairs in Phase 5 were different SFT
flavors of similar generic instruction data. This re-runs the same
merge -> perplexity -> conflict-score pipeline (reusing big_validation.run_pair)
on genuinely domain-divergent pairs instead: math-SFT, code-SFT, and
chat-SFT of the same base, for both existing model families.

Checkpoints to phase_b_results.json after every pair, same as big_validation.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from big_validation import run_pair  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "phase_b_results.json"

DOMAIN_MODELS = {
    "qwen": {
        "base": "Qwen/Qwen2.5-0.5B",
        "math": "pngwn/qwen2.5-0.5b-gsm8k-sft",
        "code": "qgallouedec/Qwen2.5-0.5B-codeforces-SFT",
        "chat": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "smollm2": {
        "base": "HuggingFaceTB/SmolLM2-360M",
        "math": "qiuyu8290/SmolLM2-360M-sft-math-gsm8k",
        "code": "CodeAtCMU/SmolLM2-360M_full_sft_code_data_120K",
        "chat": "HuggingFaceTB/SmolLM2-360M-Instruct",
    },
}

DOMAIN_PAIRS = [("math", "code"), ("math", "chat"), ("code", "chat")]


def main() -> None:
    results: list[dict] = []

    for family_name, spec in DOMAIN_MODELS.items():
        base_repo = spec["base"]
        for name_a, name_b in DOMAIN_PAIRS:
            result = run_pair(
                family_name, base_repo, name_a, spec[name_a], name_b, spec[name_b]
            )
            result["pair_type"] = "domain-divergent"
            results.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== PHASE B RESULTS (domain-divergent pairs) ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
