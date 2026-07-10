"""Phase 7 full validation: CKA-based activation divergence, scored against
all 28 pairs already validated in Phase 5/6 (big_validation_results.json +
phase_b_results.json). No re-merging or re-measuring perplexity - those
numbers are already on disk. Only re-downloads/re-scores each unique model
repo's hidden states once (cached across pairs that reuse a fine-tune),
computes per-layer linear CKA against the shared base, and correlates a
few candidate combinations against the existing perplexity column.

See phase7_probe.py for the single-pair sanity check this scales up, and
VALIDATION.txt for why the raw-weight-diff metric family was abandoned.
"""

import json
from pathlib import Path

import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_texts import EVAL_TEXTS

FAMILY_BASE = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "smollm2": "HuggingFaceTB/SmolLM2-360M",
}

RESULTS_PATH = Path(__file__).resolve().parent / "phase7_full_validation_results.json"


def collect_hidden_states(repo_id: str, tokenizer, device: str) -> list[torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float32).to(device)
    model.eval()

    per_layer: list[list[torch.Tensor]] = None
    with torch.no_grad():
        for text in EVAL_TEXTS:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = model(input_ids, output_hidden_states=True)
            hidden_states = [h.squeeze(0).float().cpu() for h in out.hidden_states]
            if per_layer is None:
                per_layer = [[] for _ in hidden_states]
            for i, h in enumerate(hidden_states):
                per_layer[i].append(h)

    del model
    torch.cuda.empty_cache()
    return [torch.cat(layer_chunks, dim=0) for layer_chunks in per_layer]


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    hsic = (y.T @ x).norm() ** 2
    normalization = (x.T @ x).norm() * (y.T @ y).norm()
    if normalization == 0:
        return 1.0
    return (hsic / normalization).item()


def divergence_from_base(base_hidden: list[torch.Tensor], model_hidden: list[torch.Tensor]) -> list[float]:
    return [1.0 - linear_cka(base_hidden[i], model_hidden[i]) for i in range(len(base_hidden))]


def main() -> None:
    pairs = json.loads((Path(__file__).resolve().parent / "big_validation_results.json").read_text())
    pairs += json.loads((Path(__file__).resolve().parent / "phase_b_results.json").read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for family_name, base_repo in FAMILY_BASE.items():
        family_pairs = [r for r in pairs if r["family"] == family_name]
        print(f"\n=== family: {family_name} ({len(family_pairs)} pairs) ===", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(base_repo)
        base_hidden = collect_hidden_states(base_repo, tokenizer, device)

        hidden_cache: dict[str, list[torch.Tensor]] = {base_repo: base_hidden}
        divergence_cache: dict[str, list[float]] = {base_repo: [0.0] * len(base_hidden)}

        def get_divergence(repo_id: str) -> list[float]:
            if repo_id not in divergence_cache:
                print(f"  [{repo_id}] computing hidden states...", flush=True)
                hidden_cache[repo_id] = collect_hidden_states(repo_id, tokenizer, device)
                divergence_cache[repo_id] = divergence_from_base(base_hidden, hidden_cache[repo_id])
            return divergence_cache[repo_id]

        for r in family_pairs:
            div_a = get_divergence(r["model_a"])
            div_b = get_divergence(r["model_b"])

            mean_div_a = sum(div_a) / len(div_a)
            mean_div_b = sum(div_b) / len(div_b)
            mean_div_a_excl = sum(div_a[:-1]) / len(div_a[:-1])
            mean_div_b_excl = sum(div_b[:-1]) / len(div_b[:-1])

            entry = {
                "family": r["family"],
                "pair": r["pair"],
                "perplexity": r["perplexity"],
                "old_conflict": r["conflict"],
                "cka_div_a": mean_div_a,
                "cka_div_b": mean_div_b,
                "cka_combined": mean_div_a + mean_div_b,
                "cka_product": mean_div_a * mean_div_b,
                "cka_combined_excl_last": mean_div_a_excl + mean_div_b_excl,
            }
            results.append(entry)
            print(f"  [{r['pair']}] {entry}", flush=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    perplexities = [r["perplexity"] for r in results]
    print(f"\n=== CORRELATIONS (n={len(results)}) ===")
    for metric in ["old_conflict", "cka_div_a", "cka_div_b", "cka_combined", "cka_product", "cka_combined_excl_last"]:
        values = [r[metric] for r in results]
        pearson_r, pearson_p = stats.pearsonr(values, perplexities)
        spearman_r, spearman_p = stats.spearmanr(values, perplexities)
        print(
            f"{metric}: pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
            f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
        )


if __name__ == "__main__":
    main()
