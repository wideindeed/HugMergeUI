"""Phase 7 plumbing sanity check: does activation divergence (linear CKA
between a fine-tune's hidden states and the shared base's hidden states,
per layer, on the eval_texts.py probe set) separate known-good merges from
known-bad ones better than the raw-weight-diff conflict score did?

Not a full validation run - just three pairs, chosen because the OLD
conflict score couldn't tell them apart (all ~0.49-0.50) despite very
different real perplexity outcomes:

  anchor-self (base+base2)          perplexity 9.563  (= base, trivial)
  instruct+michaelj1_wikitext       perplexity 9.867  (near base, best real pair)
  michaelj1+cot                     perplexity 14.985 (worst real pair)

If CKA-based divergence separates these three by a wide margin, the
approach is worth building out into a full Phase 7 metric + validation
pass. If it can't even separate the best and worst pair in this family,
it's not worth pursuing further without a different formulation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_texts import EVAL_TEXTS

BASE = "HuggingFaceTB/SmolLM2-360M"

PAIRS = {
    "anchor-self": (BASE, BASE),
    "instruct+michaelj1_wikitext": (
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Michaelj1/INSTRUCT_smolLM2-360M-finetuned-wikitext2-raw-v1",
    ),
    "michaelj1+cot": (
        "Michaelj1/finetune-smolLM2-360M",
        "prithivMLmods/SmolLM2-CoT-360M",
    ),
}


def collect_hidden_states(repo_id: str, tokenizer, device: str = "cuda") -> list[torch.Tensor]:
    """Returns one (total_tokens, hidden) matrix per layer (including the
    embedding output), concatenated across all probe texts."""
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base_hidden = collect_hidden_states(BASE, tokenizer, device)

    print(f"num layers (incl. embedding output): {len(base_hidden)}")

    for pair_name, (repo_a, repo_b) in PAIRS.items():
        print(f"\n[{pair_name}] scoring...", flush=True)

        hidden_a = base_hidden if repo_a == BASE else collect_hidden_states(repo_a, tokenizer, device)
        hidden_b = base_hidden if repo_b == BASE else collect_hidden_states(repo_b, tokenizer, device)

        per_layer_div_a = [1.0 - linear_cka(base_hidden[i], hidden_a[i]) for i in range(len(base_hidden))]
        per_layer_div_b = [1.0 - linear_cka(base_hidden[i], hidden_b[i]) for i in range(len(base_hidden))]

        mean_div_a = sum(per_layer_div_a) / len(per_layer_div_a)
        mean_div_b = sum(per_layer_div_b) / len(per_layer_div_b)
        combined = mean_div_a + mean_div_b

        print(f"[{pair_name}] mean divergence_a={mean_div_a:.4f} divergence_b={mean_div_b:.4f} combined={combined:.4f}")
        print(f"[{pair_name}] per-layer divergence_a: {[round(v, 4) for v in per_layer_div_a]}")
        print(f"[{pair_name}] per-layer divergence_b: {[round(v, 4) for v in per_layer_div_b]}")


if __name__ == "__main__":
    main()
