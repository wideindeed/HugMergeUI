"""Future-work item logged in VALIDATION.txt: can a 0.5B and a 3B model of
the same family be combined at all? Qwen2.5-0.5B (hidden=896, 24 layers,
intermediate=4864, 14 heads/2 kv heads) and Qwen2.5-3B (hidden=2048, 36
layers, intermediate=11008, 16 heads/2 kv heads) share nothing shape-wise
except vocab_size (151936, same tokenizer) - not even the tied embedding
table lines up, since its second dim is hidden_size. TIES/linear merge is
elementwise and requires identical shapes, so ordinary mergekit merging is
structurally impossible here, not merely untested.

This is candidate (1) from that future-work note: the cheap, naive,
expected-to-be-bad baseline. Every 0.5B tensor is zero-padded independently
on each axis up to the matching 3B tensor's shape (front-padding, no
attempt at semantic alignment), then blended 50/50 with the real 3B tensor.
Layers are matched by proportional index (0.5B layer i -> 3B layer
round(i * 35/23)), so 24 of the 3B's 36 layers receive an injection and 12
are left as pure 3B. This is not expected to work well - the point is to
falsify the naive approach fast and honestly before spending real effort on
candidate (2) (function-preserving growth) or (3) (logit-fusion
distillation).
"""

import json
import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from eval_texts import EVAL_TEXTS

SMALL_REPO = "Qwen/Qwen2.5-0.5B"
BIG_REPO = "Qwen/Qwen2.5-3B"
SMALL_LAYERS = 24
BIG_LAYERS = 36
BLEND_WEIGHT = 0.5  # weight given to the (padded) small model's tensor

RESULTS_PATH = Path(__file__).resolve().parent / "phase_e_cross_size_fusion_results.json"


def _weight_map(repo_id: str) -> dict[str, str]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        with safe_open(path, framework="pt") as f:
            return {name: path for name in f.keys()}
    except Exception:
        pass
    index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_paths = {fn: hf_hub_download(repo_id=repo_id, filename=fn) for fn in set(weight_map.values())}
    return {name: shard_paths[fn] for name, fn in weight_map.items()}


def _load_state_dict(repo_id: str) -> dict[str, torch.Tensor]:
    files = _weight_map(repo_id)
    by_file: dict[str, list[str]] = {}
    for name, path in files.items():
        by_file.setdefault(path, []).append(name)
    state: dict[str, torch.Tensor] = {}
    for path, names in by_file.items():
        with safe_open(path, framework="pt") as f:
            for name in names:
                state[name] = f.get_tensor(name)
    return state


def _remap_layer_index(name: str, mapping: dict[int, int]) -> str | None:
    """Renames `model.layers.<i>.*` to `model.layers.<mapping[i]>.*`. Returns
    None if the tensor belongs to a small-model layer with no target (can't
    happen given how `mapping` is built, but kept explicit)."""
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            small_idx = int(parts[idx + 1])
            if small_idx not in mapping:
                return None
            parts[idx + 1] = str(mapping[small_idx])
            return ".".join(parts)
    return name  # non-layer tensor (embeddings, final norm, ...), name unchanged


def _pad_to(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if tensor.shape == target_shape:
        return tensor
    padded = torch.zeros(target_shape, dtype=tensor.dtype)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
    src_slices = tuple(slice(0, sl.stop) for sl in slices)
    padded[slices] = tensor[src_slices]
    return padded


def build_fused_state_dict() -> dict[str, torch.Tensor]:
    print("[fuse] downloading + loading Qwen2.5-0.5B and Qwen2.5-3B state dicts...", flush=True)
    small = _load_state_dict(SMALL_REPO)
    big = _load_state_dict(BIG_REPO)

    layer_mapping = {i: round(i * (BIG_LAYERS - 1) / (SMALL_LAYERS - 1)) for i in range(SMALL_LAYERS)}
    print(f"[fuse] layer mapping (0.5B idx -> 3B idx): {layer_mapping}", flush=True)

    fused = {name: tensor.clone() for name, tensor in big.items()}

    touched = 0
    skipped_no_counterpart = 0
    for small_name, small_tensor in small.items():
        big_name = _remap_layer_index(small_name, layer_mapping)
        if big_name is None or big_name not in fused:
            skipped_no_counterpart += 1
            continue
        padded = _pad_to(small_tensor.float(), fused[big_name].shape)
        fused[big_name] = (
            (1 - BLEND_WEIGHT) * fused[big_name].float() + BLEND_WEIGHT * padded
        ).to(fused[big_name].dtype)
        touched += 1

    print(f"[fuse] blended {touched} tensors, skipped {skipped_no_counterpart} with no 3B counterpart", flush=True)
    return fused


def mean_perplexity_from_model(model, tokenizer, device="cuda") -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for text in EVAL_TEXTS:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = model(input_ids, labels=input_ids)
            losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses))


def mean_perplexity_solo(repo_id: str, device="cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16).to(device)
    ppl = mean_perplexity_from_model(model, tokenizer, device)
    del model
    torch.cuda.empty_cache()
    return ppl


def main() -> None:
    results: dict = {}

    print("[solo] Qwen2.5-0.5B...", flush=True)
    results["solo_0.5b_ppl"] = mean_perplexity_solo(SMALL_REPO)
    print(f"[solo] Qwen2.5-0.5B: {results['solo_0.5b_ppl']}", flush=True)

    print("[solo] Qwen2.5-3B...", flush=True)
    results["solo_3b_ppl"] = mean_perplexity_solo(BIG_REPO)
    print(f"[solo] Qwen2.5-3B: {results['solo_3b_ppl']}", flush=True)

    fused_state = build_fused_state_dict()

    print("[fused] instantiating Qwen2.5-3B-shaped model with fused weights...", flush=True)
    config = AutoConfig.from_pretrained(BIG_REPO)
    if config.tie_word_embeddings:
        # tied checkpoints never store lm_head.weight on disk - it's the
        # same tensor as the embedding table, restored via tie_weights()
        # after a normal from_pretrained load. Since we're loading a raw
        # state dict instead, the tied key has to be filled in by hand.
        fused_state.setdefault("lm_head.weight", fused_state["model.embed_tokens.weight"])
    fused_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    fused_model.load_state_dict({k: v.to(torch.float16) for k, v in fused_state.items()}, strict=True)
    fused_model = fused_model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(BIG_REPO)
    print("[fused] measuring perplexity...", flush=True)
    results["fused_ppl"] = mean_perplexity_from_model(fused_model, tokenizer)
    print(f"[fused] fused_ppl: {results['fused_ppl']}", flush=True)

    del fused_model
    torch.cuda.empty_cache()

    results["degradation_vs_3b"] = results["fused_ppl"] - results["solo_3b_ppl"]
    results["degradation_vs_3b_pct"] = 100 * results["degradation_vs_3b"] / results["solo_3b_ppl"]

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print("\n=== RESULT ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
