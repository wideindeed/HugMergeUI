"""Aggregates per-tensor sign-conflict and redundancy scores into a
per-layer view, so the frontend heatmap has one number per layer instead
of one per tensor.

Tensors are grouped by the `.layers.N.` index in their name (the
convention used by Llama/Qwen-family architectures); everything else
(embeddings, final norm, lm_head, ...) is aggregated into a separate
"other" bucket rather than forced into a fake layer. Per-layer numbers
are averages weighted by tensor element count, so a large down_proj
doesn't get diluted by a tiny layernorm bias in the same layer.
"""

import re
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, HFValidationError
from safetensors import safe_open

from .redundancy import redundant_magnitude_fraction
from .sign_conflict import sign_conflict_rate

_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


class ModelWeightsFetchError(Exception):
    pass


def _download_safetensors(repo_id: str) -> str:
    try:
        return hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    except (HfHubHTTPError, HFValidationError) as e:
        raise ModelWeightsFetchError(f"could not fetch model.safetensors for {repo_id!r}: {e}") from e


def _extract_layer_index(tensor_name: str) -> int | None:
    match = _LAYER_PATTERN.search(tensor_name)
    return int(match.group(1)) if match else None


def _weighted_average(entries: list[tuple], field: int) -> float:
    total_weight = sum(e[0] for e in entries)
    if total_weight == 0:
        return 0.0
    return sum(e[0] * e[field] for e in entries) / total_weight


def _summarize(entries: list[tuple]) -> dict:
    return {
        "tensor_count": len(entries),
        "conflict": _weighted_average(entries, 1),
        "redundancy_a": _weighted_average(entries, 2),
        "redundancy_b": _weighted_average(entries, 3),
    }


def score_tensors(
    base: dict[str, torch.Tensor],
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    *,
    density: float = 0.5,
) -> dict:
    common = set(base) & set(model_a) & set(model_b)

    buckets: dict[int | None, list[tuple]] = defaultdict(list)

    for name in common:
        base_t, a_t, b_t = base[name], model_a[name], model_b[name]
        if base_t.shape != a_t.shape or base_t.shape != b_t.shape:
            continue

        diff_a = (a_t - base_t).float()
        diff_b = (b_t - base_t).float()

        entry = (
            diff_a.numel(),
            sign_conflict_rate(diff_a, diff_b),
            redundant_magnitude_fraction(diff_a, density),
            redundant_magnitude_fraction(diff_b, density),
        )
        buckets[_extract_layer_index(name)].append(entry)

    layers = [
        {"layer": layer_index, **_summarize(buckets[layer_index])}
        for layer_index in sorted(k for k in buckets if k is not None)
    ]
    other = _summarize(buckets[None]) if None in buckets else None

    return {"layers": layers, "other": other}


def score_model_pair(
    base_repo_id: str, model_a_repo_id: str, model_b_repo_id: str, *, density: float = 0.5
) -> dict:
    base_path = _download_safetensors(base_repo_id)
    a_path = _download_safetensors(model_a_repo_id)
    b_path = _download_safetensors(model_b_repo_id)

    with safe_open(base_path, framework="pt") as f_base, \
         safe_open(a_path, framework="pt") as f_a, \
         safe_open(b_path, framework="pt") as f_b:
        common = set(f_base.keys()) & set(f_a.keys()) & set(f_b.keys())
        base = {k: f_base.get_tensor(k) for k in common}
        model_a = {k: f_a.get_tensor(k) for k in common}
        model_b = {k: f_b.get_tensor(k) for k in common}

    return score_tensors(base, model_a, model_b, density=density)
