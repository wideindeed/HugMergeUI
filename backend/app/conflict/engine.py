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

import json
import re
from collections import defaultdict
from collections.abc import Iterator

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, HFValidationError
from safetensors import safe_open

from .drift import drift_magnitude
from .redundancy import redundant_magnitude_fraction
from .sign_conflict import magnitude_weighted_conflict_rate, sign_conflict_rate

_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


class ModelWeightsFetchError(Exception):
    pass


def _weight_map_from_index(index_json_path: str) -> dict[str, str]:
    with open(index_json_path) as f:
        index = json.load(f)
    return index["weight_map"]


def _resolve_tensor_files(repo_id: str) -> dict[str, str]:
    """Maps every tensor name in a repo to its local (downloaded) shard path.

    Most models under ~5GB ship as a single `model.safetensors`; anything
    bigger - i.e. most real mergekit targets - is split into numbered shards
    with a `model.safetensors.index.json` weight map. Both are handled so the
    engine isn't limited to toy-sized single-file models.
    """
    try:
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        with safe_open(path, framework="pt") as f:
            return {name: path for name in f.keys()}
    except EntryNotFoundError:
        pass
    except (HfHubHTTPError, HFValidationError) as e:
        raise ModelWeightsFetchError(f"could not fetch weights for {repo_id!r}: {e}") from e

    try:
        index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    except (HfHubHTTPError, HFValidationError, EntryNotFoundError) as e:
        raise ModelWeightsFetchError(f"could not fetch weights for {repo_id!r}: {e}") from e

    weight_map = _weight_map_from_index(index_path)
    shard_paths: dict[str, str] = {}
    try:
        for shard_filename in set(weight_map.values()):
            shard_paths[shard_filename] = hf_hub_download(repo_id=repo_id, filename=shard_filename)
    except (HfHubHTTPError, HFValidationError, EntryNotFoundError) as e:
        raise ModelWeightsFetchError(f"could not fetch shard for {repo_id!r}: {e}") from e

    return {name: shard_paths[shard_filename] for name, shard_filename in weight_map.items()}


def _load_tensors(tensor_files: dict[str, str], names: set[str]) -> dict[str, torch.Tensor]:
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in names:
        names_by_file[tensor_files[name]].append(name)

    tensors: dict[str, torch.Tensor] = {}
    for path, names_in_file in names_by_file.items():
        with safe_open(path, framework="pt") as f:
            for name in names_in_file:
                tensors[name] = f.get_tensor(name)
    return tensors


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
        "conflict_weighted": _weighted_average(entries, 2),
        "redundancy_a": _weighted_average(entries, 3),
        "redundancy_b": _weighted_average(entries, 4),
        "drift_magnitude": _weighted_average(entries, 5),
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
            magnitude_weighted_conflict_rate(diff_a, diff_b),
            redundant_magnitude_fraction(diff_a, density),
            redundant_magnitude_fraction(diff_b, density),
            drift_magnitude(diff_a, diff_b, base_t.float()),
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
    base_files = _resolve_tensor_files(base_repo_id)
    a_files = _resolve_tensor_files(model_a_repo_id)
    b_files = _resolve_tensor_files(model_b_repo_id)

    common = set(base_files) & set(a_files) & set(b_files)
    if not common:
        raise ValueError(
            f"no tensors in common between {base_repo_id!r}, {model_a_repo_id!r}, "
            f"{model_b_repo_id!r} - check these are compatible architectures"
        )

    base = _load_tensors(base_files, common)
    model_a = _load_tensors(a_files, common)
    model_b = _load_tensors(b_files, common)

    return score_tensors(base, model_a, model_b, density=density)


def score_tensors_progress(
    base: dict[str, torch.Tensor],
    model_a: dict[str, torch.Tensor],
    model_b: dict[str, torch.Tensor],
    *,
    density: float = 0.5,
) -> Iterator[dict]:
    """Same computation as score_tensors, but yields a progress event after
    every few tensors so a caller can report real (not simulated) progress
    on the expensive part of the request."""
    common = sorted(set(base) & set(model_a) & set(model_b))
    total = len(common)
    buckets: dict[int | None, list[tuple]] = defaultdict(list)

    for i, name in enumerate(common):
        base_t, a_t, b_t = base[name], model_a[name], model_b[name]
        if base_t.shape != a_t.shape or base_t.shape != b_t.shape:
            continue

        diff_a = (a_t - base_t).float()
        diff_b = (b_t - base_t).float()

        entry = (
            diff_a.numel(),
            sign_conflict_rate(diff_a, diff_b),
            magnitude_weighted_conflict_rate(diff_a, diff_b),
            redundant_magnitude_fraction(diff_a, density),
            redundant_magnitude_fraction(diff_b, density),
            drift_magnitude(diff_a, diff_b, base_t.float()),
        )
        buckets[_extract_layer_index(name)].append(entry)

        if i % 4 == 0 or i == total - 1:
            yield {"stage": "scoring", "tensor_index": i + 1, "tensor_total": total}

    layers = [
        {"layer": layer_index, **_summarize(buckets[layer_index])}
        for layer_index in sorted(k for k in buckets if k is not None)
    ]
    other = _summarize(buckets[None]) if None in buckets else None

    yield {"stage": "scored", "result": {"layers": layers, "other": other}}


def score_model_pair_progress(
    base_repo_id: str, model_a_repo_id: str, model_b_repo_id: str, *, density: float = 0.5
) -> Iterator[dict]:
    """Generator variant of score_model_pair that yields progress events for
    each real stage (resolving repo file lists, downloading/loading tensors,
    scoring), so a streaming HTTP response can show genuine progress instead
    of a simulated loading bar."""
    yield {"stage": "resolve", "repo": base_repo_id}
    base_files = _resolve_tensor_files(base_repo_id)
    yield {"stage": "resolve", "repo": model_a_repo_id}
    a_files = _resolve_tensor_files(model_a_repo_id)
    yield {"stage": "resolve", "repo": model_b_repo_id}
    b_files = _resolve_tensor_files(model_b_repo_id)

    common = set(base_files) & set(a_files) & set(b_files)
    if not common:
        raise ValueError(
            f"no tensors in common between {base_repo_id!r}, {model_a_repo_id!r}, "
            f"{model_b_repo_id!r} - check these are compatible architectures"
        )

    yield {"stage": "load", "repo": base_repo_id}
    base = _load_tensors(base_files, common)
    yield {"stage": "load", "repo": model_a_repo_id}
    model_a = _load_tensors(a_files, common)
    yield {"stage": "load", "repo": model_b_repo_id}
    model_b = _load_tensors(b_files, common)

    yield from score_tensors_progress(base, model_a, model_b, density=density)
