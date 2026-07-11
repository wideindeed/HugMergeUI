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


class _TensorStreamer:
    """Fetches tensors by name from a repo's shard files one at a time,
    reusing an open safetensors handle per shard while tensors from that
    shard are still needed - and closing it the moment they aren't.

    safetensors mmaps each shard file; a handle left open keeps its touched
    pages resident (counted in RSS) even after get_tensor() has copied the
    data out. Closing eagerly (not just at the end of the whole run) is what
    actually bounds peak memory - keeping every handle open until the end,
    even without copying tensors upfront, converges back to ~holding every
    shard fully resident."""

    def __init__(self, tensor_files: dict[str, str], names: set[str]):
        self._tensor_files = tensor_files
        self._handles: dict[str, object] = {}
        self._remaining: dict[str, int] = defaultdict(int)
        for name in names:
            self._remaining[tensor_files[name]] += 1

    def get(self, name: str) -> torch.Tensor:
        path = self._tensor_files[name]
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt")
            self._handles[path] = handle

        tensor = handle.get_tensor(name)

        self._remaining[path] -= 1
        if self._remaining[path] == 0:
            handle.__exit__(None, None, None)
            del self._handles[path]

        return tensor

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()


def _score_common_streaming(
    base_files: dict[str, str],
    a_files: dict[str, str],
    b_files: dict[str, str],
    common: set[str],
    *,
    density: float = 0.5,
) -> Iterator[dict]:
    """Same computation and result shape as score_tensors, but pulls one
    tensor triple at a time from disk instead of materializing full
    base/model_a/model_b dicts first, and closes each shard's mmap as soon
    as its last needed tensor has been read. Peak memory is bounded by
    however many shards are concurrently open across the 3 repos, not the
    full model size, so scale is limited by disk space rather than RAM.

    Iteration order matters a lot here: names must be grouped by which
    physical shard file they live in (using base_files as the reference,
    since all 3 repos share the same architecture and shard split), so each
    shard's tensors are read together and its handle can close before the
    next shard opens. Neither plain sorted(names) (scatters "layers.10"
    before "layers.2" alphabetically) nor index.json's weight_map order
    (verified empirically to NOT be shard-contiguous - HF doesn't guarantee
    that) gets this right; grouping by the actual resolved file path does."""
    shard_rank = {path: i for i, path in enumerate(sorted(set(base_files.values())))}
    names = sorted(common, key=lambda n: (shard_rank[base_files[n]], n))
    total = len(names)
    buckets: dict[int | None, list[tuple]] = defaultdict(list)

    base_stream = _TensorStreamer(base_files, common)
    a_stream = _TensorStreamer(a_files, common)
    b_stream = _TensorStreamer(b_files, common)

    try:
        for i, name in enumerate(names):
            base_t, a_t, b_t = base_stream.get(name), a_stream.get(name), b_stream.get(name)
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
    finally:
        base_stream.close()
        a_stream.close()
        b_stream.close()

    layers = [
        {"layer": layer_index, **_summarize(buckets[layer_index])}
        for layer_index in sorted(k for k in buckets if k is not None)
    ]
    other = _summarize(buckets[None]) if None in buckets else None

    yield {"stage": "scored", "result": {"layers": layers, "other": other}}


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

    for event in _score_common_streaming(base_files, a_files, b_files, common, density=density):
        if event["stage"] == "scored":
            return event["result"]
    raise AssertionError("streaming scorer did not yield a final result")


def score_model_pair_progress(
    base_repo_id: str, model_a_repo_id: str, model_b_repo_id: str, *, density: float = 0.5
) -> Iterator[dict]:
    """Generator variant of score_model_pair that yields progress events for
    each real stage (resolving repo file lists, streaming+scoring tensors),
    so a streaming HTTP response can show genuine progress instead of a
    simulated loading bar. Tensors are fetched from disk on demand during
    scoring rather than loaded upfront, so there's no separate "load" stage
    - see _score_common_streaming."""
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

    yield from _score_common_streaming(base_files, a_files, b_files, common, density=density)
