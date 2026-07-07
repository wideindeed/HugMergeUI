"""Turns a validated raw mergekit config into a merge-method-agnostic
layer list, so the frontend and conflict-score engine never need to know
mergekit's YAML dialect.

- passthrough: each layer has exactly one source, no blending.
- slerp: each layer has two sources plus a resolved per-category blend_t.
- linear / ties: each layer has one entry per model with that model's own
  resolved per-category parameters (weight, density, ...).
"""

from typing import Any

from .resolver import resolve_parameter
from .schema import TENSOR_CATEGORIES, validate_raw_config


def normalize(raw: dict, *, num_layers: int | None = None) -> dict:
    validate_raw_config(raw)
    method = raw["merge_method"]

    if method == "passthrough":
        layers, models = _normalize_passthrough(raw["slices"])
    elif method == "slerp":
        layers, models = _normalize_slerp(raw["slices"], raw.get("parameters", {}))
    else:  # linear, ties
        if num_layers is None:
            raise ValueError(
                f"merge_method {method!r} has no 'slices' to infer layer count from; "
                "num_layers must be supplied (e.g. from the base model's HF config.json)"
            )
        layers, models = _normalize_whole_model(raw["models"], num_layers)

    return {
        "merge_method": method,
        "base_model": raw.get("base_model"),
        "dtype": raw.get("dtype"),
        "merge_parameters": raw.get("parameters") if method in ("linear", "ties") else None,
        "models": models,
        "layers": layers,
    }


def _normalize_passthrough(slices: list[dict]) -> tuple[list[dict], list[str]]:
    layers = []
    models: list[str] = []
    index = 0
    for slice_spec in slices:
        sources = slice_spec["sources"]
        if len(sources) != 1:
            raise ValueError("passthrough slices must have exactly one source each")
        model = sources[0]["model"]
        if model not in models:
            models.append(model)
        start, end = sources[0]["layer_range"]
        for source_layer_index in range(start, end):
            layers.append({
                "index": index,
                "sources": [{"model": model, "source_layer_index": source_layer_index}],
            })
            index += 1
    return layers, models


def _normalize_slerp(slices: list[dict], parameters: dict) -> tuple[list[dict], list[str]]:
    layers = []
    models: list[str] = []
    index = 0
    t_spec = parameters.get("t", 0.5)

    for slice_spec in slices:
        sources = slice_spec["sources"]
        if len(sources) != 2:
            raise ValueError("slerp slices must have exactly two sources")
        ranges = [s["layer_range"] for s in sources]
        length = ranges[0][1] - ranges[0][0]
        if any((end - start) != length for start, end in ranges):
            raise ValueError(f"mismatched layer_range lengths in slerp slice: {ranges}")

        for m in (s["model"] for s in sources):
            if m not in models:
                models.append(m)

        for depth in range(length):
            blend_t = _resolve_categories(t_spec, layer_index=depth, num_layers=length)
            layers.append({
                "index": index,
                "sources": [
                    {"model": s["model"], "source_layer_index": s["layer_range"][0] + depth}
                    for s in sources
                ],
                "blend_t": blend_t,
            })
            index += 1
    return layers, models


def _normalize_whole_model(models_spec: list[dict], num_layers: int) -> tuple[list[dict], list[str]]:
    layers = []
    models = [m["model"] for m in models_spec]

    for layer_index in range(num_layers):
        sources = [
            {
                "model": m["model"],
                "parameters": {
                    category: {
                        name: resolve_parameter(
                            spec, tensor_name=f"layers.{layer_index}.{category}",
                            layer_index=layer_index, num_layers=num_layers,
                        )
                        for name, spec in m.get("parameters", {}).items()
                    }
                    for category in TENSOR_CATEGORIES
                },
            }
            for m in models_spec
        ]
        layers.append({"index": layer_index, "sources": sources})
    return layers, models


def _resolve_categories(spec: Any, *, layer_index: int, num_layers: int) -> dict:
    return {
        category: resolve_parameter(
            spec, tensor_name=f"layers.{layer_index}.{category}",
            layer_index=layer_index, num_layers=num_layers,
        )
        for category in TENSOR_CATEGORIES
    }
