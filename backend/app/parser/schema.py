"""Structural validation for raw mergekit YAML configs.

Only the merge methods we have real example configs for are supported so
far: linear, ties (whole-model, via `models:`) and slerp, passthrough
(layer-range based, via `slices:`). Add methods here as fixtures for them
are added, rather than guessing at their shape upfront.
"""

KNOWN_MERGE_METHODS = {"linear", "slerp", "ties", "passthrough"}
METHODS_REQUIRING_BASE_MODEL = {"slerp", "ties"}
METHODS_REQUIRING_SLICES = {"slerp", "passthrough"}
METHODS_REQUIRING_MODELS = {"linear", "ties"}
TENSOR_CATEGORIES = ("self_attn", "mlp", "other")


def validate_raw_config(raw: dict) -> None:
    method = raw.get("merge_method")
    if method not in KNOWN_MERGE_METHODS:
        raise ValueError(f"unsupported merge_method: {method!r} (supported: {sorted(KNOWN_MERGE_METHODS)})")

    has_slices = "slices" in raw
    has_models = "models" in raw
    if has_slices == has_models:
        raise ValueError("config must specify exactly one of 'slices' or 'models'")

    if method in METHODS_REQUIRING_BASE_MODEL and "base_model" not in raw:
        raise ValueError(f"merge_method {method!r} requires 'base_model'")

    if method in METHODS_REQUIRING_SLICES and not has_slices:
        raise ValueError(f"merge_method {method!r} requires 'slices' with layer_range")

    if method in METHODS_REQUIRING_MODELS and not has_models:
        raise ValueError(f"merge_method {method!r} requires 'models'")


def extract_model_ids(raw: dict) -> list[str]:
    """All model ids referenced by a raw config, base_model first, in
    first-seen order, regardless of whether it uses 'models' or 'slices'.
    """
    ids: list[str] = []

    def add(model_id: str) -> None:
        if model_id not in ids:
            ids.append(model_id)

    if "base_model" in raw:
        add(raw["base_model"])

    for model_spec in raw.get("models", []):
        add(model_spec["model"])

    for slice_spec in raw.get("slices", []):
        for source in slice_spec["sources"]:
            add(source["model"])

    return ids
