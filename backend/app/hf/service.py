"""Fetches architecture metadata for every model referenced by a config
and reports mismatches, plus derives the base model's layer count for
merge methods that need it (linear, ties)."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..parser.schema import extract_model_ids
from .checks import compare_architectures
from .client import ModelConfigFetchError, fetch_config_json, fetch_total_params, list_models_by_family

# Architecture families and parameter range where drift_magnitude has actually
# been validated against real merge quality (see VALIDATION.txt, Rounds 4-9).
# Outside this zone, results are exploratory rather than diagnostic.
VALIDATED_MODEL_TYPES = {"qwen2", "llama", "stablelm"}
VALIDATED_MIN_PARAMS = 1_300_000_000
VALIDATED_MAX_PARAMS = 3_200_000_000

# Below this, we're not just untested, VALIDATION.txt Rounds 1-3 measured
# 0.5B/360M directly and found the signal doesn't hold. "below_range" means
# known to break, not merely unmeasured.
#
# Round Sixteen re-tested this specifically at 0.5B/360M with a properly
# powered n=28-per-family sample (up from n=10). qwen2 cleared significance
# (drift_magnitude r=0.417, p=0.027) bracketing Qwen2.5-0.5B-scale models -
# the "known to break" verdict no longer holds for qwen2 in that narrow
# band. llama (tested via SmolLM2-360M) did not replicate (r=-0.345,
# p=0.072, still non-significant, still wrong-signed) - "below_range" stands
# for llama/stablelm, and for qwen2 outside this band too, since the gap
# between it and VALIDATED_MIN_PARAMS was never tested.
QWEN2_EXTENDED_MIN_PARAMS = 400_000_000
QWEN2_EXTENDED_MAX_PARAMS = 650_000_000


def _zone(model_type: str | None, total_params: int | None) -> str:
    if model_type is None or total_params is None:
        return "unknown"
    if model_type not in VALIDATED_MODEL_TYPES:
        return "untested_family"
    if model_type == "qwen2" and QWEN2_EXTENDED_MIN_PARAMS <= total_params <= QWEN2_EXTENDED_MAX_PARAMS:
        return "validated"
    if total_params < VALIDATED_MIN_PARAMS:
        return "below_range"
    if total_params > VALIDATED_MAX_PARAMS:
        return "above_range"
    return "validated"


def check_model(model_id: str) -> dict:
    """Reports architecture family, parameter count, and where this model
    sits relative to the validated zone, for a single arbitrary model id, no
    merge config required."""
    try:
        config = fetch_config_json(model_id)
    except ModelConfigFetchError as e:
        return {"model_type": None, "total_params": None, "validated": None, "zone": "unknown", "error": str(e)}

    model_type = config.get("model_type")
    total_params = fetch_total_params(model_id)
    zone = _zone(model_type, total_params)

    return {
        "model_type": model_type,
        "total_params": total_params,
        "validated": None if zone == "unknown" else zone == "validated",
        "zone": zone,
    }


_BROWSE_CANDIDATES_PER_FAMILY = 40
_BROWSE_CACHE_TTL_SECONDS = 30 * 60
_browse_cache: dict = {"models": None, "fetched_at": 0.0}


def _probe_candidate(model_id: str, downloads: int) -> dict | None:
    total_params = fetch_total_params(model_id)
    if total_params is None or not (QWEN2_EXTENDED_MIN_PARAMS <= total_params <= VALIDATED_MAX_PARAMS):
        return None
    try:
        config = fetch_config_json(model_id)
    except ModelConfigFetchError:
        return None
    model_type = config.get("model_type")
    if _zone(model_type, total_params) != "validated":
        return None
    return {"id": model_id, "model_type": model_type, "total_params": total_params, "downloads": downloads}


def browse_validated_models(force_refresh: bool = False) -> list[dict]:
    """The pool of models actually known to sit in the validated zone
    (family + 1-3B size), for the quick-compare model browser. Computed by
    fanning out to the Hub and caching the result, rather than hardcoding a
    list, so it stays current as new checkpoints are published."""
    now = time.time()
    cached = _browse_cache["models"]
    if not force_refresh and cached is not None and now - _browse_cache["fetched_at"] < _BROWSE_CACHE_TTL_SECONDS:
        return cached

    candidates: dict[str, int] = {}
    for family in VALIDATED_MODEL_TYPES:
        for model_id, downloads in list_models_by_family(family, limit=_BROWSE_CANDIDATES_PER_FAMILY):
            candidates[model_id] = downloads

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_probe_candidate, model_id, downloads): model_id for model_id, downloads in candidates.items()}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda m: m["downloads"], reverse=True)
    _browse_cache["models"] = results
    _browse_cache["fetched_at"] = now
    return results


def check_architecture(raw: dict) -> dict:
    model_ids = extract_model_ids(raw)

    configs: dict[str, dict | None] = {}
    for model_id in model_ids:
        try:
            configs[model_id] = fetch_config_json(model_id)
        except ModelConfigFetchError:
            configs[model_id] = None

    warnings = compare_architectures(configs)

    reference_model = raw.get("base_model") or (model_ids[0] if model_ids else None)
    reference_config = configs.get(reference_model) if reference_model else None
    num_layers = reference_config.get("num_hidden_layers") if reference_config else None

    return {
        "models": configs,
        "warnings": warnings,
        "num_layers": num_layers,
    }
