"""Fetches architecture metadata for every model referenced by a config
and reports mismatches, plus derives the base model's layer count for
merge methods that need it (linear, ties)."""

from ..parser.schema import extract_model_ids
from .checks import compare_architectures
from .client import ModelConfigFetchError, fetch_config_json


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
