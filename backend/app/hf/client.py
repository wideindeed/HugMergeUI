"""Fetches config.json (architecture metadata only, no weights) for a
Hugging Face model. huggingface_hub caches the file locally after the
first fetch, so repeated calls for the same model are instant.
"""

import json

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, HFValidationError

_api = HfApi()


class ModelConfigFetchError(Exception):
    pass


def fetch_config_json(model_id: str) -> dict:
    try:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
    except (HfHubHTTPError, HFValidationError) as e:
        raise ModelConfigFetchError(f"could not fetch config.json for {model_id!r}: {e}") from e

    with open(path) as f:
        return json.load(f)


def search_model_ids(query: str, limit: int = 15) -> list[str]:
    """Live search against the HF Hub for autocomplete, no weights or configs
    are fetched here, just repo names."""
    if not query.strip():
        return []
    models = _api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
    return [m.id for m in models]


def fetch_total_params(model_id: str) -> int | None:
    """Best-effort parameter count from repo metadata (safetensors header),
    no weight download required. Returns None if the repo doesn't publish
    safetensors metadata."""
    try:
        info = _api.model_info(model_id)
    except (HfHubHTTPError, HFValidationError):
        return None
    if info.safetensors and info.safetensors.total:
        return info.safetensors.total
    return None


def list_models_by_family(model_type: str, limit: int) -> list[tuple[str, int]]:
    """Repo ids and download counts for the most-downloaded models tagged
    with a given architecture family, no configs or weights fetched here."""
    models = _api.list_models(filter=model_type, sort="downloads", direction=-1, limit=limit)
    return [(m.id, m.downloads or 0) for m in models]
