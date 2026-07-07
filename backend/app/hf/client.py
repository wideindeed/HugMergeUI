"""Fetches config.json (architecture metadata only, no weights) for a
Hugging Face model. huggingface_hub caches the file locally after the
first fetch, so repeated calls for the same model are instant.
"""

import json

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, HFValidationError


class ModelConfigFetchError(Exception):
    pass


def fetch_config_json(model_id: str) -> dict:
    try:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
    except (HfHubHTTPError, HFValidationError) as e:
        raise ModelConfigFetchError(f"could not fetch config.json for {model_id!r}: {e}") from e

    with open(path) as f:
        return json.load(f)
