"""Entrypoint tying YAML parsing, validation, and normalization together."""

import yaml

from .normalize import normalize
from .schema import validate_raw_config


def parse_raw_config(yaml_text: str) -> dict:
    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping at the top level")

    validate_raw_config(raw)
    return raw


def load_config(yaml_text: str, *, num_layers: int | None = None) -> dict:
    raw = parse_raw_config(yaml_text)
    return normalize(raw, num_layers=num_layers)
