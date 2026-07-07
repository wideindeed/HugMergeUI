"""Compares architecture metadata across a merge's referenced models and
surfaces mismatches as structured warnings the frontend can flag on the
diagram, rather than plain strings.
"""

ARCHITECTURE_FIELDS = ("num_hidden_layers", "hidden_size", "vocab_size")


def compare_architectures(configs: dict[str, dict | None]) -> list[dict]:
    warnings = []

    missing = [name for name, config in configs.items() if config is None]
    for name in missing:
        warnings.append({"type": "config_fetch_failed", "model": name})

    present = {name: config for name, config in configs.items() if config is not None}
    names = list(present)
    if len(names) < 2:
        return warnings

    reference_name = names[0]
    reference_config = present[reference_name]

    for field in ARCHITECTURE_FIELDS:
        reference_value = reference_config.get(field)
        for name in names[1:]:
            value = present[name].get(field)
            if value != reference_value:
                warnings.append({
                    "type": "architecture_mismatch",
                    "field": field,
                    "values": {reference_name: reference_value, name: value},
                })

    return warnings
