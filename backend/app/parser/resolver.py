"""Resolves mergekit parameter specs to a concrete value for one tensor.

A parameter spec is one of:
  - a scalar (int/float/bool/str) -> used as-is everywhere
  - a list of scalars -> a depth gradient, linearly interpolated across
    layer_index/num_layers
  - a list of {filter, value} rules -> matched in order against the tensor
    name (substring match); a rule with no "filter" key matches anything
    and acts as a fallback. The matched rule's value is itself a scalar
    or gradient, resolved the same way.
"""

from typing import Any


def resolve_parameter(spec: Any, *, tensor_name: str, layer_index: int, num_layers: int) -> Any:
    if _is_filter_rules(spec):
        spec = _select_filter_rule(spec, tensor_name)
    return _resolve_scalar_or_gradient(spec, layer_index, num_layers)


def _is_filter_rules(spec: Any) -> bool:
    return isinstance(spec, list) and len(spec) > 0 and all(isinstance(item, dict) for item in spec)


def _select_filter_rule(rules: list[dict], tensor_name: str) -> Any:
    for rule in rules:
        filt = rule.get("filter")
        if filt is None or filt in tensor_name:
            return rule["value"]
    raise ValueError(f"no matching filter rule for tensor {tensor_name!r}")


def _resolve_scalar_or_gradient(value: Any, layer_index: int, num_layers: int) -> Any:
    if not isinstance(value, list):
        return value
    if len(value) == 1 or num_layers <= 1:
        return value[0]

    t = layer_index / (num_layers - 1)
    control_positions = [i / (len(value) - 1) for i in range(len(value))]

    for i, pos in enumerate(control_positions):
        if t == pos:
            return value[i]

    for i in range(len(control_positions) - 1):
        lo, hi = control_positions[i], control_positions[i + 1]
        if lo < t < hi:
            local_t = (t - lo) / (hi - lo)
            return value[i] + local_t * (value[i + 1] - value[i])
    return value[-1]
