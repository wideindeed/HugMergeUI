from pathlib import Path

import pytest
import yaml

from app.parser.resolver import resolve_parameter

FIXTURES = Path(__file__).parent / "fixtures" / "mergekit_examples"


def load(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_scalar_passthrough():
    assert resolve_parameter(0.5, tensor_name="model.layers.0.mlp", layer_index=0, num_layers=10) == 0.5


def test_gradient_endpoints_and_midpoint():
    config = load("gradient-slerp.yml")
    t_rules = config["parameters"]["t"]
    self_attn_value = next(r["value"] for r in t_rules if r.get("filter") == "self_attn")

    assert resolve_parameter(self_attn_value, tensor_name="x", layer_index=0, num_layers=5) == 0
    assert resolve_parameter(self_attn_value, tensor_name="x", layer_index=1, num_layers=5) == 0.5
    assert resolve_parameter(self_attn_value, tensor_name="x", layer_index=2, num_layers=5) == 0.3
    assert resolve_parameter(self_attn_value, tensor_name="x", layer_index=4, num_layers=5) == 1


def test_filter_matches_self_attn_vs_mlp_vs_fallback():
    config = load("gradient-slerp.yml")
    t_rules = config["parameters"]["t"]

    attn = resolve_parameter(t_rules, tensor_name="model.layers.3.self_attn.q_proj", layer_index=0, num_layers=5)
    mlp = resolve_parameter(t_rules, tensor_name="model.layers.3.mlp.down_proj", layer_index=0, num_layers=5)
    other = resolve_parameter(t_rules, tensor_name="model.embed_tokens", layer_index=0, num_layers=5)

    assert attn == 0
    assert mlp == 1
    assert other == 0.5


def test_ties_weight_filter_with_scalar_fallback():
    config = load("ties.yml")
    weight_rules = config["models"][2]["parameters"]["weight"]

    mlp = resolve_parameter(weight_rules, tensor_name="model.layers.0.mlp.up_proj", layer_index=0, num_layers=1)
    other = resolve_parameter(weight_rules, tensor_name="model.layers.0.self_attn.q_proj", layer_index=0, num_layers=1)

    assert mlp == 0.5
    assert other == 0


def test_ties_density_gradient_no_filter():
    config = load("ties.yml")
    density = config["models"][0]["parameters"]["density"]

    assert resolve_parameter(density, tensor_name="anything", layer_index=0, num_layers=3) == pytest.approx(1)
    assert resolve_parameter(density, tensor_name="anything", layer_index=1, num_layers=3) == pytest.approx(0.7)
    assert resolve_parameter(density, tensor_name="anything", layer_index=2, num_layers=3) == pytest.approx(0.1)


def test_no_matching_filter_raises():
    with pytest.raises(ValueError):
        resolve_parameter([{"filter": "mlp", "value": 1.0}], tensor_name="self_attn.q_proj", layer_index=0, num_layers=1)
