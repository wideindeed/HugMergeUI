from pathlib import Path

import pytest
import yaml

from app.parser.normalize import normalize

FIXTURES = Path(__file__).parent / "fixtures" / "mergekit_examples"


def load(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_passthrough_layer_count_and_source_mapping():
    result = normalize(load("orcamini-platy-44layer.yml"))

    assert len(result["layers"]) == 44
    assert result["layers"][0]["sources"] == [
        {"model": "psmathur/orca_mini_v3_13b", "source_layer_index": 0}
    ]
    assert result["layers"][23]["sources"][0]["source_layer_index"] == 23
    assert result["layers"][24]["sources"] == [
        {"model": "garage-bAInd/Platypus2-13B", "source_layer_index": 20}
    ]
    assert result["layers"][43]["sources"][0]["source_layer_index"] == 39


def test_slerp_blend_gradient_matches_filters():
    result = normalize(load("gradient-slerp.yml"))

    assert len(result["layers"]) == 40
    assert result["models"] == ["psmathur/orca_mini_v3_13b", "garage-bAInd/Platypus2-13B"]

    first, last = result["layers"][0], result["layers"][-1]
    assert first["blend_t"]["self_attn"] == 0
    assert first["blend_t"]["mlp"] == 1
    assert first["blend_t"]["other"] == 0.5
    assert last["blend_t"]["self_attn"] == 1
    assert last["blend_t"]["mlp"] == 0


def test_linear_scalar_weights_constant_across_layers():
    result = normalize(load("linear.yml"), num_layers=4)

    assert result["merge_parameters"] is None
    assert len(result["layers"]) == 4
    weights = {
        s["model"]: s["parameters"]["self_attn"]["weight"]
        for s in result["layers"][2]["sources"]
    }
    assert weights == {
        "psmathur/orca_mini_v3_13b": 1.0,
        "WizardLM/WizardLM-13B-V1.2": 0.3,
        "garage-bAInd/Platypus2-13B": 0.5,
    }


def test_linear_requires_num_layers():
    with pytest.raises(ValueError):
        normalize(load("linear.yml"))


def test_ties_density_gradient_and_weight_filter():
    result = normalize(load("ties.yml"), num_layers=3)

    assert result["merge_parameters"] == {"normalize": True, "int8_mask": True}

    orca, platy, wizardmath = result["layers"][0]["sources"]
    assert orca["parameters"]["self_attn"]["density"] == 1
    assert platy["parameters"]["self_attn"]["density"] == 0.5

    wizardmath_last = result["layers"][2]["sources"][2]
    assert wizardmath_last["parameters"]["mlp"]["weight"] == 0.5
    assert wizardmath_last["parameters"]["self_attn"]["weight"] == 0
