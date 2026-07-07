from pathlib import Path

import pytest

from app.parser.loader import load_config

FIXTURES = Path(__file__).parent / "fixtures" / "mergekit_examples"


def read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_load_slerp_config_end_to_end():
    result = load_config(read("gradient-slerp.yml"))
    assert result["merge_method"] == "slerp"
    assert len(result["layers"]) == 40


def test_load_rejects_invalid_yaml():
    with pytest.raises(ValueError):
        load_config("not: valid: yaml: [")


def test_load_rejects_non_mapping_yaml():
    with pytest.raises(ValueError):
        load_config("- just\n- a\n- list\n")


def test_load_linear_requires_num_layers_end_to_end():
    with pytest.raises(ValueError):
        load_config(read("linear.yml"))

    result = load_config(read("linear.yml"), num_layers=2)
    assert len(result["layers"]) == 2
