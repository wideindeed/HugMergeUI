from pathlib import Path

import yaml

from app.parser.schema import extract_model_ids

FIXTURES = Path(__file__).parent / "fixtures" / "mergekit_examples"


def load(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_extract_from_models_list():
    ids = extract_model_ids(load("linear.yml"))
    assert ids == [
        "psmathur/orca_mini_v3_13b",
        "WizardLM/WizardLM-13B-V1.2",
        "garage-bAInd/Platypus2-13B",
    ]


def test_extract_from_slices_puts_base_model_first():
    ids = extract_model_ids(load("gradient-slerp.yml"))
    assert ids == ["psmathur/orca_mini_v3_13b", "garage-bAInd/Platypus2-13B"]


def test_extract_dedupes():
    ids = extract_model_ids(load("ties.yml"))
    assert ids == [
        "TheBloke/Llama-2-13B-fp16",
        "psmathur/orca_mini_v3_13b",
        "garage-bAInd/Platypus2-13B",
        "WizardLM/WizardMath-13B-V1.0",
    ]
