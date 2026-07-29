import pytest

from app.parser.schema import validate_raw_config


def test_unsupported_merge_method_raises():
    with pytest.raises(ValueError, match="unsupported merge_method"):
        validate_raw_config({"merge_method": "dare_ties", "models": []})


def test_missing_merge_method_raises():
    with pytest.raises(ValueError, match="unsupported merge_method"):
        validate_raw_config({"models": []})


def test_both_slices_and_models_raises():
    with pytest.raises(ValueError, match="exactly one of"):
        validate_raw_config({"merge_method": "linear", "models": [], "slices": []})


def test_neither_slices_nor_models_raises():
    with pytest.raises(ValueError, match="exactly one of"):
        validate_raw_config({"merge_method": "linear"})


def test_slerp_without_base_model_raises():
    with pytest.raises(ValueError, match="requires 'base_model'"):
        validate_raw_config({"merge_method": "slerp", "slices": []})


def test_ties_without_base_model_raises():
    with pytest.raises(ValueError, match="requires 'base_model'"):
        validate_raw_config({"merge_method": "ties", "models": []})


def test_linear_does_not_require_base_model():
    validate_raw_config({"merge_method": "linear", "models": []})


def test_slerp_without_slices_raises():
    with pytest.raises(ValueError, match="requires 'slices'"):
        validate_raw_config({"merge_method": "slerp", "base_model": "base", "models": []})


def test_passthrough_without_slices_raises():
    with pytest.raises(ValueError, match="requires 'slices'"):
        validate_raw_config({"merge_method": "passthrough", "models": []})


def test_linear_without_models_raises():
    with pytest.raises(ValueError, match="requires 'models'"):
        validate_raw_config({"merge_method": "linear", "slices": []})


def test_ties_without_models_raises():
    with pytest.raises(ValueError, match="requires 'models'"):
        validate_raw_config({"merge_method": "ties", "base_model": "base", "slices": []})


def test_passthrough_does_not_require_models():
    validate_raw_config({"merge_method": "passthrough", "slices": []})


def test_valid_ties_config_passes():
    validate_raw_config({"merge_method": "ties", "base_model": "base", "models": []})


def test_valid_slerp_config_passes():
    validate_raw_config({"merge_method": "slerp", "base_model": "base", "slices": []})
