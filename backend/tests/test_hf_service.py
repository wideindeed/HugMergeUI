"""Network-dependent: fetches real config.json files from Hugging Face Hub.
huggingface_hub caches them locally after the first run, so only the very
first run pays the download cost.
"""

from app.hf.service import VALIDATED_MAX_PARAMS, VALIDATED_MIN_PARAMS, VALIDATED_MODEL_TYPES, browse_validated_models, check_architecture, check_model

SAME_ARCH_CONFIG = {
    "merge_method": "linear",
    "models": [
        {"model": "Qwen/Qwen2.5-0.5B"},
        {"model": "Qwen/Qwen2.5-0.5B-Instruct"},
    ],
}

MISMATCHED_ARCH_CONFIG = {
    "merge_method": "linear",
    "models": [
        {"model": "Qwen/Qwen2.5-0.5B"},
        {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
    ],
}


def test_same_architecture_pair_has_no_warnings_and_resolves_num_layers():
    result = check_architecture(SAME_ARCH_CONFIG)
    assert result["warnings"] == []
    assert result["num_layers"] == result["models"]["Qwen/Qwen2.5-0.5B"]["num_hidden_layers"]


def test_mismatched_architecture_pair_flags_differences():
    result = check_architecture(MISMATCHED_ARCH_CONFIG)
    mismatch_fields = {w["field"] for w in result["warnings"] if w["type"] == "architecture_mismatch"}
    assert "hidden_size" in mismatch_fields
    assert "vocab_size" in mismatch_fields


def test_unknown_model_reports_fetch_failure():
    result = check_architecture({
        "merge_method": "linear",
        "models": [{"model": "this-org/does-not-exist-hugmergeui-test"}],
    })
    assert result["warnings"] == [
        {"type": "config_fetch_failed", "model": "this-org/does-not-exist-hugmergeui-test"}
    ]
    assert result["num_layers"] is None


def test_check_model_in_validated_zone():
    result = check_model("Qwen/Qwen2.5-1.5B")
    assert result["model_type"] == "qwen2"
    assert result["validated"] is True
    assert result["zone"] == "validated"


def test_check_model_below_validated_range():
    result = check_model("Qwen/Qwen2.5-0.5B")
    assert result["model_type"] == "qwen2"
    assert result["validated"] is False
    assert result["zone"] == "below_range"


def test_check_model_below_validated_range_wrong_family_too():
    result = check_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert result["model_type"] == "llama"
    assert result["zone"] == "below_range"


def test_check_model_untested_family():
    result = check_model("EleutherAI/pythia-1.4b")
    assert result["model_type"] == "gpt_neox"
    assert result["zone"] == "untested_family"


def test_check_model_unknown_repo_reports_error():
    result = check_model("this-org/does-not-exist-hugmergeui-test")
    assert result["validated"] is None
    assert result["zone"] == "unknown"
    assert result["error"]


def test_browse_validated_models_only_returns_in_zone_models():
    results = browse_validated_models(force_refresh=True)
    assert len(results) > 0
    for m in results:
        assert m["model_type"] in VALIDATED_MODEL_TYPES
        assert VALIDATED_MIN_PARAMS <= m["total_params"] <= VALIDATED_MAX_PARAMS


def test_browse_validated_models_is_cached():
    first = browse_validated_models(force_refresh=True)
    second = browse_validated_models()
    assert first is second
