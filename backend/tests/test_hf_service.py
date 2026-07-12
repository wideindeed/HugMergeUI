"""Network-dependent: fetches real config.json files from Hugging Face Hub.
huggingface_hub caches them locally after the first run, so only the very
first run pays the download cost.
"""

from app.hf.service import (
    QWEN2_EXTENDED_MAX_PARAMS,
    QWEN2_EXTENDED_MIN_PARAMS,
    VALIDATED_MAX_PARAMS,
    VALIDATED_MIN_PARAMS,
    VALIDATED_MODEL_TYPES,
    browse_validated_models,
    check_architecture,
    check_model,
)

_IN_ZONE_RANGES = [(VALIDATED_MIN_PARAMS, VALIDATED_MAX_PARAMS), (QWEN2_EXTENDED_MIN_PARAMS, QWEN2_EXTENDED_MAX_PARAMS)]

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


def test_check_model_in_qwen2_extended_zone():
    # Round Sixteen (VALIDATION.txt): re-tested at this exact scale with
    # n=28 and found drift_magnitude reaches significance for qwen2, unlike
    # llama at the same size - so this is validated, not below_range.
    result = check_model("Qwen/Qwen2.5-0.5B")
    assert result["model_type"] == "qwen2"
    assert result["validated"] is True
    assert result["zone"] == "validated"


def test_check_model_below_validated_range_wrong_family_stays_below_range():
    # SmolLM2-360M is llama-architecture and was tested at this exact scale
    # in Round Sixteen (n=28) - it did NOT replicate qwen2's result, so
    # llama has no extended zone and this must stay below_range.
    result = check_model("HuggingFaceTB/SmolLM2-360M")
    assert result["model_type"] == "llama"
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
        in_a_validated_range = any(lo <= m["total_params"] <= hi for lo, hi in _IN_ZONE_RANGES)
        assert in_a_validated_range
        if m["model_type"] != "qwen2":
            assert VALIDATED_MIN_PARAMS <= m["total_params"] <= VALIDATED_MAX_PARAMS


def test_browse_validated_models_is_cached():
    first = browse_validated_models(force_refresh=True)
    second = browse_validated_models()
    assert first is second
