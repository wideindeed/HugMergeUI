"""Network-dependent: fetches real config.json files from Hugging Face Hub.
huggingface_hub caches them locally after the first run, so only the very
first run pays the download cost.
"""

from app.hf.service import check_architecture

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
