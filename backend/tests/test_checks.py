from app.hf.checks import compare_architectures


def test_no_warnings_when_configs_match():
    configs = {
        "a": {"num_hidden_layers": 24, "hidden_size": 896, "vocab_size": 151936},
        "b": {"num_hidden_layers": 24, "hidden_size": 896, "vocab_size": 151936},
    }
    assert compare_architectures(configs) == []


def test_reports_each_mismatched_field():
    configs = {
        "a": {"num_hidden_layers": 24, "hidden_size": 896, "vocab_size": 151936},
        "b": {"num_hidden_layers": 22, "hidden_size": 2048, "vocab_size": 151936},
    }
    warnings = compare_architectures(configs)
    fields = {w["field"] for w in warnings if w["type"] == "architecture_mismatch"}
    assert fields == {"num_hidden_layers", "hidden_size"}


def test_flags_failed_fetch_and_skips_comparison_for_it():
    configs = {
        "a": {"num_hidden_layers": 24},
        "b": None,
    }
    warnings = compare_architectures(configs)
    assert {"type": "config_fetch_failed", "model": "b"} in warnings
    assert not any(w["type"] == "architecture_mismatch" for w in warnings)


def test_single_model_no_comparison():
    assert compare_architectures({"a": {"num_hidden_layers": 24}}) == []
