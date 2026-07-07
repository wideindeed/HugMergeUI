import json

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from app.conflict.engine import (
    _extract_layer_index,
    _load_tensors,
    _weight_map_from_index,
    score_model_pair,
    score_tensors,
)


def test_extract_layer_index():
    assert _extract_layer_index("model.layers.14.self_attn.q_proj.weight") == 14
    assert _extract_layer_index("model.embed_tokens.weight") is None
    assert _extract_layer_index("model.layers.0.mlp.down_proj.weight") == 0


def test_aggregates_by_layer_and_buckets_non_layer_tensors():
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(10),
        "model.layers.1.self_attn.q_proj.weight": torch.zeros(10),
        "model.embed_tokens.weight": torch.zeros(10),
    }
    shared_diff = torch.ones(10)
    model_a = {
        "model.layers.0.self_attn.q_proj.weight": shared_diff,
        "model.layers.1.self_attn.q_proj.weight": shared_diff,
        "model.embed_tokens.weight": shared_diff,
    }
    model_b = {
        "model.layers.0.self_attn.q_proj.weight": shared_diff,       # agrees -> 0 conflict
        "model.layers.1.self_attn.q_proj.weight": -shared_diff,      # opposes -> full conflict
        "model.embed_tokens.weight": shared_diff,
    }

    result = score_tensors(base, model_a, model_b, density=0.5)

    layers_by_index = {layer["layer"]: layer for layer in result["layers"]}
    assert layers_by_index[0]["conflict"] == 0.0
    assert layers_by_index[1]["conflict"] == 1.0
    assert result["other"]["conflict"] == 0.0
    assert result["other"]["tensor_count"] == 1


def test_skips_mismatched_shapes():
    base = {"model.layers.0.x": torch.zeros(10)}
    model_a = {"model.layers.0.x": torch.ones(10)}
    model_b = {"model.layers.0.x": torch.ones(5)}  # shape mismatch

    result = score_tensors(base, model_a, model_b)
    assert result["layers"] == []
    assert result["other"] is None


def test_weighted_average_favors_larger_tensor():
    base = {
        "model.layers.0.small": torch.zeros(2),
        "model.layers.0.big": torch.zeros(100),
    }
    model_a = {
        "model.layers.0.small": torch.ones(2),
        "model.layers.0.big": torch.ones(100),
    }
    model_b = {
        "model.layers.0.small": -torch.ones(2),   # full conflict, tiny tensor
        "model.layers.0.big": torch.ones(100),    # no conflict, huge tensor
    }

    result = score_tensors(base, model_a, model_b)
    assert result["layers"][0]["conflict"] < 0.05  # dominated by the big agreeing tensor


def test_weight_map_from_index(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({
        "metadata": {"total_size": 123},
        "weight_map": {
            "model.layers.0.x": "model-00001-of-00002.safetensors",
            "model.layers.1.x": "model-00002-of-00002.safetensors",
        },
    }))

    weight_map = _weight_map_from_index(str(index_path))
    assert weight_map == {
        "model.layers.0.x": "model-00001-of-00002.safetensors",
        "model.layers.1.x": "model-00002-of-00002.safetensors",
    }


def test_load_tensors_reads_across_multiple_shard_files(tmp_path):
    """A sharded model spreads tensors across several .safetensors files;
    _load_tensors must open each shard only once and reassemble one dict."""
    shard_a = tmp_path / "shard-a.safetensors"
    shard_b = tmp_path / "shard-b.safetensors"
    save_file({"model.layers.0.x": torch.ones(4)}, str(shard_a))
    save_file({"model.layers.1.x": torch.zeros(4)}, str(shard_b))

    tensor_files = {
        "model.layers.0.x": str(shard_a),
        "model.layers.1.x": str(shard_b),
    }
    tensors = _load_tensors(tensor_files, {"model.layers.0.x", "model.layers.1.x"})

    assert torch.equal(tensors["model.layers.0.x"], torch.ones(4))
    assert torch.equal(tensors["model.layers.1.x"], torch.zeros(4))


def test_score_model_pair_identical_models_have_near_zero_conflict():
    """Network-dependent, uses cached weights from earlier tests/sessions."""
    result = score_model_pair(
        "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    assert len(result["layers"]) > 0
    assert all(layer["conflict"] == 0.0 for layer in result["layers"])


def test_score_model_pair_independent_finetunes_show_real_conflict():
    """The actual target scenario: two unrelated SFT tunes of the same base
    (Dolphin3.0, trained by cognitivecomputations, and a Capybara SFT run by
    a different author) show near-chance sign conflict, unlike the
    identical-model case above which is exactly 0. Confirms the metric
    differentiates genuine independent drift from a degenerate self-pair.
    """
    result = score_model_pair(
        "Qwen/Qwen2.5-0.5B",
        "dphn/Dolphin3.0-Qwen2.5-0.5B",
        "wulli/Qwen2.5-0.5B-sft-capybara",
    )
    layers = result["layers"]
    assert len(layers) == 24

    avg_conflict = sum(layer["conflict"] for layer in layers) / len(layers)
    assert 0.4 < avg_conflict < 0.5

    assert all(0.0 < layer["redundancy_a"] < 1.0 for layer in layers)
    assert all(0.0 < layer["redundancy_b"] < 1.0 for layer in layers)


@pytest.mark.slow
def test_score_model_pair_handles_real_sharded_model():
    """Qwen2.5-3B ships as 2 safetensors shards + an index.json - the
    common case for any model past ~5GB, i.e. most real mergekit targets.
    Downloads ~6GB on first run; skip with `-m "not slow"` for quick loops.
    """
    result = score_model_pair(
        "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
    )
    assert len(result["layers"]) > 0
    assert all(layer["conflict"] == 0.0 for layer in result["layers"])
