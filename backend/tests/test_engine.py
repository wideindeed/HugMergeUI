import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from app.conflict.engine import _extract_layer_index, score_model_pair, score_tensors


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


def test_score_model_pair_identical_models_have_near_zero_conflict():
    """Network-dependent, uses cached weights from earlier tests/sessions."""
    result = score_model_pair(
        "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    assert len(result["layers"]) > 0
    assert all(layer["conflict"] == 0.0 for layer in result["layers"])
