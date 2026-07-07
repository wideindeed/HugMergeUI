"""Network-dependent: downloads real model.safetensors weight files
(~1GB each) from Hugging Face Hub on first run, cached afterward.
"""

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from app.conflict.sign_conflict import magnitude_weighted_conflict_rate, sign_conflict_rate


@pytest.fixture(scope="module")
def real_weight_diff() -> torch.Tensor:
    base_path = hf_hub_download(repo_id="Qwen/Qwen2.5-0.5B", filename="model.safetensors")
    inst_path = hf_hub_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", filename="model.safetensors")

    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    with safe_open(base_path, framework="pt") as f_base, safe_open(inst_path, framework="pt") as f_inst:
        base_t = f_base.get_tensor(tensor_name).float()
        inst_t = f_inst.get_tensor(tensor_name).float()

    return inst_t - base_t


def test_identical_diff_has_zero_conflict(real_weight_diff):
    assert sign_conflict_rate(real_weight_diff, real_weight_diff) == 0.0


def test_negated_diff_has_full_conflict(real_weight_diff):
    assert sign_conflict_rate(real_weight_diff, -real_weight_diff) == 1.0


def test_random_diff_has_chance_level_conflict(real_weight_diff):
    torch.manual_seed(0)
    random_diff = torch.randn_like(real_weight_diff)
    rate = sign_conflict_rate(real_weight_diff, random_diff)
    assert 0.45 < rate < 0.55


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sign_conflict_rate(torch.zeros(4), torch.zeros(5))


def test_zero_diffs_excluded_from_denominator():
    a = torch.tensor([1.0, 0.0, -1.0])
    b = torch.tensor([1.0, 5.0, -1.0])
    assert sign_conflict_rate(a, b) == 0.0


def test_weighted_identical_diff_has_zero_conflict(real_weight_diff):
    assert magnitude_weighted_conflict_rate(real_weight_diff, real_weight_diff) == 0.0


def test_weighted_negated_diff_has_full_conflict(real_weight_diff):
    assert magnitude_weighted_conflict_rate(real_weight_diff, -real_weight_diff) == pytest.approx(1.0)


def test_weighted_shape_mismatch_raises():
    with pytest.raises(ValueError):
        magnitude_weighted_conflict_rate(torch.zeros(4), torch.zeros(5))


def test_weighted_conflict_favors_large_magnitude_over_count():
    """The whole point of the weighted metric: plain sign_conflict_rate
    counts this as 1 conflict out of 10 (10%), but the one conflict is a
    huge disagreement swamping nine tiny agreements - the weighted metric
    should reflect that most of the contested magnitude is in conflict.
    """
    diff_a = torch.tensor([0.01] * 9 + [10.0])
    diff_b = torch.tensor([0.01] * 9 + [-10.0])

    raw = sign_conflict_rate(diff_a, diff_b)
    weighted = magnitude_weighted_conflict_rate(diff_a, diff_b)

    assert raw == pytest.approx(0.1)
    assert weighted > 0.9
