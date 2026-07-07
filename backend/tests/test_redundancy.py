import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from app.conflict.redundancy import redundant_magnitude_fraction


def test_uniform_magnitude_redundancy_matches_count_fraction():
    diff = torch.ones(1000)
    assert redundant_magnitude_fraction(diff, density=0.5) == pytest.approx(0.5, abs=0.01)


def test_concentrated_signal_survives_pruning_far_better_than_uniform():
    diff = torch.cat([torch.tensor([1000.0]), torch.ones(99)])
    rate = redundant_magnitude_fraction(diff, density=0.5)
    assert rate < 0.1


def test_density_one_keeps_everything():
    diff = torch.randn(100)
    assert redundant_magnitude_fraction(diff, density=1.0) == 0.0


def test_density_zero_prunes_everything():
    diff = torch.randn(100)
    assert redundant_magnitude_fraction(diff, density=0.0) == 1.0


def test_all_zero_diff_has_no_redundancy():
    assert redundant_magnitude_fraction(torch.zeros(100), density=0.5) == 0.0


def test_invalid_density_raises():
    with pytest.raises(ValueError):
        redundant_magnitude_fraction(torch.randn(10), density=1.5)


@pytest.fixture(scope="module")
def real_weight_diff() -> torch.Tensor:
    base_path = hf_hub_download(repo_id="Qwen/Qwen2.5-0.5B", filename="model.safetensors")
    inst_path = hf_hub_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", filename="model.safetensors")
    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    with safe_open(base_path, framework="pt") as f_base, safe_open(inst_path, framework="pt") as f_inst:
        return (f_inst.get_tensor(tensor_name).float() - f_base.get_tensor(tensor_name).float())


def test_real_diff_redundancy_decreases_as_density_increases(real_weight_diff):
    low_density = redundant_magnitude_fraction(real_weight_diff, density=0.1)
    high_density = redundant_magnitude_fraction(real_weight_diff, density=0.9)
    assert 0.0 <= high_density < low_density <= 1.0
