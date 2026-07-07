import pytest
import torch

from app.conflict.drift import drift_magnitude


def test_zero_diffs_have_zero_drift():
    base = torch.ones(10)
    zero = torch.zeros(10)
    assert drift_magnitude(zero, zero, base) == 0.0


def test_larger_diffs_have_larger_drift():
    base = torch.ones(10)
    small = torch.full((10,), 0.01)
    large = torch.full((10,), 1.0)

    small_drift = drift_magnitude(small, small, base)
    large_drift = drift_magnitude(large, large, base)
    assert large_drift > small_drift


def test_drift_is_relative_to_base_scale():
    """Same absolute diff, but a base tensor with 10x the magnitude, should
    read as a smaller relative drift - this is what lets the metric compare
    across differently-scaled layers (e.g. a down_proj vs. a layernorm)."""
    diff = torch.full((10,), 1.0)
    small_base = torch.full((10,), 1.0)
    large_base = torch.full((10,), 10.0)

    assert drift_magnitude(diff, diff, large_base) < drift_magnitude(diff, diff, small_base)


def test_sign_agreement_does_not_reduce_drift():
    """Unlike the conflict metrics, drift magnitude doesn't care whether the
    two updates agree or oppose in sign - only how large they are."""
    base = torch.ones(10)
    diff = torch.full((10,), 1.0)

    agreeing = drift_magnitude(diff, diff, base)
    opposing = drift_magnitude(diff, -diff, base)
    assert agreeing == pytest.approx(opposing)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        drift_magnitude(torch.zeros(4), torch.zeros(4), torch.zeros(5))
