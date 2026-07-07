"""Sign-conflict metrics between two models' weight updates relative to a
shared base, per the TIES-Merging notion of interference: for each
parameter, does model A's update direction disagree with model B's?

Elements where either update is exactly zero are excluded from both the
numerator and denominator — a parameter one model didn't touch can't be
in conflict.
"""

import torch


def sign_conflict_rate(diff_a: torch.Tensor, diff_b: torch.Tensor) -> float:
    if diff_a.shape != diff_b.shape:
        raise ValueError(f"shape mismatch: {diff_a.shape} vs {diff_b.shape}")

    sign_a = torch.sign(diff_a)
    sign_b = torch.sign(diff_b)

    comparable = (sign_a != 0) & (sign_b != 0)
    total = comparable.sum().item()
    if total == 0:
        return 0.0

    conflicts = ((sign_a != sign_b) & comparable).sum().item()
    return conflicts / total


def magnitude_weighted_conflict_rate(diff_a: torch.Tensor, diff_b: torch.Tensor) -> float:
    """Like sign_conflict_rate, but weighted by how much magnitude is
    actually in dispute rather than a raw element count.

    Plain sign-conflict-rate is blind to magnitude: for two independent
    random updates it converges toward ~50% regardless of whether the
    disagreements are tiny numerical noise or large, load-bearing changes.
    Weighting each element by min(|diff_a|, |diff_b|) - the amount of
    update that's actually contested, since you can't disagree by more
    than the smaller of the two magnitudes - lets large, confident
    disagreements dominate over a sea of small ones.
    """
    if diff_a.shape != diff_b.shape:
        raise ValueError(f"shape mismatch: {diff_a.shape} vs {diff_b.shape}")

    sign_a = torch.sign(diff_a)
    sign_b = torch.sign(diff_b)

    comparable = (sign_a != 0) & (sign_b != 0)
    weight = torch.minimum(diff_a.abs(), diff_b.abs()) * comparable
    total_weight = weight.sum().item()
    if total_weight == 0:
        return 0.0

    conflict_weight = (weight * (sign_a != sign_b)).sum().item()
    return conflict_weight / total_weight
