"""Sign-conflict rate between two models' weight updates relative to a
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
