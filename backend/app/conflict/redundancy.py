"""Redundant magnitude: how much of a single model's own weight-diff
signal TIES-style magnitude trimming would throw away at a given density.

TIES trims each tensor down to its top-`density` fraction of elements by
magnitude, zeroing the rest. Counting *how many* elements get zeroed
isn't informative (it's `1 - density` by construction) - what matters is
how much of the diff's total magnitude those pruned elements represent.
If a tensor's signal is concentrated in a few large elements, pruning
loses almost nothing; if it's spread evenly, pruning loses real signal.
"""

import torch


def redundant_magnitude_fraction(diff: torch.Tensor, density: float) -> float:
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")

    flat = diff.abs().flatten()
    total_mass = flat.sum()
    if total_mass.item() == 0.0:
        return 0.0

    n = flat.numel()
    k = min(n, max(0, round(density * n)))
    if k == 0:
        return 1.0
    if k == n:
        return 0.0

    kept_mass = torch.topk(flat, k).values.sum()
    return 1.0 - (kept_mass / total_mass).item()
