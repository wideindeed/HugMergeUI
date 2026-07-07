"""Relative drift magnitude: how much total change two fine-tunes' updates
carry, relative to the base weight's own scale.

Phase 5 found that sign-conflict-rate (and its magnitude-weighted variant)
both failed to rank-order two independently-updated model pairs by merge
quality (measured via perplexity) - they're normalized ratios, blind to
absolute scale. The combined magnitude of the two updates, however, tracked
perplexity degradation exactly across all three tested pairs: more total
change forced into the merge meant a worse merge, independent of whether
that change agreed in sign.
"""

import torch


def drift_magnitude(diff_a: torch.Tensor, diff_b: torch.Tensor, base: torch.Tensor) -> float:
    if diff_a.shape != diff_b.shape or diff_a.shape != base.shape:
        raise ValueError(f"shape mismatch: {diff_a.shape}, {diff_b.shape}, {base.shape}")

    base_rms = base.float().pow(2).mean().sqrt().item()
    if base_rms == 0:
        return 0.0

    rms_a = diff_a.pow(2).mean().sqrt().item()
    rms_b = diff_b.pow(2).mean().sqrt().item()
    return (rms_a + rms_b) / base_rms
