"""Runs mergekit-yaml with the TIES magnitude-sparsify k=0 bug patched.

mergekit's `magnitude()` does `k = int(density * tensor.numel())` with no
floor, so any tensor small enough that `density * numel < 1` crashes with
"not gonna zero out the whole tensor buddy" instead of just retaining it.
GPT-NeoX's `attention.masked_bias` is a 1-element buffer that hits this at
density=0.5 - and it's a required (non-optional) tensor in mergekit's
architecture definition, so it can't just be dropped from the source model.
Flooring k at 1 keeps the intended "retain the top-density fraction"
behavior for real tensors and degrades gracefully (keep-the-only-element)
for this one degenerate case. mergekit-yaml runs as its own subprocess, so
the patch is applied here rather than in the calling process.
"""

import sys

import torch
import mergekit.sparsify as sparsify


def _patched_magnitude(tensor, density, rescale_norm=None):
    if density >= 1:
        return tensor
    k = max(1, int(density * tensor.numel()))
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.argsort(w, descending=True)[:k]
    mask.view(-1)[topk] = 1
    return sparsify.rescaled_masked_tensor(tensor, mask, rescale_norm)


sparsify.magnitude = _patched_magnitude

from mergekit.scripts.run_yaml import main  # noqa: E402

if __name__ == "__main__":
    main()
