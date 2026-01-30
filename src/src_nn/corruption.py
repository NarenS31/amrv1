# src_nn/corruption.py
from __future__ import annotations
import torch


def make_block_indices(n_features: int, block_size: int) -> torch.Tensor:
    """
    Returns a tensor of shape (n_blocks, 2) where each row is [start, end)
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    starts = list(range(0, n_features, block_size))
    blocks = [(s, min(s + block_size, n_features)) for s in starts]
    return torch.tensor(blocks, dtype=torch.long)


def block_dropout(x: torch.Tensor,
                  blocks: torch.Tensor,
                  drop_frac: float,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Structured feature dropout by dropping whole contiguous blocks of features.
    x: (B, D)
    blocks: (n_blocks, 2) of [start, end)
    drop_frac: fraction of blocks to drop (0..1)
    """
    if drop_frac <= 0:
        return x
    if drop_frac >= 1:
        return torch.zeros_like(x)

    b, d = x.shape
    n_blocks = blocks.shape[0]
    k = max(1, int(round(drop_frac * n_blocks)))

    # choose blocks to drop
    idx = torch.randperm(n_blocks, generator=generator, device=x.device)[:k]
    chosen = blocks[idx]  # (k, 2)

    x2 = x.clone()
    # apply block masking
    for s, e in chosen.tolist():
        x2[:, s:e] = 0.0
    return x2
