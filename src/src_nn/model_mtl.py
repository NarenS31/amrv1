# src_nn/model_mtl.py
from __future__ import annotations
import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MultiTaskAMRNet(nn.Module):
    """
    Shared encoder -> per-antibiotic heads (multi-task).
    Output logits shape: (B, n_tasks)
    """

    def __init__(self, in_dim: int, n_tasks: int, hidden: int = 1024,
                 emb_dim: int = 256, n_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
        )
        self.resblocks = nn.Sequential(
            *[ResidualMLPBlock(emb_dim, dropout) for _ in range(n_blocks)])
        self.heads = nn.Linear(emb_dim, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.resblocks(z)
        logits = self.heads(z)
        return logits
