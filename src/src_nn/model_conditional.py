from __future__ import annotations
import torch
import torch.nn as nn


class ConditionalAMRNet(nn.Module):
    """
    Predicts S/I/R conditioned on antibiotic ID.
    Inputs:
      x: (B, D) kmer features
      ab: (B,) antibiotic id
    Output:
      logits: (B, 3)
    """

    def __init__(self, in_dim: int, n_antibiotics: int, ab_emb_dim: int = 64,
                 hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.ab_emb = nn.Embedding(n_antibiotics, ab_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim + ab_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        e = self.ab_emb(ab)                 # (B, ab_emb_dim)
        z = torch.cat([x, e], dim=1)        # (B, D+ab_emb_dim)
        return self.net(z)                  # (B, 3)
