"""Learnable pruning gate for DAG-scheduled execution.

The gate observes execution state features and predicts Δ̂_k — the advantage
of stopping now vs running all remaining operators. Positive Δ means stopping
is better (same/better accuracy + latency/cost savings).

Training: MSE regression on Δ_k from execution traces.
Inference: p(stop) = sigmoid(Δ̂_k / τ), then sample Bernoulli.
"""

import torch
import torch.nn as nn


class PruningGate(nn.Module):
    """MLP that maps execution state + compressed query embedding to Δ̂_k.

    Input: (state_features [2-dim], query_embedding [384-dim]) → concatenated internally.
    The query embedding is first compressed 384→4 via a learned linear projection,
    then concatenated with the 2 state features → 6-dim → hidden → 1.

    State features (2-dim):
        0: n_completed / n_total          — execution progress
        1: max_agreement / n_solutions    — consensus rate (0 if no solutions)

    Output: scalar Δ̂_k (unbounded, positive = stop is better)
    """

    STATE_DIM = 2
    QUERY_DIM = 384
    QUERY_COMPRESSED_DIM = 4

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.query_proj = nn.Linear(self.QUERY_DIM, self.QUERY_COMPRESSED_DIM)
        self.net = nn.Sequential(
            nn.Linear(self.STATE_DIM + self.QUERY_COMPRESSED_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        state: shape (6,) or (B, 6)
        query_emb: shape (384,) or (B, 384)
        Returns: shape () or (B,)
        """
        q = self.query_proj(query_emb)  # (4,) or (B, 4)
        x = torch.cat([state, q], dim=-1)  # (10,) or (B, 10)
        return self.net(x).squeeze(-1)
