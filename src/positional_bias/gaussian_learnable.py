"""
Gaussian Relative Positional Bias — learnable per-head sigma.

sigma_h is an nn.Parameter per attention head, stored as log_sigma
to ensure sigma > 0 throughout training:

    sigma_h = exp(log_sigma_h)
    G_h(i, j) = -(i - j)^2 / (2 * sigma_h^2)

Each head can learn a different locality scale, which is the core
contribution of this thesis chapter.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_adapter import PositionalBiasAdapter


class GaussianLearnableBias(PositionalBiasAdapter):
    """
    Learnable per-head sigma Gaussian relative positional bias.

    Args:
        num_heads:   Number of attention heads.
        sigma_init:  Initial value for all sigma_h  (default 3.0).
    """

    def __init__(self, num_heads: int, sigma_init: float = 3.0) -> None:
        super().__init__()
        # Parameterise in log-space so exp() always gives positive sigma
        log_init = torch.full((num_heads,), fill_value=torch.log(torch.tensor(sigma_init)))
        self.log_sigma = nn.Parameter(log_init)

    @property
    def sigma(self) -> torch.Tensor:
        """Per-head sigma values, shape (num_heads,)."""
        return self.log_sigma.exp()

    def inject_attention_bias(
        self,
        attention_scores: torch.Tensor,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _batch, heads, seq_len, _ = attention_scores.shape
        device = attention_scores.device

        pos = torch.arange(seq_len, dtype=torch.float, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)           # (seq_len, seq_len)

        sigma = self.sigma.to(device)                       # (heads,)
        # (heads, seq_len, seq_len)  — broadcast over heads
        bias = -(rel.unsqueeze(0) ** 2) / (2.0 * sigma.view(-1, 1, 1) ** 2)

        # Broadcast over batch dim
        return attention_scores + bias.unsqueeze(0)

    def extra_repr(self) -> str:
        sigma_vals = self.sigma.detach().cpu().tolist()
        return f"num_heads={len(sigma_vals)}, sigma_init≈{sigma_vals[0]:.2f}"
