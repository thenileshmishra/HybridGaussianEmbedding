"""
Sentence-Aware Gaussian Positional Bias.

Instead of token-level distance, uses sentence-index distance:

    G_h(i, j) = -(s_i - s_j)^2 / (2 * sigma_h^2)

where s_i is the sentence index of token i, provided by a precomputed
sentence_map tensor of shape (batch, seq_len).

Motivation: extractive summarization operates at sentence granularity;
modelling locality at that level should improve sentence selection.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_adapter import PositionalBiasAdapter


class SentenceAwareGaussianBias(PositionalBiasAdapter):
    """
    Learnable per-head Gaussian bias over sentence-index distance.

    Requires sentence_map to be passed into inject_attention_bias().
    Falls back to identity (no bias) if sentence_map is None so it does
    not crash when the map is unavailable (e.g., during sanity checks).

    Args:
        num_heads:   Number of attention heads.
        sigma_init:  Initial sigma for all heads.
    """

    def __init__(self, num_heads: int, sigma_init: float = 1.5) -> None:
        super().__init__()
        log_init = torch.full((num_heads,), fill_value=torch.log(torch.tensor(sigma_init)))
        self.log_sigma = nn.Parameter(log_init)

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def inject_attention_bias(
        self,
        attention_scores: torch.Tensor,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sentence_map is None:
            return attention_scores

        device = attention_scores.device

        # sentence_map: (batch, seq_len)
        s = sentence_map.float().to(device)
        # Sentence-index relative distances: (batch, seq_len, seq_len)
        rel_sent = s.unsqueeze(2) - s.unsqueeze(1)

        sigma = self.sigma.to(device)                       # (heads,)
        # (batch, heads, seq_len, seq_len)
        bias = -(rel_sent.unsqueeze(1) ** 2) / (2.0 * sigma.view(1, -1, 1, 1) ** 2)

        return attention_scores + bias

    def extra_repr(self) -> str:
        sigma_vals = self.sigma.detach().cpu().tolist()
        return f"num_heads={len(sigma_vals)}, sigma_init≈{sigma_vals[0]:.2f}"
