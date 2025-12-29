"""
Gaussian Relative Positional Bias — fixed sigma.

Adds G(i, j) = -(i - j)^2 / (2 * sigma^2) to every attention score
(batch, heads, seq_len, seq_len) before softmax.

sigma is a float hyperparameter set in configs/encoding/gaussian_relative.yaml.
"""

from typing import Optional

import torch

from src.models.base_adapter import PositionalBiasAdapter


class GaussianRelativeBias(PositionalBiasAdapter):
    """
    Fixed-sigma Gaussian relative positional bias.

    The bias matrix is recomputed for each forward pass so that it
    naturally handles variable sequence lengths.
    """

    def __init__(self, sigma: float = 3.0) -> None:
        super().__init__()
        self.sigma = sigma

    def inject_attention_bias(
        self,
        attention_scores: torch.Tensor,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = attention_scores.size(-1)
        device = attention_scores.device

        # Relative position matrix: (seq_len, seq_len)
        pos = torch.arange(seq_len, dtype=torch.float, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)          # (seq_len, seq_len)
        bias = -(rel ** 2) / (2.0 * self.sigma ** 2)       # (seq_len, seq_len)

        # Broadcast over (batch, heads)
        return attention_scores + bias.unsqueeze(0).unsqueeze(0)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}"
