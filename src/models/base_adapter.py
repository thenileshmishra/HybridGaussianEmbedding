"""
Base contract for all positional encoding adapters.

Two injection points are supported:
  1. Attention-level  — inject_attention_bias() adds bias before softmax.
  2. Embedding-level  — replace_position_embeddings() swaps the embedding table.

All encoding variants must subclass PositionalBiasAdapter and implement
inject_attention_bias(). The embedding-level hook is optional (default: no-op).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class PositionalBiasAdapter(ABC, nn.Module):
    """
    Interface that every positional encoding variant must satisfy.

    Subclasses are plugged into model wrappers (RoBERTaWithBias, etc.)
    which call inject_attention_bias() inside the attention forward pass,
    immediately after computing raw attention scores and before softmax.
    """

    @abstractmethod
    def inject_attention_bias(
        self,
        attention_scores: torch.Tensor,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add positional bias to raw attention scores (before softmax).

        Args:
            attention_scores: FloatTensor (batch, heads, seq_len, seq_len).
            sentence_map:     LongTensor  (batch, seq_len) with integer
                              sentence indices. None for token-level adapters.

        Returns:
            Biased attention scores, same shape as input.
        """
        ...

    def replace_position_embeddings(
        self,
        model: nn.Module,
        config: dict,
    ) -> nn.Module:
        """
        Optional hook for embedding-level positional replacement.

        Override this only for absolute-embedding variants (sinusoidal,
        gaussian_absolute). Attention-level adapters can ignore it.

        Returns the model unchanged by default.
        """
        return model
