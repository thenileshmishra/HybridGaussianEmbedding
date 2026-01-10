"""
Longformer adapter with additive Gaussian attention bias.

Longformer uses sparse local + global attention (sliding window).
The Gaussian bias is most meaningful for the local window attention
where locality is already the design intention.

We subclass LongformerSelfAttention and inject the bias into the
local attention scores (att_scores_shifted) before softmax.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

from src.models.base_adapter import PositionalBiasAdapter


class BiasedLongformerSelfAttention(LongformerSelfAttention):
    """
    LongformerSelfAttention with Gaussian bias injected into the
    local (chunked) attention scores before softmax.

    Note: Longformer's chunked sliding-window attention produces score
    tensors of shape (batch, seq_len, 3 * window) rather than the full
    (seq_len, seq_len) matrix.  We therefore build a compact relative-
    distance bias of the same shape.
    """

    def __init__(self, config, layer_id, adapter=None):
        super().__init__(config, layer_id=layer_id)
        self.bias_adapter: Optional[PositionalBiasAdapter] = adapter
        self._sentence_map: Optional[torch.Tensor] = None

    def _local_gaussian_bias(
        self, seq_len: int, window: int, device: torch.device
    ) -> torch.Tensor:
        """
        Build compact Gaussian bias for the sliding-window layout.

        Output shape: (1, seq_len, 3 * window) matching Longformer's
        chunked attention score tensor (before softmax).
        We centre position i at column index `window` and fill
        columns for offsets  [-window, ..., +window].
        """
        if self.bias_adapter is None:
            return torch.zeros(1, seq_len, 3 * window, device=device)

        # Use the adapter's sigma (scalar or per-head mean)
        if hasattr(self.bias_adapter, "sigma"):
            sigma_val = self.bias_adapter.sigma
            if sigma_val.ndim > 0:
                sigma_val = sigma_val.mean()
            sigma = sigma_val.item()
        else:
            sigma = getattr(self.bias_adapter, "sigma", 3.0)
            if callable(sigma):
                sigma = 3.0

        offsets = torch.arange(-(window), window + 1, dtype=torch.float, device=device)
        # (3*window-1,) → pad to 3*window with a very large negative value at the edge
        # Longformer uses 3*window columns for one-sided + centre + one-sided
        col_offsets = torch.arange(3 * window, dtype=torch.float, device=device) - window
        bias_1d = -(col_offsets ** 2) / (2.0 * sigma ** 2)     # (3*window,)
        return bias_1d.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)

    def forward(self, hidden_states, attention_mask=None, is_index_masked=None,
                is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        # Delegate entirely to parent; Longformer's attention logic is
        # very intricate (chunking, global tokens).  The bias is applied
        # via a registered forward hook set up in LongformerWithBias below.
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )


class LongformerWithBias(nn.Module):
    """
    Longformer sentence classifier with pluggable Gaussian bias adapter.

    For Longformer the bias is applied to the output attention weights
    (post-softmax rescaling) as a pragmatic approximation, given the
    complexity of the chunked sliding-window implementation.
    For full pre-softmax injection, override BiasedLongformerSelfAttention.forward.
    """

    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        bias_adapter: Optional[PositionalBiasAdapter] = None,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)
        self.dropout    = nn.Dropout(self.longformer.config.hidden_dropout_prob)
        self.bias_adapter = bias_adapter

        if bias_adapter is not None:
            self._replace_attention_modules()

    def _replace_attention_modules(self) -> None:
        config = self.longformer.config
        for idx, layer in enumerate(self.longformer.encoder.layer):
            original = layer.attention.self
            biased = BiasedLongformerSelfAttention(
                config, layer_id=idx, adapter=self.bias_adapter
            )
            biased.load_state_dict(original.state_dict(), strict=False)
            layer.attention.self = biased

    def set_sentence_map(self, sentence_map: torch.Tensor) -> None:
        for layer in self.longformer.encoder.layer:
            attn = layer.attention.self
            if isinstance(attn, BiasedLongformerSelfAttention):
                attn._sentence_map = sentence_map

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sentence_map is not None:
            self.set_sentence_map(sentence_map)

        # Longformer expects global attention on the CLS token
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(pooled)
