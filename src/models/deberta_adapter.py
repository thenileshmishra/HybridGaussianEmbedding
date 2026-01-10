"""
DeBERTa adapter with additive Gaussian attention bias.

DeBERTa uses disentangled relative positional encoding internally.
Per the PRD mitigation strategy, we inject the Gaussian bias *additively*
without altering the content-position separation:

    att_logits += gaussian_bias   (before XSoftmax)

We subclass DebertaSelfOutput... no — the injection point is in
DisentangledSelfAttention.forward() after att_score is assembled.
We subclass that class directly.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import DebertaModel
from transformers.models.deberta.modeling_deberta import (
    DisentangledSelfAttention,
    XSoftmax,
)

from src.models.base_adapter import PositionalBiasAdapter


class BiasedDisentangledSelfAttention(DisentangledSelfAttention):
    """
    DisentangledSelfAttention with additive Gaussian bias before XSoftmax.

    Inherits the full disentangled relative attention logic; we only
    intercept the moment after att_score is fully assembled.
    """

    def __init__(self, config, adapter=None):
        super().__init__(config)
        self.bias_adapter: Optional[PositionalBiasAdapter] = adapter
        self._sentence_map: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        # ── replicate parent score assembly ──────────────────────────────
        if query_states is None:
            query_states = hidden_states

        query_layer = self.transpose_for_scores(self.query_proj(query_states))
        key_layer   = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))

        rel_att = None
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        att_score = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            att_score = att_score + rel_att

        # ── Gaussian bias injection ──────────────────────────────────────
        if self.bias_adapter is not None:
            att_score = self.bias_adapter.inject_attention_bias(
                att_score, self._sentence_map
            )
        # ────────────────────────────────────────────────────────────────

        att_probs = XSoftmax.apply(att_score, attention_mask, -1)
        att_probs = self.dropout(att_probs)

        context_layer = torch.matmul(att_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_shape)

        if output_attentions:
            return (context_layer, att_probs)
        return (context_layer,)


class DeBERTaWithBias(nn.Module):
    """
    DeBERTa sentence classifier with pluggable Gaussian bias adapter.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-base",
        bias_adapter: Optional[PositionalBiasAdapter] = None,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.deberta    = DebertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
        self.dropout    = nn.Dropout(self.deberta.config.hidden_dropout_prob)
        self.bias_adapter = bias_adapter

        if bias_adapter is not None:
            self._replace_attention_modules()

    def _replace_attention_modules(self) -> None:
        config = self.deberta.config
        for layer in self.deberta.encoder.layer:
            original = layer.attention.self
            biased = BiasedDisentangledSelfAttention(config, adapter=self.bias_adapter)
            biased.load_state_dict(original.state_dict(), strict=False)
            layer.attention.self = biased

    def set_sentence_map(self, sentence_map: torch.Tensor) -> None:
        for layer in self.deberta.encoder.layer:
            attn = layer.attention.self
            if isinstance(attn, BiasedDisentangledSelfAttention):
                attn._sentence_map = sentence_map

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sentence_map is not None:
            self.set_sentence_map(sentence_map)

        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(pooled)
