"""
RoBERTa adapter with pluggable Gaussian attention bias.

Strategy: subclass RobertaSelfAttention and override forward() to inject
the bias after raw attention scores are computed but before softmax.
Each layer's attention.self module is replaced with BiasSelfAttention
at model construction time — no runtime monkey-patching.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from src.models.base_adapter import PositionalBiasAdapter


class BiasSelfAttention(RobertaSelfAttention):
    """
    Drop-in replacement for RobertaSelfAttention that injects a
    PositionalBiasAdapter between score computation and softmax.
    """

    def __init__(self, config, position_embedding_type=None, adapter=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.bias_adapter: Optional[PositionalBiasAdapter] = adapter
        # sentence_map is set externally before each forward when using
        # the sentence-aware adapter; otherwise remains None.
        self._sentence_map: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        mixed_query = self.query(hidden_states)

        is_cross = encoder_hidden_states is not None
        if is_cross:
            key_layer   = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer   = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer   = torch.cat([past_key_value[0], key_layer],   dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer   = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Raw attention scores: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)

        # ── Gaussian bias injection ──────────────────────────────────────
        if self.bias_adapter is not None:
            scores = self.bias_adapter.inject_attention_bias(scores, self._sentence_map)
        # ────────────────────────────────────────────────────────────────

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        if head_mask is not None:
            probs = probs * head_mask

        context = torch.matmul(probs, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))

        outputs = (context, probs) if output_attentions else (context,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RoBERTaWithBias(nn.Module):
    """
    RoBERTa sentence classifier with a pluggable positional bias adapter.

    Usage:
        adapter = GaussianLearnableBias(num_heads=12)
        model   = RoBERTaWithBias("roberta-base", bias_adapter=adapter)
        logits  = model(input_ids, attention_mask, sentence_map=sent_map)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        bias_adapter: Optional[PositionalBiasAdapter] = None,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.dropout    = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.bias_adapter = bias_adapter

        if bias_adapter is not None:
            self._replace_attention_modules()

    def _replace_attention_modules(self) -> None:
        """Replace every RobertaSelfAttention with BiasSelfAttention."""
        config = self.roberta.config
        for layer in self.roberta.encoder.layer:
            original = layer.attention.self
            biased = BiasSelfAttention(config, adapter=self.bias_adapter)
            # Copy trained weights from the original module
            biased.load_state_dict(original.state_dict(), strict=False)
            layer.attention.self = biased

    def set_sentence_map(self, sentence_map: torch.Tensor) -> None:
        """Propagate sentence_map to all attention layers before a forward pass."""
        for layer in self.roberta.encoder.layer:
            attn = layer.attention.self
            if isinstance(attn, BiasSelfAttention):
                attn._sentence_map = sentence_map

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sentence_map is not None:
            self.set_sentence_map(sentence_map)

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token → sentence-level classifier
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(pooled)
