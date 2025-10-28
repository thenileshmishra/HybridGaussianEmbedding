"""RoBERTa encoder module for extractive summarization."""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class RoBERTaEncoder(nn.Module):
    """Custom RoBERTa encoder that accepts pre-computed input embeddings.

    Takes combined token + positional + segment embeddings as input
    and produces contextual CLS token embeddings for each sentence.

    Args:
        config: RobertaConfig instance.
        embeddings: Pre-initialized embedding module (for parameter sharing).
    """

    def __init__(self, config, embeddings):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            "roberta-base", output_attentions=True
        )
        config.update({
            "output_hidden_states": True,
            "layer_norm_eps": 1e-7,
            "max_position_embeddings": 1024,
        })

        self.model = RobertaModel.from_pretrained(
            "roberta-base", config=config, ignore_mismatched_sizes=True
        )
        self.embeddings = embeddings

        self.attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 768),
        )

    def forward(self, input_embeddings, cls_indices, attention_mask):
        """Forward pass through the RoBERTa encoder.

        Args:
            input_embeddings: Pre-computed embeddings (batch x seq_len x d).
            cls_indices: Indices of CLS tokens for each sentence.
            attention_mask: Attention mask tensor.

        Returns:
            Tuple of (cls_embeddings, attentions) where:
                - cls_embeddings: Stacked CLS token embeddings
                - attentions: Attention weights from all layers
        """
        input_embeddings = input_embeddings.to(dtype=torch.float32)  # Ensure consistent dtype
        input_embeddings = input_embeddings.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        outputs = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        embedding_output = outputs.last_hidden_state
        attention_tup = outputs.attentions

        # Extract CLS token embeddings for each sentence
        cls_embeddings = []
        for i in cls_indices:
            cls_embedding = embedding_output[:, i, :]
            cls_embeddings.append(cls_embedding)
        cls_embeddings_tensor = torch.stack(cls_embeddings)

        return cls_embeddings_tensor, attention_tup
