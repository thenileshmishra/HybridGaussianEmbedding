"""
BERTSum extractive summarization model (G0 baseline, no Gaussian).

Architecture:
    [CLS] S1 [SEP] [CLS] S2 [SEP] ...
        -> Transformer backbone (RoBERTa-base by default)
        -> gather CLS embeddings per sentence       (B, N, d)
        -> add fixed sinusoidal sentence-level PE
        -> 2-layer inter-sentence Transformer
        -> Linear(d -> 1)                           per-sentence logit

The forward returns raw logits; BCEWithLogitsLoss is applied externally.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


def sinusoidal_pe(max_len, d, n=10000.0):
    """Standard sin/cos positional encoding."""
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.pow(n, torch.arange(0, d, 2).float() / d)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


class BERTSumExt(nn.Module):
    def __init__(
        self,
        backbone_name='roberta-base',
        d=768,
        n_heads=6,
        n_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        max_sents=50,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.register_buffer('sent_pe', sinusoidal_pe(max_sents, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.inter_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask, cls_positions, cls_mask):
        """
        Args:
            input_ids:      (B, L) token ids
            attention_mask: (B, L) 1/0 mask over tokens
            cls_positions:  (B, N) per-sentence [CLS] index in the token sequence
                            (padded with 0; use cls_mask to identify real slots)
            cls_mask:       (B, N) 1.0 for real sentence, 0.0 for padding

        Returns:
            logits:         (B, N) per-sentence logit (BCE-ready)
        """
        # Backbone (no token_type_ids -> RoBERTa uses zeros internally)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                      # (B, L, d)

        # Gather per-sentence CLS embeddings
        d = h.size(-1)
        idx = cls_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
        cls_h = torch.gather(h, dim=1, index=idx)      # (B, N, d)

        # Zero out padded slots, add sentence-level PE
        cls_h = cls_h * cls_mask.unsqueeze(-1)
        N = cls_h.size(1)
        cls_h = cls_h + self.sent_pe[:N].unsqueeze(0)
        cls_h = self.dropout(cls_h)

        # Inter-sentence transformer (mask out padded sentences)
        pad_mask = (cls_mask == 0)                     # True = ignore
        x = self.inter_encoder(cls_h, src_key_padding_mask=pad_mask)

        # Per-sentence logit
        logits = self.classifier(x).squeeze(-1)        # (B, N)
        return logits
