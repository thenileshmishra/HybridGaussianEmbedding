"""
BERTSum extractive summarization model.

Supports variants:
    G0 — sinusoidal sentence PE only (baseline)
    G1 — fixed-scalar Gaussian bias  (mu=0.5, sigma=0.2, alpha=1.0)
    G2 — learnable-scalar Gaussian   (mu, sigma, alpha are nn.Parameters)
    G3 — fixed mu/sigma, learnable vector w per dimension
    G4 — fully learnable Gaussian    (mu, sigma, vector w)

Gaussian formula:  G(i) = exp( -(i_norm - mu)^2 / (2*sigma^2) )
Applied as:        cls_h[i] += alpha * G(i)        for G1/G2
                   cls_h[i] += G(i) * w             for G3/G4
where i_norm = i / (N-1), normalized to [0, 1].

Sigma parameterization for G2/G4:
    Stored as raw_sigma; effective sigma = softplus(raw_sigma) + _MIN_SIGMA
    This prevents sigma from collapsing to zero at large-scale training.
    _MIN_SIGMA=0.05 keeps the Gaussian spread over at least 1/20 of the doc.
    raw_sigma is initialized to -1.82 so that effective sigma ≈ 0.20 at start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

_MIN_SIGMA = 0.05
# softplus(-1.82) + 0.05 ≈ 0.147 + 0.05 = 0.197 ≈ 0.20
_INIT_RAW_SIGMA = -1.82


def sinusoidal_pe(max_len, d, n=10000.0):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.pow(n, torch.arange(0, d, 2).float() / d)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


class GaussianBias(nn.Module):
    """Sentence-position Gaussian bias added to CLS embeddings."""

    def __init__(self, d, variant='G0'):
        super().__init__()
        self.variant = variant

        if variant == 'G1':
            self.register_buffer('mu',    torch.tensor(0.5))
            self.register_buffer('sigma', torch.tensor(0.2))
            self.register_buffer('alpha', torch.tensor(1.0))
        elif variant == 'G2':
            self.mu        = nn.Parameter(torch.tensor(0.5))
            self.raw_sigma = nn.Parameter(torch.tensor(_INIT_RAW_SIGMA))
            self.alpha     = nn.Parameter(torch.tensor(1.0))
        elif variant == 'G3':
            self.register_buffer('mu',    torch.tensor(0.5))
            self.register_buffer('sigma', torch.tensor(0.2))
            self.w = nn.Parameter(torch.zeros(d))
        elif variant == 'G4':
            self.mu        = nn.Parameter(torch.tensor(0.5))
            self.raw_sigma = nn.Parameter(torch.tensor(_INIT_RAW_SIGMA))
            self.w         = nn.Parameter(torch.zeros(d))

    def _effective_sigma(self):
        """Effective sigma: softplus(raw_sigma)+_MIN_SIGMA for learnable variants, buffer for fixed."""
        if hasattr(self, 'raw_sigma'):
            return F.softplus(self.raw_sigma) + _MIN_SIGMA
        return self.sigma

    def forward(self, cls_h, cls_mask):
        """
        Args:
            cls_h:    (B, N, d) sentence embeddings
            cls_mask: (B, N)   1 for real sentences, 0 for padding
        Returns:
            cls_h + Gaussian bias, same shape
        """
        if self.variant == 'G0':
            return cls_h

        B, N, d = cls_h.shape
        positions = torch.arange(N, device=cls_h.device).float() / max(N - 1, 1)  # (N,)
        mu    = self.mu.clamp(0.01, 0.99)
        sigma = self._effective_sigma()

        gauss = torch.exp(-((positions - mu) ** 2) / (2 * sigma ** 2))  # (N,)
        gauss = gauss.unsqueeze(0) * cls_mask                            # (B, N)

        if self.variant in ('G1', 'G2'):
            bias = self.alpha * gauss.unsqueeze(-1)                      # (B, N, 1) -> (B, N, d)
        else:  # G3, G4
            bias = gauss.unsqueeze(-1) * self.w                          # (B, N, d)

        return cls_h + bias

    def log_params(self):
        """Current parameter values for console logging (sigma shown as effective value)."""
        if self.variant == 'G0':
            return {}
        out = {'mu': self.mu.item(), 'sigma': self._effective_sigma().item()}
        if self.variant in ('G1', 'G2'):
            out['alpha'] = self.alpha.item()
        else:
            out['w_norm'] = self.w.norm().item()
        return out


class BERTSumExt(nn.Module):
    def __init__(
        self,
        backbone_name='roberta-base',
        variant='G0',
        d=768,
        n_heads=6,
        n_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        max_sents=50,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.gaussian  = GaussianBias(d, variant=variant)
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
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask, cls_positions, cls_mask):
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            cls_positions:  (B, N) [CLS] token index per sentence, 0-padded
            cls_mask:       (B, N) 1.0 for real sentence, 0.0 for padding
        Returns:
            logits: (B, N) raw per-sentence scores
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h   = out.last_hidden_state                                   # (B, L, d)

        d   = h.size(-1)
        idx = cls_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
        cls_h = torch.gather(h, dim=1, index=idx)                     # (B, N, d)

        cls_h = cls_h * cls_mask.unsqueeze(-1)                        # zero padding
        cls_h = self.gaussian(cls_h, cls_mask)                        # Gaussian bias
        N     = cls_h.size(1)
        cls_h = cls_h + self.sent_pe[:N].unsqueeze(0)                 # sinusoidal PE
        cls_h = self.dropout(cls_h)

        pad_mask = (cls_mask == 0)                                    # True = ignore
        x = self.inter_encoder(cls_h, src_key_padding_mask=pad_mask)

        logits = self.classifier(x).squeeze(-1)                       # (B, N)
        return logits
