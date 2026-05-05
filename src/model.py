"""
ParaCNN extractive summarizer — faithful re-implementation of the notebook
(see important/ParaCNN_PosEmbed.ipynb).

Architecture pipeline (per document):
  1. RoBertaDocumentEmbedder
       - tokenize each sentence with RoBERTa tokenizer; concatenate up to max_len=800
       - T (seq_len, d): T[i, j] = input_ids[i]   (broadcast token id across all dims)
       - S (seq_len, d): alternating segment 0/1 between SEP positions
       - H (seq_len, d): hybrid PE = sinusoidal + windowed Gaussian
            * Gaussian per-dim scaling: dim_scaling = linspace(s_min, s_max, d//2)
            * Gaussian is windowed: nonzero only on tokens k in [mu - sigma, mu + sigma]
       - sum_embeddings = T + S + H  →  fed into RoBERTa via inputs_embeds
  2. RoBERTA wrapper
       - RobertaModel.forward(inputs_embeds=sum_embeddings, attention_mask=...)
       - gather CLS embeddings at each per-sentence CLS index
  3. TransformerInterEncoder
       - sentence-level sinusoidal PE added at each CLS position
       - num_inter_layers (default 2) custom TransformerEncoderLayer
       - layer norm + linear → sigmoid sentence scores
  4. logit_layer  (nn.Linear(d, 1)) — used for BCEWithLogitsLoss training

Default GA-tuned hyperparameters from the paper:
    s_max=2  s_min=0.3  mu=260  sigma=65
For the sinusoidal-only baseline pass: s_max=s_min=mu=sigma=0
"""

import math
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast


# ============================ Encoding helpers ============================

def get_position_encoding(seq_len, d, n=10000):
    """Standard sinusoidal positional encoding (seq_len, d)."""
    P = np.zeros((seq_len, d))
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d // 2)[None, :]
    denom = np.power(n, 2 * i / d)
    P[:, 0::2] = np.sin(pos / denom)
    P[:, 1::2] = np.cos(pos / denom)
    return P


def get_gaussian_encoding(seq_len, d, s_max, s_min, mu, sigma):
    """
    Windowed Gaussian PE with per-dimension sigma scaling.
    Pairs G[k, 2i] and G[k, 2i+1] match (so the sum with sin-cos pair stays
    distinguishable).  Nonzero only for tokens k in [mu - sigma, mu + sigma].
    Returns zeros if sigma <= 0 (sinusoidal-only baseline mode).
    """
    G = np.zeros((seq_len, d))
    if sigma is None or sigma <= 0:
        return G
    if s_max == 0 and s_min == 0:
        return G
    start = max(0, int(mu - sigma))
    end   = min(seq_len, int(mu + sigma))
    if end <= start:
        return G
    dim_scaling = np.linspace(s_min, s_max, d // 2)            # (d/2,)
    sigmas      = sigma * dim_scaling                           # (d/2,)
    sigmas      = np.where(sigmas == 0, 1e-12, sigmas)          # avoid div-by-0 at s_min=0 edge
    k = np.arange(start, end)                                   # (W,)
    diff = (k[:, None] - mu) ** 2                               # (W, 1)
    G_pairs = np.exp(-diff / (2.0 * sigmas[None, :] ** 2))      # (W, d/2)
    G[start:end, 0::2] = G_pairs
    G[start:end, 1::2] = G_pairs
    return G


def get_hybrid_encoding(seq_len, d, s_max, s_min, mu, sigma, n=10000):
    """Hybrid = sinusoidal + windowed Gaussian (sum)."""
    return get_position_encoding(seq_len, d, n) + \
           get_gaussian_encoding(seq_len, d, s_max, s_min, mu, sigma)


def get_segment_encoding(seq_len, d, sep_indices):
    """Alternate 0/1 between SEP positions. Broadcast across all dims."""
    S = np.zeros((seq_len, d))
    counter = 0
    prev = 0
    for j in sep_indices:
        seg = counter % 2
        if j + 1 > prev:
            S[prev:j + 1, :] = seg
        counter += 1
        prev = j + 1
    return S


# ============================ Document embedder ============================

class RoBertaDocumentEmbedder(nn.Module):
    """
    Build the token + segment + hybrid-position embedding for a single document.
    Returns the sum (1, seq_len, d) ready to be fed to RoBERTa via inputs_embeds.
    """

    def __init__(self, max_len=800, d=768, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.d = d
        self.tokenizer = tokenizer or RobertaTokenizerFast.from_pretrained('roberta-base')

    def forward(self, input_text, s_max, s_min, mu, sigma):
        seq_len, d = self.max_len, self.d

        # 1) tokenize each sentence and concatenate the token streams
        input_ids = []
        attention_mask = []
        for sent in input_text:
            enc = self.tokenizer(sent, return_attention_mask=True,
                                 add_special_tokens=True)
            input_ids.extend(enc['input_ids'])
            attention_mask.extend(enc['attention_mask'])

        input_ids      = (input_ids      + [0] * seq_len)[:seq_len]
        attention_mask = (attention_mask + [0] * seq_len)[:seq_len]

        # 2) Token encoding: T[i, j] = input_ids[i]   (broadcast across all d dims)
        T = np.tile(np.array(input_ids, dtype=np.float32)[:, None], (1, d))

        # 3) Segment encoding: alternating 0/1 between </s> tokens (id=2 in RoBERTa)
        sep_indices = [i for i, t in enumerate(input_ids) if t == 2]
        S = get_segment_encoding(seq_len, d, sep_indices)

        # 4) Hybrid position encoding
        H = get_hybrid_encoding(seq_len, d, s_max, s_min, mu, sigma, n=10000)

        # 5) Sum
        sum_embeddings = T + S + H
        sum_embeddings = torch.from_numpy(sum_embeddings).float().unsqueeze(0)  # (1, seq_len, d)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        return sum_embeddings, attention_mask, sep_indices


# ============================ RoBERTa wrapper ============================

class RoBERTA(nn.Module):
    """
    RobertaModel that consumes pre-built embeddings via inputs_embeds and
    returns CLS embeddings (one per sentence) gathered at the given indices.
    """

    def __init__(self, max_position_embeddings=1024):
        super().__init__()
        config = RobertaConfig.from_pretrained("roberta-base", output_attentions=True)
        config.update({
            "output_hidden_states": True,
            "layer_norm_eps": 1e-7,
            "max_position_embeddings": max_position_embeddings,
        })
        self.model = RobertaModel.from_pretrained(
            "roberta-base", config=config, ignore_mismatched_sizes=True
        )

    def forward(self, input_embeddings, cls_indices, attention_mask):
        device = self.model.device
        input_embeddings = input_embeddings.float().to(device)
        attention_mask   = attention_mask.to(device)
        out = self.model(inputs_embeds=input_embeddings,
                         attention_mask=attention_mask,
                         output_attentions=True)
        h = out.last_hidden_state  # (B, L, d)

        cls_embeddings = []
        for i in cls_indices:
            cls_embeddings.append(h[:, i, :])           # (B, d)
        cls_tensor = torch.stack(cls_embeddings)        # (n_cls, B, d)
        return cls_tensor, out.attentions


# ============================ Inter-sentence transformer ============================

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
                                     (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(self.w_2.weight.dtype)
        inter  = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Per the notebook, the mask is computed but NOT applied to the softmax
    (the masked_fill block is commented out in the original). Preserved.
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        super().__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim    = model_dim
        self.head_count   = head_count
        self.linear_keys   = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query  = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        B = key.size(0)
        h = self.head_count
        d_per_h = self.dim_per_head
        device = self.linear_keys.weight.device
        key   = key.to(device).to(self.linear_keys.weight.dtype)
        value = value.to(device).to(self.linear_values.weight.dtype)
        query = query.to(device).to(self.linear_query.weight.dtype)

        def shape(x):
            return x.view(B, -1, h, d_per_h).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(B, -1, h * d_per_h)

        key   = shape(self.linear_keys(key))
        value = shape(self.linear_values(value))
        query = shape(self.linear_query(query))

        query = query / math.sqrt(d_per_h)
        scores = torch.matmul(query, key.transpose(2, 3))
        # NOTE: masked_fill block is intentionally omitted to match the notebook
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, value))
            return self.final_linear(context)
        return torch.matmul(drop_attn, value)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn    = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm   = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, iter_idx, query, inputs, mask):
        if iter_idx != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs
        mask = mask.clone().detach().unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        inputs = inputs.to(context.device)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    """
    Sentence-level (inter-sentence) transformer.  Adds sinusoidal PE *at the
    CLS positions of the document*, then runs num_inter_layers blocks of
    self-attention + FFN, then a layer-norm + sigmoid head.
    """

    def __init__(self, d_model, seq_len, num_inter_layers, dropout, heads=6):
        super().__init__()
        self.d_model = d_model
        self.d_ff = seq_len                  # ParaCNN sets d_ff = seq_len (=800)
        self.num_inter_layers = num_inter_layers
        self.register_buffer(
            'pos_emb',
            torch.tensor(get_position_encoding(seq_len, d_model, n=10000), dtype=torch.float32)
        )
        self.transformer_inter = nn.ModuleList([
            TransformerEncoderLayer(d_model, heads, self.d_ff, dropout)
            for _ in range(num_inter_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_embeddings_tensor, attention_mask, cls_indices, d):
        # cls_embeddings_tensor: (B, n_sents, d)
        B, n_sents, _ = cls_embeddings_tensor.size()
        device = cls_embeddings_tensor.device

        # Per-sentence mask gathered from the token-level attention_mask at each CLS position.
        am = attention_mask[0]                                # (L,)
        gather_idx = torch.tensor(cls_indices, device=am.device, dtype=torch.long)
        per_sent = am.index_select(0, gather_idx).to(device).to(cls_embeddings_tensor.dtype)  # (n_sents,)
        maskval_t = per_sent.unsqueeze(1).expand(n_sents, d)  # (n_sents, d)

        x = cls_embeddings_tensor * maskval_t

        # Add sinusoidal PE at the CLS positions (token positions, not sentence indices).
        pos_emb = self.pos_emb.to(device)
        posval_t = pos_emb.index_select(0, gather_idx.to(device))         # (n_sents, d)
        x = x + posval_t.unsqueeze(0)

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - maskval_t)

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x)).squeeze(-1)    # (B, n_sents)
        return sent_scores, cls_indices, x


# ============================ Top-level wrapper ============================

class ParaCNNExt(nn.Module):
    """
    Wraps embedder + RoBERTa + inter-encoder + logit head into one nn.Module.
    Forward is per-document (input_text is a list of sentence strings).
    Returns (logits, n_sentences) where logits is (1, n_sentences) — raw, BCE-ready.
    """

    def __init__(self,
                 max_len=800,
                 d=768,
                 num_inter_layers=2,
                 heads=6,
                 dropout=0.1,
                 max_position_embeddings=1024,
                 tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.d = d
        self.embedder      = RoBertaDocumentEmbedder(max_len=max_len, d=d, tokenizer=tokenizer)
        self.roberta       = RoBERTA(max_position_embeddings=max_position_embeddings)
        self.inter_encoder = TransformerInterEncoder(d, max_len, num_inter_layers, dropout, heads)
        self.logit_layer   = nn.Linear(d, 1)

    def _device(self):
        return next(self.parameters()).device

    def forward(self, input_text, s_max, s_min, mu, sigma):
        sum_emb, am, sep_indices = self.embedder(input_text, s_max, s_min, mu, sigma)
        # CLS positions: [0] + [s+1 for s in sep_indices][:-1]
        cls_indices = [0] + [s + 1 for s in sep_indices]
        cls_indices = cls_indices[:-1]
        cls_indices = [i for i in cls_indices if i < self.max_len]
        if len(cls_indices) == 0:
            return None, 0

        cls_t, _ = self.roberta(sum_emb, cls_indices, am)     # (n_cls, 1, d)
        cls_reshaped = cls_t.permute(1, 0, 2)                  # (1, n_cls, d)
        _, _, x = self.inter_encoder(cls_reshaped, am, cls_indices, self.d)
        logits = self.logit_layer(x).squeeze(-1)               # (1, n_cls)
        return logits, len(cls_indices)
