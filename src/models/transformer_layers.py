"""Transformer encoder layers for the inter-sentence summarization stage."""

import math

import numpy as np
import torch
import torch.nn as nn

from src.encoding.positional import get_position_encoding


def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with residual connection and layer normalization.

    Args:
        d_model: Size of input for the first layer.
        d_ff: Hidden layer size of the second layer.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(self.w_2.weight.dtype)
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """Multi-head attention mechanism for the summarization transformer.

    Args:
        head_count: Number of attention heads.
        model_dim: Dimension of the model.
        dropout: Dropout probability.
        use_final_linear: Whether to apply a final linear projection.
    """

    def __init__(self, head_count: int, model_dim: int, dropout: float = 0.1,
                 use_final_linear: bool = True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super().__init__()

        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if self.use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """Compute multi-head attention.

        Args:
            key: Key tensor.
            value: Value tensor.
            query: Query tensor.
            mask: Optional attention mask.
            layer_cache: Optional cached key/value states.
            type: Cache type ('self' or 'context').
            predefined_graph_1: Optional predefined attention graph.

        Returns:
            Context vector after attention computation.
        """
        batch_size = key.size(0)
        device = self.linear_keys.weight.device
        key = key.to(device).to(self.linear_keys.weight.dtype)
        value = value.to(device).to(self.linear_values.weight.dtype)
        query = query.to(device).to(self.linear_query.weight.dtype)
        if mask is not None:
            mask = mask.to(device)

        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        if layer_cache is not None:
            if type == "self":
                query, key, value = (
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat((layer_cache["self_keys"].to(device), key), dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat((layer_cache["self_values"].to(device), value), dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)

        key = shape(key)
        value = shape(value)
        query = shape(query)

        # Scaled dot-product attention
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        attn = self.softmax(scores)
        if predefined_graph_1 is not None:
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)
            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)

        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, value))
            return self.final_linear(context)
        else:
            return torch.matmul(drop_attn, value)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention and feed-forward.

    Args:
        d_model: Model dimension.
        heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.clone().detach()
        mask = mask.unsqueeze(1)

        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        inputs = inputs.to(context.device)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    """Inter-sentence transformer encoder for document-level summarization.

    Processes CLS token embeddings from the RoBERTa encoder through
    additional transformer layers to produce sentence-level scores.

    Args:
        d_model: Model dimension.
        seq_len: Sequence length for positional encoding.
        num_inter_layers: Number of stacked transformer layers.
        dropout: Dropout probability.
        heads: Number of attention heads.
    """

    def __init__(self, d_model: int, seq_len: int, num_inter_layers: int,
                 dropout: float, heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.d_ff = seq_len
        self.num_inter_layers = num_inter_layers
        self.pos_emb = get_position_encoding(seq_len, d_model, n=10000)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, 2048, dropout)
             for _ in range(num_inter_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_embeddings_tensor, mask, cls_indices, d):
        """Forward pass through inter-sentence transformer layers.

        Args:
            cls_embeddings_tensor: CLS embeddings (batch x n_sents x d).
            mask: Attention mask from the encoder.
            cls_indices: Indices of CLS tokens.
            d: Model dimension.

        Returns:
            Tuple of (sent_scores, cls_indices, hidden_states).
        """
        batch_size = cls_embeddings_tensor.size(0)
        n_sents = cls_embeddings_tensor.size(1)

        # Build CLS attention mask
        maskval = np.zeros((n_sents, d))
        for k in cls_indices:
            for i in range(n_sents):
                val = mask[0][k]
                integer_value = int(val.item())
                maskval[i, :] = integer_value

        x = cls_embeddings_tensor * torch.tensor(maskval).to(cls_embeddings_tensor.device)

        # Add positional encodings for CLS tokens
        posval = np.zeros((n_sents, d))
        for idx, k in enumerate(cls_indices):
            if idx < n_sents and k < self.pos_emb.shape[0]:
                posval[idx, :] = self.pos_emb[k, :]

        x = x + torch.tensor(posval).to(x.device)

        # Pass through transformer layers
        for i in range(self.num_inter_layers):
            maskval_tensor = torch.tensor(maskval).to(x.device)
            x = self.transformer_inter[i](i, x, x, 1 - maskval_tensor)

        x = self.layer_norm(x)

        # Classifier
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1)

        return sent_scores, cls_indices, x
