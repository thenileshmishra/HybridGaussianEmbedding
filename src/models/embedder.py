"""RoBERTa document embedder that combines token, position, and segment encodings."""

import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel, RobertaTokenizer

from src.encoding.token import get_token_encoding
from src.encoding.positional import get_position_encoding, get_hybrid_encoding
from src.encoding.segment import get_segment_encoding


class RoBertaDocumentEmbedder(nn.Module):
    """Generates input embeddings by combining token, positional, and segment encodings.

    The input embedding for the RoBERTa encoder is computed as:
        E = T + P + S
    where T is the token encoding, P is the positional (or hybrid) encoding,
    and S is the segment encoding.
    """

    def __init__(self, model_name: str = "roberta-base", seq_len: int = 800, d: int = 768):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.seq_len = seq_len
        self.d = d

    def forward(self, input_text: list, s_max: float, s_min: float, mu: float, sigma: float):
        """Generate combined embeddings for a document.

        Args:
            input_text: List of sentences in the document.
            s_max: Maximum Gaussian scaling factor.
            s_min: Minimum Gaussian scaling factor.
            mu: Gaussian center position.
            sigma: Gaussian standard deviation.

        Returns:
            Tuple of (embeddings, attention_mask, sep_indices) where:
                - embeddings: (1 x seq_len x d) tensor of combined encodings
                - attention_mask: (1 x seq_len) tensor
                - sep_indices: list of separator token positions
        """
        seq_len = self.seq_len
        d = self.d

        # Token encoding
        T, input_ids, attention_mask = get_token_encoding(
            seq_len, d, input_text, self.tokenizer, max_len=seq_len
        )
        token_ids = torch.tensor(T).unsqueeze(0)
        attention_mask_tensor = torch.tensor(attention_mask).unsqueeze(0)

        # Find separator token indices (RoBERTa uses token ID 2 for </s>)
        sep_indices = [i for i, tid in enumerate(input_ids) if tid == 2]

        # Segment encoding
        S = get_segment_encoding(len(input_ids), d, sep_indices)
        segment_ids = torch.tensor(S).unsqueeze(0)

        # Hybrid positional encoding
        H = get_hybrid_encoding(seq_len, d, s_max, s_min, mu, sigma)
        position_ids = torch.tensor(H).unsqueeze(0)

        # Combined input embedding
        sum_embeddings = token_ids + segment_ids + position_ids

        return sum_embeddings, attention_mask_tensor, sep_indices
