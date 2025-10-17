"""Token encoding for converting input text to token ID matrices."""

import numpy as np


def get_token_encoding(seq_len: int, d: int, input_text: list, tokenizer, max_len: int = 800):
    """Obtain token embeddings for input text.

    Tokenizes each sentence in the input, concatenates token IDs,
    and creates a (seq_len x d) token encoding matrix.

    Args:
        seq_len: Target sequence length.
        d: Dimension of the encoding.
        input_text: List of sentences to encode.
        tokenizer: HuggingFace tokenizer instance.
        max_len: Maximum number of tokens.

    Returns:
        Tuple of (token_matrix, input_ids, attention_mask).
    """
    T = np.zeros((seq_len, d))
    input_ids = []
    attention_mask = []

    for sentence in input_text:
        encoded = tokenizer.encode_plus(sentence, return_attention_mask=True)
        input_ids.extend(encoded["input_ids"])
        attention_mask.extend(encoded["attention_mask"])

    # Pad or truncate to max_len
    input_ids = (input_ids + [0] * max(0, max_len - len(input_ids)))[:max_len]
    attention_mask = (attention_mask + [0] * max(0, max_len - len(attention_mask)))[:max_len]

    # Fill the token matrix
    for i in range(min(seq_len, len(input_ids))):
        T[i, :] = input_ids[i]

    return T, input_ids, attention_mask
