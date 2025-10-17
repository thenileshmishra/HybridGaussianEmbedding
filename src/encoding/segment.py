"""Segment encoding for distinguishing sentence boundaries."""

import numpy as np


def get_segment_encoding(seq_len: int, d: int, sep_indices: list) -> np.ndarray:
    """Generate segment encoding based on sentence separator positions.

    Alternates between 0 and 1 for even and odd sentences respectively,
    using the separator token indices to determine sentence boundaries.

    Args:
        seq_len: Length of the sequence.
        d: Dimension of the encoding.
        sep_indices: List of indices where separator tokens appear.

    Returns:
        A (seq_len x d) segment encoding matrix.
    """
    S = np.zeros((seq_len, d))
    counter = 0
    prev = 0

    for j in sep_indices:
        if counter % 2 == 1:
            for k in range(prev, min(j + 1, seq_len)):
                S[k, :] = 1
        counter += 1
        prev = j + 1

    return S
