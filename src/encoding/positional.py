"""Positional encoding implementations: sinusoidal, Gaussian, and hybrid."""

import numpy as np
import torch


def get_position_encoding(seq_len: int, d: int, n: int = 10000) -> np.ndarray:
    """Generate sinusoidal positional encoding.

    Uses the standard sine/cosine formula from 'Attention Is All You Need'.

    Args:
        seq_len: Length of the sequence (number of positions).
        d: Dimension of the encoding.
        n: Base for the frequency computation.

    Returns:
        A (seq_len x d) positional encoding matrix.
    """
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


def get_gaussian_encoding(
    seq_len: int,
    d: int,
    s_max: float,
    s_min: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Generate Gaussian positional encoding.

    Applies a Gaussian distribution centered at mu with spread sigma,
    using linear scaling across dimensions to balance local and global
    positional information.

    Args:
        seq_len: Length of the sequence.
        d: Dimension of the encoding.
        s_max: Maximum scaling factor for Gaussian spread.
        s_min: Minimum scaling factor for Gaussian spread.
        mu: Center position for the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        A (seq_len x d) Gaussian positional encoding matrix.
    """
    G = np.zeros((seq_len, d))
    if mu is None:
        mu = np.linspace(0, seq_len - 1, d)

    start = max(0, int(mu - sigma))
    end = min(seq_len, int(mu + sigma))

    # Linear scaling ensures balanced local-global representation
    dim_scaling = torch.linspace(s_min, s_max, d // 2)

    for k in range(start, end):
        for i, scale in enumerate(dim_scaling):
            gaussian_val = np.exp(-((k - mu) ** 2) / (2 * (sigma * scale) ** 2))
            G[k, 2 * i] = gaussian_val
            G[k, 2 * i + 1] = gaussian_val

    return G


def get_hybrid_encoding(
    seq_len: int,
    d: int,
    s_max: float,
    s_min: float,
    mu: float,
    sigma: float,
    n: int = 10000,
) -> np.ndarray:
    """Generate hybrid positional encoding combining sinusoidal and Gaussian.

    The hybrid encoding sums sinusoidal and Gaussian components. Since
    sinusoidal encoding has unique values for even and odd dimensions,
    adding Gaussian matching pairs preserves uniqueness in the final
    hybrid encoding.

    Args:
        seq_len: Length of the sequence.
        d: Dimension of the encoding.
        s_max: Maximum Gaussian scaling factor.
        s_min: Minimum Gaussian scaling factor.
        mu: Gaussian center position.
        sigma: Gaussian standard deviation.
        n: Base for sinusoidal encoding.

    Returns:
        A (seq_len x d) hybrid positional encoding matrix.
    """
    sin_encoding = get_position_encoding(seq_len, d, n)
    gauss_encoding = get_gaussian_encoding(seq_len, d, s_max, s_min, mu, sigma)
    return sin_encoding + gauss_encoding
