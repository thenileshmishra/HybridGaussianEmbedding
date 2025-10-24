"""Unit tests for positional encoding modules."""

import numpy as np
import pytest

from src.encoding.positional import (
    get_position_encoding,
    get_gaussian_encoding,
    get_hybrid_encoding,
)
from src.encoding.segment import get_segment_encoding


class TestPositionEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self):
        P = get_position_encoding(seq_len=100, d=64)
        assert P.shape == (100, 64)

    def test_zero_position(self):
        P = get_position_encoding(seq_len=10, d=64)
        # sin(0) = 0 for all even positions at k=0
        assert P[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_values_bounded(self):
        P = get_position_encoding(seq_len=100, d=64)
        assert np.all(P >= -1.0)
        assert np.all(P <= 1.0)

    def test_unique_positions(self):
        P = get_position_encoding(seq_len=50, d=64)
        # Each position should have a unique encoding
        for i in range(50):
            for j in range(i + 1, 50):
                assert not np.allclose(P[i], P[j])


class TestGaussianEncoding:
    """Tests for Gaussian positional encoding."""

    def test_output_shape(self):
        G = get_gaussian_encoding(seq_len=100, d=64, s_max=2.0, s_min=0.3, mu=50.0, sigma=20.0)
        assert G.shape == (100, 64)

    def test_peak_at_center(self):
        G = get_gaussian_encoding(seq_len=200, d=64, s_max=2.0, s_min=0.3, mu=100.0, sigma=30.0)
        # Value at mu should be highest
        center_val = np.sum(np.abs(G[100, :]))
        edge_val = np.sum(np.abs(G[0, :]))
        assert center_val >= edge_val

    def test_non_negative(self):
        G = get_gaussian_encoding(seq_len=100, d=64, s_max=2.0, s_min=0.3, mu=50.0, sigma=20.0)
        assert np.all(G >= 0)


class TestHybridEncoding:
    """Tests for hybrid positional encoding."""

    def test_output_shape(self):
        H = get_hybrid_encoding(seq_len=100, d=64, s_max=2.0, s_min=0.3, mu=50.0, sigma=20.0)
        assert H.shape == (100, 64)

    def test_hybrid_is_sum(self):
        P = get_position_encoding(100, 64)
        G = get_gaussian_encoding(100, 64, 2.0, 0.3, 50.0, 20.0)
        H = get_hybrid_encoding(100, 64, 2.0, 0.3, 50.0, 20.0)
        np.testing.assert_array_almost_equal(H, P + G)


class TestSegmentEncoding:
    """Tests for segment encoding."""

    def test_output_shape(self):
        S = get_segment_encoding(seq_len=100, d=64, sep_indices=[20, 50, 80])
        assert S.shape == (100, 64)

    def test_alternating_segments(self):
        S = get_segment_encoding(seq_len=100, d=4, sep_indices=[30, 60, 90])
        # First segment (0-30) should be 0s
        assert np.all(S[0:30, :] == 0)
        # Second segment (31-60) should be 1s
        assert np.all(S[31:60, :] == 1)

    def test_empty_indices(self):
        S = get_segment_encoding(seq_len=50, d=32, sep_indices=[])
        assert S.shape == (50, 32)
        assert np.all(S == 0)
