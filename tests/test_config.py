"""Unit tests for the configuration module."""

from src.config.model_config import ModelConfig, get_default_config


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.d_model == 768
        assert config.seq_len == 800
        assert config.num_heads == 6
        assert config.dropout == 0.1

    def test_custom_overrides(self):
        config = get_default_config(num_epochs=10, learning_rate=1e-4)
        assert config.num_epochs == 10
        assert config.learning_rate == 1e-4
        # Other defaults preserved
        assert config.d_model == 768

    def test_gaussian_params(self):
        config = ModelConfig(s_max=3.0, s_min=0.5, mu=300.0, sigma=80.0)
        assert config.s_max == 3.0
        assert config.s_min == 0.5
        assert config.mu == 300.0
        assert config.sigma == 80.0
