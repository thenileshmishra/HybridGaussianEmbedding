"""Configuration module for model hyperparameters and training settings."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Central configuration for the extractive summarization pipeline."""

    # Model architecture
    model_name: str = "roberta-base"
    d_model: int = 768
    seq_len: int = 800
    max_position_embeddings: int = 1024
    num_inter_layers: int = 2
    num_heads: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-7

    # Positional encoding
    encoding_type: str = "hybrid"  # Options: sinusoidal, gaussian, hybrid
    n_base: int = 10000

    # Gaussian encoding hyperparameters
    s_max: float = 2.0
    s_min: float = 0.3
    mu: float = 260.0
    sigma: float = 65.0

    # Training
    learning_rate: float = 2e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    num_epochs: int = 2
    batch_size: int = 8
    max_grad_norm: float = 1.0
    scheduler_step_size: int = 2
    scheduler_gamma: float = 0.9

    # Data
    sample_size: int = 1000
    test_split: float = 0.2
    val_split: float = 0.2
    random_seed: int = 42

    # Checkpointing
    checkpoint_interval: int = 25  # Save every N documents
    checkpoint_dir: str = "checkpoints"
    metrics_dir: str = "metrics"

    # Device
    device: str = "cuda"


def get_default_config(**overrides) -> ModelConfig:
    """Create a ModelConfig with optional overrides."""
    return ModelConfig(**overrides)
