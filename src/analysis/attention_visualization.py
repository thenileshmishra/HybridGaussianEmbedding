"""
Attention heatmap visualization.

Provides utilities to:
  - Capture attention weight tensors from a model forward pass.
  - Plot single-head attention heatmaps.
  - Produce side-by-side baseline vs Gaussian-bias comparisons.

All figures are saved as PNG and logged to MLflow automatically.
"""

from pathlib import Path
from typing import List, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def capture_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[np.ndarray]:
    """
    Run a forward pass with output_attentions=True and collect weights.

    Returns:
        List of length num_layers, each element shape (heads, seq_len, seq_len).
        The batch dimension is dropped (uses index 0).
    """
    model.eval()
    with torch.no_grad():
        # Access the underlying HF model regardless of wrapper type
        hf_model = getattr(model, "roberta",
                   getattr(model, "deberta",
                   getattr(model, "longformer", None)))
        if hf_model is None:
            raise AttributeError("Cannot locate inner HuggingFace model on wrapper.")

        outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # outputs.attentions: tuple of (batch, heads, seq, seq)
    return [attn[0].cpu().numpy() for attn in outputs.attentions]


def plot_attention_heatmap(
    layer_weights: np.ndarray,
    layer_idx: int,
    head_idx: int,
    output_path: Path,
    tokens: Optional[List[str]] = None,
    title: str = "",
) -> None:
    """
    Plot a single attention head as a heatmap.

    Args:
        layer_weights:  (heads, seq_len, seq_len) for one layer.
        layer_idx:      Layer index (for the plot title).
        head_idx:       Which attention head to visualise.
        tokens:         Optional list of token strings for axis labels.
    """
    weights = layer_weights[head_idx]   # (seq_len, seq_len)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if tokens:
        n = min(len(tokens), weights.shape[0])
        ticks = range(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tokens[:n], rotation=90, fontsize=6)
        ax.set_yticklabels(tokens[:n], fontsize=6)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"{title}  |  Layer {layer_idx}  Head {head_idx}", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(output_path), artifact_path="analysis/attention")


def compare_attention_heatmaps(
    baseline_weights: List[np.ndarray],
    biased_weights: List[np.ndarray],
    layer_idx: int,
    head_idx: int,
    output_path: Path,
) -> None:
    """
    Side-by-side comparison: baseline (left) vs Gaussian-biased (right).
    """
    base_w  = baseline_weights[layer_idx][head_idx]
    biased_w = biased_weights[layer_idx][head_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, weights, title in zip(axes, [base_w, biased_w], ["Baseline", "Gaussian Bias"]):
        im = ax.imshow(weights, cmap="Blues", aspect="auto")
        ax.set_title(f"{title}  —  Layer {layer_idx}  Head {head_idx}", fontsize=10)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(output_path), artifact_path="analysis/attention")
