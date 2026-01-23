"""
Sigma distribution analysis for learned Gaussian bias adapters.

After training, the per-head sigma values reveal which locality scale
each attention head has learned. This module extracts, plots, and logs
those distributions to MLflow.
"""

from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import torch.nn as nn

# Use non-interactive backend so this works headlessly on clusters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_sigma_values(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Walk the model and collect sigma values from every module that has
    a `log_sigma` parameter (GaussianLearnableBias, SentenceAwareGaussianBias).

    Returns:
        dict mapping dotted module path → sigma array of shape (num_heads,)
    """
    sigma_dict: Dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if hasattr(module, "log_sigma"):
            sigma_dict[name] = module.sigma.detach().cpu().numpy()
    return sigma_dict


def plot_sigma_distribution(
    sigma_dict: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Learned σ per Head",
) -> None:
    """
    Bar chart: one subplot per layer showing sigma per attention head.
    Saves to output_path and logs as an MLflow artifact.
    """
    if not sigma_dict:
        return

    n = len(sigma_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), squeeze=False)

    for ax, (layer_name, sigmas) in zip(axes[0], sigma_dict.items()):
        short_name = ".".join(layer_name.split(".")[-3:-1]) or layer_name
        ax.bar(range(len(sigmas)), sigmas, color="steelblue", edgecolor="white")
        ax.set_title(short_name, fontsize=9)
        ax.set_xlabel("Head index")
        ax.set_ylabel("σ")
        ax.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(output_path), artifact_path="analysis")


def log_sigma_stats(model: nn.Module, step: int = 0) -> None:
    """
    Log aggregate sigma statistics (mean/std/min/max) to MLflow.
    Safe to call even if the model has no learnable sigma.
    """
    sigma_dict = extract_sigma_values(model)
    if not sigma_dict:
        return

    all_vals = np.concatenate(list(sigma_dict.values()))
    mlflow.log_metric("sigma_mean", float(np.mean(all_vals)),  step=step)
    mlflow.log_metric("sigma_std",  float(np.std(all_vals)),   step=step)
    mlflow.log_metric("sigma_min",  float(np.min(all_vals)),   step=step)
    mlflow.log_metric("sigma_max",  float(np.max(all_vals)),   step=step)

    print(
        f"  [σ stats] mean={np.mean(all_vals):.3f}  "
        f"std={np.std(all_vals):.3f}  "
        f"range=[{np.min(all_vals):.3f}, {np.max(all_vals):.3f}]"
    )
