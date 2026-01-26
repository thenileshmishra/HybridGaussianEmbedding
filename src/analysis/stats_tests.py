"""
Statistical significance testing for metric comparisons.

Implements:
  - Paired t-test  (paired_ttest)
  - Bootstrap confidence intervals  (bootstrap_ci)
  - Multi-system comparison summary  (compare_runs)

Results are printed and logged to the active MLflow run.
"""

from typing import List, Optional, Tuple

import mlflow
import numpy as np
from scipy import stats


def paired_ttest(
    baseline_scores: List[float],
    model_scores: List[float],
    metric_name: str = "rouge1",
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Two-sided paired t-test between baseline and proposed model.

    Args:
        baseline_scores: Per-document scores for the baseline system.
        model_scores:    Per-document scores for the proposed system.
        metric_name:     Name used for MLflow metric keys.
        alpha:           Significance level.

    Returns:
        (t_statistic, p_value, is_significant)
    """
    t_stat, p_value = stats.ttest_rel(baseline_scores, model_scores)
    is_sig = p_value < alpha

    mlflow.log_metric(f"{metric_name}_t_stat", float(t_stat))
    mlflow.log_metric(f"{metric_name}_p_value", float(p_value))

    direction = "↑" if np.mean(model_scores) > np.mean(baseline_scores) else "↓"
    sig_label = f"p={p_value:.4f} ({'significant' if is_sig else 'not significant'})"
    print(
        f"  [{metric_name}] t={t_stat:+.4f}  {sig_label}  "
        f"Δmean={np.mean(model_scores) - np.mean(baseline_scores):+.4f} {direction}"
    )
    return float(t_stat), float(p_value), is_sig


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for a metric mean.

    Returns:
        (lower_bound, upper_bound) at the requested confidence level.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores)
    boot_means = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def compare_runs(
    run_scores: dict,
    metric_name: str = "rouge1",
    baseline_key: str = "model_native",
    alpha: float = 0.05,
) -> None:
    """
    Compare all systems against a designated baseline using paired t-tests.

    Args:
        run_scores:    dict mapping run_name → list[float] of per-doc scores.
        metric_name:   Metric label for logging.
        baseline_key:  Key in run_scores for the baseline system.
    """
    if baseline_key not in run_scores:
        raise KeyError(f"Baseline key {baseline_key!r} not found in run_scores.")

    baseline = run_scores[baseline_key]
    print(f"\n{'─'*55}")
    print(f"  Statistical comparison on {metric_name}  (baseline: {baseline_key})")
    print(f"{'─'*55}")
    for name, scores in run_scores.items():
        if name == baseline_key:
            continue
        paired_ttest(baseline, scores, metric_name=f"{name}_vs_baseline_{metric_name}", alpha=alpha)
    print(f"{'─'*55}\n")
