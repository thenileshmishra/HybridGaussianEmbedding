"""
Paired t-test + bootstrap confidence intervals on per-document ROUGE scores.

Usage:
    python src/stats.py --log_dir logs --a G0_roberta-base --b G2_roberta-base
    python src/stats.py --log_dir logs --a G0_roberta-base --b G2_roberta-base --metric r1
"""

import argparse
import json
import os
import numpy as np
from scipy import stats


def bootstrap_ci(deltas, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    boot_means = [
        np.mean(rng.choice(deltas, size=len(deltas), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi


def load_scores(log_dir, tag, metric):
    path = os.path.join(log_dir, f'{tag}_test_scores.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Score file not found: {path}\n'
                                f'Run train.py for {tag} first.')
    with open(path) as f:
        d = json.load(f)
    return np.array(d[metric])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--a',      required=True, help='Baseline tag, e.g. G0_roberta-base')
    parser.add_argument('--b',      required=True, help='Proposed tag, e.g. G2_roberta-base')
    parser.add_argument('--metric', default='rl',
                        choices=['r1', 'r2', 'rl', 'rlsum', 'bertscore', 'meteor'])
    args = parser.parse_args()

    metric_name = {
        'r1': 'ROUGE-1', 'r2': 'ROUGE-2', 'rl': 'ROUGE-L', 'rlsum': 'ROUGE-Lsum',
        'bertscore': 'BERTScore-F1', 'meteor': 'METEOR',
    }[args.metric]
    a = load_scores(args.log_dir, args.a, args.metric)
    b = load_scores(args.log_dir, args.b, args.metric)

    if len(a) != len(b):
        raise ValueError(f'Document count mismatch: {args.a}={len(a)}, {args.b}={len(b)}')

    deltas = b - a
    mean_delta = np.mean(deltas)
    t_stat, p_value = stats.ttest_rel(b, a)
    lo, hi = bootstrap_ci(deltas)

    print(f'\n=== {args.b}  vs  {args.a}  [{metric_name}] ===')
    print(f'  Docs compared  : {len(a)}')
    print(f'  Mean {args.a:30s}: {np.mean(a):.4f}')
    print(f'  Mean {args.b:30s}: {np.mean(b):.4f}')
    print(f'  Mean delta (B − A)    : {mean_delta:+.4f}')
    print(f'  95% Bootstrap CI      : [{lo:+.4f}, {hi:+.4f}]')
    print(f'  Paired t-test         : t={t_stat:.3f},  p={p_value:.4f}')
    sig = 'SIGNIFICANT (p < 0.05)' if p_value < 0.05 else 'not significant (p ≥ 0.05)'
    print(f'  Result                : {sig}')


if __name__ == '__main__':
    main()
