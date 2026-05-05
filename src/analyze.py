"""
Generate analysis figures from experiment logs.

Produces four PNG files in the log directory:
    g2_trajectory.png      — G2 mu/sigma/alpha across epochs
    rouge_bar.png          — ROUGE test scores all variants (RoBERTa, CNN/DM)
    gaussian_shapes.png    — Gaussian curve shape for G1 vs G2 epochs
    backbone_heatmap.png   — G2 vs G0 ROUGE-L delta per backbone

Usage:
    python src/analyze.py
    python src/analyze.py --log_dir /content/drive/MyDrive/GaussianBERTSum/logs
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_test_row(log_dir, tag):
    path = os.path.join(log_dir, f'{tag}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df['epoch'].astype(str) == 'test']
    return row.iloc[0] if not row.empty else None


def plot_g2_trajectory(log_dir, out_dir):
    path = os.path.join(log_dir, 'G2_roberta-base.csv')
    if not os.path.exists(path):
        print(f'  skip trajectory: {path} not found')
        return
    df = pd.read_csv(path)
    df = df[df['epoch'].astype(str).str.isdigit()].copy()
    df['epoch'] = df['epoch'].astype(int)
    df = df[df['gauss_mu'].astype(str) != '']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['epoch'], df['gauss_mu'].astype(float),              'o-',  label='μ (center)')
    ax.plot(df['epoch'], df['gauss_sigma'].astype(float),           's--', label='σ (spread)')
    ax.plot(df['epoch'], df['gauss_alpha_or_wnorm'].astype(float),  '^:',  label='α (scale)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4, label='μ init=0.5')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Parameter value')
    ax.set_title('G2 Gaussian Parameter Trajectory\n(RoBERTa-base, CNN/DM)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, 'g2_trajectory.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


def plot_rouge_bar(log_dir, out_dir):
    variants = ['G0', 'G1', 'G2', 'G3', 'G4']
    found, r1s, r2s, rls = [], [], [], []
    for v in variants:
        row = _load_test_row(log_dir, f'{v}_roberta-base')
        if row is None:
            continue
        found.append(v)
        r1s.append(float(row['val_r1']))
        r2s.append(float(row['val_r2']))
        rls.append(float(row['val_rl']))

    if not found:
        print('  skip bar chart: no test rows found')
        return

    x, w = np.arange(len(found)), 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, r1s, w, label='ROUGE-1')
    ax.bar(x,     r2s, w, label='ROUGE-2')
    ax.bar(x + w, rls, w, label='ROUGE-L')
    ax.set_xticks(x)
    ax.set_xticklabels(found)
    ax.set_ylabel('F1 Score')
    ax.set_title('Test ROUGE — All Variants\n(RoBERTa-base, CNN/DM)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = os.path.join(out_dir, 'rouge_bar.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


def plot_gaussian_shapes(out_dir):
    positions = np.linspace(0, 1, 200)
    configs = [
        ('G1 fixed  (μ=0.50, σ=0.20)', 0.50, 0.20, 'steelblue',  '-'),
        ('G2 epoch1 (μ=0.51, σ=0.15)', 0.51, 0.15, 'darkorange', '--'),
        ('G2 epoch3 (μ=0.53, σ=0.12)', 0.53, 0.12, 'crimson',    ':'),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, mu, sigma, color, ls in configs:
        g = np.exp(-((positions - mu) ** 2) / (2 * sigma ** 2))
        ax.plot(positions, g, color=color, linestyle=ls, linewidth=2, label=label)
    ax.set_xlabel('Normalized sentence position  (0 = start, 1 = end)')
    ax.set_ylabel('Gaussian weight  G(i)')
    ax.set_title('Gaussian Shape: G1 Fixed vs G2 Learned')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, 'gaussian_shapes.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


def plot_backbone_heatmap(log_dir, out_dir):
    backbones = [
        ('roberta-base',              'RoBERTa'),
        ('distilbert-base-uncased',   'DistilBERT'),
        ('albert-base-v2',            'ALBERT'),
        ('microsoft_deberta-base',    'DeBERTa'),
    ]
    metric_keys  = ['val_r1', 'val_r2', 'val_rl']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

    deltas, labels = [], []
    for bb, name in backbones:
        g0 = _load_test_row(log_dir, f'G0_{bb}')
        g2 = _load_test_row(log_dir, f'G2_{bb}')
        if g0 is None or g2 is None:
            continue
        deltas.append([float(g2[m]) - float(g0[m]) for m in metric_keys])
        labels.append(name)

    if not deltas:
        print('  skip heatmap: insufficient data')
        return

    data = np.array(deltas)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.015, vmax=0.015)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(metric_names)):
            ax.text(j, i, f'{data[i, j]:+.4f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, label='Δ ROUGE  (G2 − G0)')
    ax.set_title('G2 vs G0 ROUGE Delta by Backbone')
    fig.tight_layout()
    out = os.path.join(out_dir, 'backbone_heatmap.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    print(f'Reading logs from: {args.log_dir}')

    plot_g2_trajectory(args.log_dir, args.log_dir)
    plot_rouge_bar(args.log_dir, args.log_dir)
    plot_gaussian_shapes(args.log_dir)
    plot_backbone_heatmap(args.log_dir, args.log_dir)

    print('Done. All figures saved to:', args.log_dir)


if __name__ == '__main__':
    main()
