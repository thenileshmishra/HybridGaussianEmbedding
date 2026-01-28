# Modular Positional Encoding Framework
### Learnable Gaussian Relative Bias for Extractive Summarization

> Thesis research by **Nilesh Mishra**

---

## Research Objective

Evaluate whether a **learnable Gaussian relative positional bias** injected
directly into the attention mechanism improves extractive summarization by
modelling locality more effectively than standard absolute or relative encodings.

Core hypothesis:

```
Attention(Q, K, V) = Softmax((QK^T / √d) + G(i − j)) V

G(i − j) = −(i − j)² / (2 σ²)          # fixed σ variant
G_h(i − j) = −(i − j)² / (2 σ_h²)      # learnable per-head σ
```

---

## Directory Structure

```
.
├── configs/
│   ├── model/          roberta.yaml · deberta.yaml · longformer.yaml
│   ├── encoding/       model_native · gaussian_relative · gaussian_learnable · gaussian_sentence
│   └── dataset/        cnndm.yaml · gigaword.yaml · xsum.yaml
│
├── src/
│   ├── models/
│   │   ├── base_adapter.py          ← PositionalBiasAdapter ABC
│   │   ├── roberta_adapter.py       ← BiasSelfAttention + RoBERTaWithBias
│   │   ├── deberta_adapter.py       ← BiasedDisentangledSelfAttention + DeBERTaWithBias
│   │   └── longformer_adapter.py    ← BiasedLongformerSelfAttention + LongformerWithBias
│   │
│   ├── positional_bias/
│   │   ├── gaussian_relative.py     ← fixed σ
│   │   ├── gaussian_learnable.py    ← learnable per-head σ
│   │   └── sentence_aware.py        ← sentence-index distance bias
│   │
│   ├── runner/
│   │   ├── experiment.py            ← CLI entry-point + MLflow orchestration
│   │   ├── train.py                 ← training loop + checkpointing
│   │   └── evaluate.py              ← ROUGE + BERTScore evaluation
│   │
│   └── analysis/
│       ├── sigma_analysis.py        ← learned σ distribution plots
│       ├── attention_visualization.py ← attention heatmaps
│       └── stats_tests.py           ← paired t-test + bootstrap CI
│
├── logs/               (MLflow local store)
├── outputs/            (per-run artefacts)
├── notebooks/          (exploration notebooks)
├── run_experiment.sh   ← example sweep commands
└── requirements.txt
```

---

## Encoding Variants

| Name | Type | Description |
|------|------|-------------|
| `model_native` | — | Unmodified model-native encoding |
| `gaussian_relative` | attention bias | Fixed σ Gaussian relative bias |
| `gaussian_relative_learnable` | attention bias | Per-head learnable σ |
| `gaussian_sentence_aware` | attention bias | Learnable σ over sentence-index distance |

---

## Quick Start

```bash
pip install -r requirements.txt

# Single run
python -m src.runner.experiment \
    --model   roberta \
    --encoding gaussian_relative_learnable \
    --dataset  cnndm \
    --seed     42

# Full sweep
bash run_experiment.sh

# View MLflow dashboard
mlflow ui --port 5000
```

---

## Experiment Matrix

| Model | model_native | gaussian_rel | learnable_σ | sentence_aware |
|-------|:---:|:---:|:---:|:---:|
| RoBERTa   | ✓ | ✓ | ✓ | ✓ |
| DeBERTa   | ✓ | ✓ | ✓ | ✓ |
| Longformer| ✓ | ✓ | ✓ | ✓ |

Datasets: **CNN/DM** · **Gigaword** · **XSum**

---

## Adapter Contract

Every encoding variant must subclass `PositionalBiasAdapter` and implement:

```python
def inject_attention_bias(
    self,
    attention_scores: Tensor,   # (batch, heads, seq, seq)
    sentence_map: Tensor | None # (batch, seq)  — sentence indices
) -> Tensor:
    ...
```

The bias is injected **before softmax** inside the attention forward of each
wrapped model (RoBERTa → `BiasSelfAttention`, DeBERTa → `BiasedDisentangledSelfAttention`).

---

## Reproducibility

Every run:
1. Calls `set_seed(args.seed)` (Python · NumPy · PyTorch · CUDA).
2. Saves a full `config_snapshot.yaml` as an MLflow artefact.
3. Logs all hyperparameters under `mlflow.log_params`.
