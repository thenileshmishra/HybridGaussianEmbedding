# Product Requirements Document (PRD)

## Title
Modular Positional Encoding Framework with Learnable Gaussian Relative Bias for Extractive Summarization

## Status
- Owner: Nilesh Mishra
- Date: 2026-02-09
- Status: Draft

## Problem Statement
The current prototype is a single RoBERTa notebook with hybrid Gaussian + sinusoidal positional encoding. This limits:
- extension to other positional encoding paradigms,
- attention-level bias experiments,
- reproducibility needed for thesis-level research.

Positional encoding differs across model families (absolute learned, relative bias, disentangled). The system must support model-specific behavior while enabling controlled comparisons.

## Research Objective
Evaluate whether learnable Gaussian relative positional bias injected into attention improves extractive summarization by modeling locality better than standard absolute or relative encodings.

## Scope
### In Scope
- Modular refactor of the notebook into a reusable codebase.
- Attention-level Gaussian relative bias, including learnable per-head sigma.
- Sentence-aware Gaussian bias for extractive tasks.
- Controlled experiments across selected encoder models and datasets.

### Out of Scope
- Decoder-only models (GPT-2, LLaMA).
- Abstractive summarization pipelines.
- Large-scale hyperparameter sweeps.

## Model and Dataset Selection
### Models
Primary target and baselines:
- RoBERTa (learned absolute positional embeddings).
- DeBERTa (disentangled relative positional encoding).
- Longformer (sparse attention for long documents).

Rationale for removals:
- GPT-2 and LLaMA are decoder-only and introduce generative biases not aligned with extractive summarization.

### Datasets
- CNN/DailyMail (medium-long news).
- Gigaword (short headline-style).
- XSum (highly abstractive, weak lead bias).

## Positional Encoding Variants
### Baselines
- model_native
- sinusoidal_absolute
- gaussian_absolute (embedding-level)

### Core Contribution: Gaussian Relative Bias
Attention is modified instead of token embeddings:

Attention(Q, K, V) = Softmax((QK^T / sqrt(d)) + G(i - j)) V

G(i - j) = - (i - j)^2 / (2 * sigma^2)

### Learnable Multi-Head Gaussian Bias
- sigma becomes per-head and learnable: sigma -> sigma_h
- Enables different locality scales across heads.

### Sentence-Aware Gaussian Bias
- Uses sentence index distance: G(s_i - s_j)
- Aligns bias with extractive summarization granularity.

## Functional Requirements
### Experiment Runner
- model: roberta, deberta, longformer
- encoding:
  - model_native
  - sinusoidal_absolute
  - gaussian_absolute
  - gaussian_relative
  - gaussian_relative_learnable
  - gaussian_sentence_aware
- dataset: cnndm, gigaword, xsum
- Reproducible runs with fixed seed and config snapshots.

### Adapter Contract
Adapters must support:
- embedding-level positional replacement,
- attention-level bias injection,
- access to attention scores before softmax,
- sentence-level pooling.

Required method:
- inject_attention_bias(attention_scores, sentence_map=None)

## Experiment Matrix
| Model | model_native | sinusoidal | gaussian_abs | gaussian_rel | learnable_sigma | sentence_aware |
|---|---|---|---|---|---|---|
| RoBERTa | yes | yes | yes | yes | yes | yes |
| DeBERTa | yes | no | no | yes | yes | yes |
| Longformer | yes | no | no | yes | yes | yes |

## Evaluation
### Metrics
- ROUGE-1/2/L/Lsum
- BERTScore
- Lead-bias comparison

### Analysis
- Learned sigma distribution by head
- Attention heatmaps (baseline vs Gaussian)
- Performance vs document length
- Dataset-wise optimal sigma
- Statistical significance testing

## Risks and Mitigations
- Risk: Gaussian bias conflicts with DeBERTa disentangled attention.
  - Mitigation: Inject bias additively without altering content-position separation.
- Risk: Sentence-aware bias increases compute.
  - Mitigation: Precompute sentence index matrix.

## Milestones
1. Modular refactor (RoBERTa baseline).
2. Implement Gaussian relative bias injection.
3. Add learnable per-head sigma.
4. Implement sentence-aware Gaussian bias.
5. Add DeBERTa baseline.
6. Add Longformer baseline.
7. Run full dataset matrix.
8. Perform ablations and analysis.
9. Draft thesis chapter.

## Open Questions
- Should Gaussian bias replace or augment native relative bias?
- Share sigma across layers or per-layer?
- Combine sentence-aware and token-level bias?
- Does locality strength correlate with document length?


modular-pe/
│
├── configs/
│   ├── model/
│   │   ├── roberta.yaml
│   │   ├── deberta.yaml
│   │   └── longformer.yaml
│   │
│   ├── encoding/
│   │   ├── model_native.yaml
│   │   ├── gaussian_relative.yaml
│   │   ├── gaussian_learnable.yaml
│   │   └── gaussian_sentence.yaml
│   │
│   └── dataset/
│       ├── cnndm.yaml
│       ├── gigaword.yaml
│       └── xsum.yaml
│
├── src/
│   ├── models/
│   │   ├── roberta_adapter.py
│   │   ├── deberta_adapter.py
│   │   └── longformer_adapter.py
│   │
│   ├── positional_bias/
│   │   ├── gaussian_relative.py
│   │   ├── gaussian_learnable.py
│   │   └── sentence_aware.py
│   │
│   ├── runner/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── experiment.py
│   │
│   └── analysis/
│       ├── sigma_analysis.py
│       ├── attention_visualization.py
│       └── stats_tests.py
│
├── logs/
├── outputs/
├── notebooks/
├── requirements.txt
└── README.md
