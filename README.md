# Hybrid Gaussian Positional Embedding for Extractive Summarization

Thesis research by **Nilesh Mishra**

---

## What this project does

Implements and evaluates a **learnable Gaussian sentence-position bias** for BERTSum-style extractive summarization. The bias is injected at the sentence-level CLS embeddings before the inter-sentence transformer, allowing the model to learn *which region of a document* typically contains the most important sentences.

**Gaussian formula:**

```
G(i) = exp( -(i_norm - μ)² / (2σ²) )
h_i  = h_i + α · G(i)
```

where `i_norm = i / (N-1)` normalizes sentence index to [0, 1], and μ, σ, α are either fixed (G1) or learned end-to-end (G2).

---

## Repository structure

```
src/
  data.py      — CNN/DM and XSum data pipeline, oracle label generation
  model.py     — BERTSumExt with GaussianBias module (G0–G4 variants)
  train.py     — Training loop, evaluation, checkpoint saving
  stats.py     — Paired t-test + bootstrap CI on per-document ROUGE
  analyze.py   — Four analysis figures from experiment logs
requirements.txt
```

---

## Variants

| Variant | What's learned | Description |
|---------|---------------|-------------|
| G0 | nothing | Sinusoidal sentence PE only (baseline) |
| G1 | nothing | Fixed Gaussian scalar (μ=0.5, σ=0.2, α=1.0) |
| G2 | μ, σ, α | Learnable scalar Gaussian — **best variant** |
| G3 | vector w | Fixed μ/σ, learnable per-dimension weight |
| G4 | μ, σ, w | Fully learnable Gaussian with vector weight |

---

## Datasets

| Dataset | Docs sampled | Split | Style |
|---------|-------------|-------|-------|
| CNN/DailyMail | 999 | 640/160/199 | Multi-sentence extractive |
| XSum | 1000 | 640/160/200 | Single-sentence abstractive |

---

## Results (RoBERTa-base, CNN/DM, 3 epochs)

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------|---------|---------|---------|
| G0 | 0.3694 | 0.1659 | 0.2457 |
| G1 | 0.3692 | 0.1660 | 0.2470 |
| **G2** | **0.3715** | **0.1662** | **0.2472** |
| G3 | 0.3682 | 0.1651 | 0.2455 |
| G4 | 0.3686 | 0.1656 | 0.2462 |

G2 learned μ=0.525 (slightly past document midpoint) and σ=0.12 (narrower than init), showing the model learns CNN/DM document structure.

---

## How to run (Google Colab)

```python
# Step 1 — mount Drive and clone
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/thenileshmishra/HybridGaussianEmbedding.git
%cd HybridGaussianEmbedding
!pip install -q rouge-score bert-score datasets nltk sentencepiece

# Step 2 — build data cache
!python src/data.py --dataset cnndm
!python src/data.py --dataset xsum

# Step 3 — train variants
!python src/train.py --backbone roberta-base --variant G0 --epochs 3
!python src/train.py --backbone roberta-base --variant G2 --epochs 3

# Step 4 — statistical significance
!python src/stats.py --a G0_roberta-base --b G2_roberta-base

# Step 5 — generate figures
!python src/analyze.py --log_dir /content/drive/MyDrive/GaussianBERTSum/logs
```

---

## Key finding

G2's learnable Gaussian provides consistent improvement on RoBERTa (absolute PE). It does **not** help on DeBERTa, which already encodes relative position inside every attention head — confirming the Gaussian fills a gap rather than adding noise when backbone PE is limited.
