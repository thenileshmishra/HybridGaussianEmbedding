#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Example experiment execution commands for the Modular PE framework.
#
# Usage:
#   bash run_experiment.sh
#
# Or run a single variant directly:
#   python -m src.runner.experiment --model roberta \
#          --encoding gaussian_relative_learnable    \
#          --dataset cnndm --seed 42
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SEED=42
EPOCHS=3
BATCH=16
LR=2e-5

# ── RoBERTa sweep ─────────────────────────────────────────────────────────────
for ENCODING in model_native gaussian_relative gaussian_relative_learnable gaussian_sentence_aware; do
    echo "==> RoBERTa | ${ENCODING} | cnndm | seed ${SEED}"
    python -m src.runner.experiment \
        --model     roberta    \
        --encoding  "${ENCODING}" \
        --dataset   cnndm      \
        --seed      "${SEED}"  \
        --epochs    "${EPOCHS}" \
        --batch_size "${BATCH}" \
        --lr        "${LR}"
done

# ── DeBERTa baselines ────────────────────────────────────────────────────────
for ENCODING in model_native gaussian_relative gaussian_relative_learnable; do
    echo "==> DeBERTa | ${ENCODING} | cnndm | seed ${SEED}"
    python -m src.runner.experiment \
        --model     deberta    \
        --encoding  "${ENCODING}" \
        --dataset   cnndm      \
        --seed      "${SEED}"  \
        --epochs    "${EPOCHS}" \
        --batch_size "${BATCH}" \
        --lr        "${LR}"
done

# ── Cross-dataset: best encoding on Gigaword and XSum ───────────────────────
for DATASET in gigaword xsum; do
    echo "==> RoBERTa | gaussian_relative_learnable | ${DATASET} | seed ${SEED}"
    python -m src.runner.experiment \
        --model     roberta                     \
        --encoding  gaussian_relative_learnable \
        --dataset   "${DATASET}"               \
        --seed      "${SEED}"                  \
        --epochs    "${EPOCHS}"                 \
        --batch_size "${BATCH}"                 \
        --lr        "${LR}"
done

echo "All experiments complete.  Results in: outputs/"
