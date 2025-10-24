#!/bin/bash
# Run extractive summarization experiment with hybrid Gaussian positional encoding.
#
# Usage: bash scripts/run_experiment.sh

set -e

echo "=== Hybrid Gaussian Positional Encoding Experiment ==="
echo ""

# Default parameters
EPOCHS=2
BATCH_SIZE=8
S_MAX=2.0
S_MIN=0.3
MU=260
SIGMA=65
SEED=42

echo "Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  s_max: $S_MAX, s_min: $S_MIN"
echo "  mu: $MU, sigma: $SIGMA"
echo "  Seed: $SEED"
echo ""

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run training and evaluation
python main.py \
    --mode both \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --s_max $S_MAX \
    --s_min $S_MIN \
    --mu $MU \
    --sigma $SIGMA \
    --seed $SEED

echo ""
echo "=== Experiment complete ==="
