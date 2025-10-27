# Hybrid Gaussian Positional Encoding for Extractive Summarization

A modular framework for evaluating hybrid positional encoding strategies in transformer-based extractive summarization. Combines sinusoidal and Gaussian positional encodings to capture both global position and local context signals within long documents.

## Architecture

The pipeline implements an extractive summarization system with:

1. **Hybrid Positional Encoding** - Combines standard sinusoidal encoding with a Gaussian component centered at a configurable position (mu), with learnable spread (sigma). This biases the model's attention toward content-relevant regions.

2. **RoBERTa Document Encoder** - Processes tokenized documents with custom input embeddings (token + hybrid positional + segment encodings).

3. **Inter-Sentence Transformer** - Applies additional transformer layers over CLS token representations for sentence-level scoring.

4. **Extractive Summary Generator** - Selects top-k sentences based on learned scores.

## Project Structure

```
HybridGaussianEmbedding/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── src/
│   ├── config/
│   │   └── model_config.py     # Hyperparameter configuration
│   ├── encoding/
│   │   ├── positional.py       # Sinusoidal, Gaussian, Hybrid encodings
│   │   ├── segment.py          # Segment encoding
│   │   └── token.py            # Token encoding
│   ├── models/
│   │   ├── embedder.py         # RoBERTa document embedder
│   │   ├── roberta_encoder.py  # Custom RoBERTa encoder
│   │   ├── transformer_layers.py # Inter-sentence transformer
│   │   ├── classifier.py       # Sentence classifier
│   │   └── summary_generator.py # Extractive summary selection
│   ├── data/
│   │   ├── loader.py           # CNN/DailyMail data loading
│   │   ├── preprocessing.py    # Text preprocessing
│   │   ├── sampling.py         # Stratified sampling
│   │   ├── dataset.py          # PyTorch dataset
│   │   └── extractive.py       # Gold extractive summary creation
│   ├── evaluation/
│   │   ├── rouge.py            # ROUGE metrics
│   │   └── bertscore.py        # BERTScore metrics
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── validator.py        # Validation loop
│   │   └── tester.py           # Test evaluation
│   └── utils/
│       ├── device.py           # Device management
│       ├── memory.py           # CPU memory monitoring
│       ├── gpu.py              # GPU monitoring and logging
│       └── batch.py            # Batch processing utilities
├── tests/
│   ├── test_encoding.py        # Encoding unit tests
│   ├── test_config.py          # Config unit tests
│   └── test_data.py            # Data processing tests
└── scripts/
    └── run_experiment.sh       # Experiment runner
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Training and Evaluation

```bash
# Full training + testing pipeline
python main.py --mode both --epochs 2 --batch_size 8

# Training only
python main.py --mode train --epochs 5 --s_max 2.0 --s_min 0.3 --mu 260 --sigma 65

# Testing with a saved checkpoint
python main.py --mode test --checkpoint_dir checkpoints

# With preprocessed data
python main.py --data_path path/to/preprocessed.pt
```

### Gaussian Encoding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s_max`   | 2.0     | Maximum scaling factor for Gaussian spread |
| `s_min`   | 0.3     | Minimum scaling factor for Gaussian spread |
| `mu`      | 260     | Center position for the Gaussian distribution |
| `sigma`   | 65      | Standard deviation of the Gaussian distribution |

### Running Tests

```bash
pytest tests/ -v
```

## Evaluation Metrics

- **ROUGE-1/2/L/Lsum** - N-gram overlap metrics
- **BERTScore** - Contextual embedding similarity

## Dataset

Evaluated on CNN/DailyMail with stratified sampling (validated by Kolmogorov-Smirnov test) to preserve the document length distribution of the full corpus.

## Citation

If you use this work, please cite:

```bibtex
@article{hybrid_gaussian_pos_encoding,
  title={Hybrid Gaussian Positional Encoding for Extractive Summarization},
  author={Mishra, Nilesh},
  year={2025}
}
```
