"""
Experiment runner for the Modular Positional Encoding framework.

CLI usage:
    python -m src.runner.experiment \\
        --model   roberta \\
        --encoding gaussian_relative_learnable \\
        --dataset  cnndm \\
        --seed     42

The runner:
  1. Loads YAML configs and merges them with OmegaConf.
  2. Sets the global random seed for reproducibility.
  3. Saves a full config snapshot as an MLflow artifact.
  4. Builds the adapter and the model wrapper.
  5. Runs training and evaluation.
  6. Logs all params, metrics, and analysis artifacts to MLflow.
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Type

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from src.models.base_adapter import PositionalBiasAdapter
from src.runner.train import run_training
from src.runner.evaluate import run_evaluation

CONFIG_ROOT = Path("configs")

VALID_MODELS    = ["roberta", "deberta", "longformer"]
VALID_ENCODINGS = [
    "model_native",
    "sinusoidal_absolute",
    "gaussian_absolute",
    "gaussian_relative",
    "gaussian_relative_learnable",
    "gaussian_sentence_aware",
]
VALID_DATASETS  = ["cnndm", "gigaword", "xsum"]


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(model: str, encoding: str, dataset: str) -> DictConfig:
    """Merge model / encoding / dataset YAML configs into a single DictConfig."""
    # Map CLI encoding name to config file name
    enc_file_map = {
        "gaussian_relative_learnable": "gaussian_learnable",
        "gaussian_sentence_aware":     "gaussian_sentence",
        "model_native":                "model_native",
        "gaussian_relative":           "gaussian_relative",
        "sinusoidal_absolute":         "model_native",   # no custom file
        "gaussian_absolute":           "model_native",   # no custom file
    }
    enc_file = enc_file_map.get(encoding, encoding)

    model_cfg   = OmegaConf.load(CONFIG_ROOT / "model"    / f"{model}.yaml")
    enc_cfg     = OmegaConf.load(CONFIG_ROOT / "encoding" / f"{enc_file}.yaml")
    dataset_cfg = OmegaConf.load(CONFIG_ROOT / "dataset"  / f"{dataset}.yaml")

    cfg = OmegaConf.merge(model_cfg, {"encoding": enc_cfg, "dataset": dataset_cfg})
    # Override encoding name with the CLI value (so the runner can dispatch)
    cfg.encoding.name = encoding
    return cfg


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Adapter factory ───────────────────────────────────────────────────────────

def build_adapter(
    encoding_cfg: DictConfig,
    num_heads: int,
) -> Optional[PositionalBiasAdapter]:
    """Instantiate the correct adapter from encoding config."""
    enc_name = encoding_cfg.name

    if enc_name in ("model_native", "sinusoidal_absolute", "gaussian_absolute"):
        return None

    if enc_name == "gaussian_relative":
        from src.positional_bias.gaussian_relative import GaussianRelativeBias
        return GaussianRelativeBias(sigma=encoding_cfg.get("sigma", 3.0))

    if enc_name == "gaussian_relative_learnable":
        from src.positional_bias.gaussian_learnable import GaussianLearnableBias
        return GaussianLearnableBias(
            num_heads=num_heads,
            sigma_init=encoding_cfg.get("sigma_init", 3.0),
        )

    if enc_name == "gaussian_sentence_aware":
        from src.positional_bias.sentence_aware import SentenceAwareGaussianBias
        return SentenceAwareGaussianBias(
            num_heads=num_heads,
            sigma_init=encoding_cfg.get("sigma_init", 1.5),
        )

    raise ValueError(f"Unknown encoding: {enc_name!r}")


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig, adapter: Optional[PositionalBiasAdapter]):
    """Instantiate the correct model wrapper."""
    model_name = cfg.name

    if model_name == "roberta":
        from src.models.roberta_adapter import RoBERTaWithBias
        return RoBERTaWithBias(
            model_name=cfg.pretrained,
            bias_adapter=adapter,
            num_labels=cfg.num_labels,
        )

    if model_name == "deberta":
        from src.models.deberta_adapter import DeBERTaWithBias
        return DeBERTaWithBias(
            model_name=cfg.pretrained,
            bias_adapter=adapter,
            num_labels=cfg.num_labels,
        )

    if model_name == "longformer":
        from src.models.longformer_adapter import LongformerWithBias
        return LongformerWithBias(
            model_name=cfg.pretrained,
            bias_adapter=adapter,
            num_labels=cfg.num_labels,
        )

    raise ValueError(f"Unknown model: {model_name!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Modular PE experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      required=True, choices=VALID_MODELS)
    p.add_argument("--encoding",   required=True, choices=VALID_ENCODINGS)
    p.add_argument("--dataset",    required=True, choices=VALID_DATASETS)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--output_dir", type=str,   default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.model, args.encoding, args.dataset)

    run_name   = f"{args.model}__{args.encoding}__{args.dataset}__seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist full config snapshot for exact reproducibility
    cfg_snapshot = output_dir / "config_snapshot.yaml"
    OmegaConf.save(cfg, cfg_snapshot)

    mlflow.set_experiment("modular-pe-gaussian")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model":      args.model,
            "encoding":   args.encoding,
            "dataset":    args.dataset,
            "seed":       args.seed,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
        })
        mlflow.log_artifact(str(cfg_snapshot), artifact_path="config")

        adapter = build_adapter(cfg.encoding, num_heads=cfg.num_attention_heads)
        model   = build_model(cfg, adapter)

        model = run_training(model=model, cfg=cfg, args=args, output_dir=output_dir)
        metrics = run_evaluation(model=model, cfg=cfg, args=args, output_dir=output_dir)

        mlflow.log_metrics(metrics)

        # Log learned sigma distributions if adapter supports it
        if adapter is not None and hasattr(adapter, "log_sigma"):
            from src.analysis.sigma_analysis import plot_sigma_distribution, log_sigma_stats
            sigma_plot = output_dir / "sigma_distribution.png"
            log_sigma_stats(model, step=args.epochs)

        print(f"\n{'─'*55}")
        print(f"  Run: {run_name}")
        print(f"{'─'*55}")
        for k, v in metrics.items():
            print(f"  {k:<25} {v:.4f}")
        print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
