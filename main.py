"""Main entry point for the Hybrid Gaussian Positional Encoding experiment."""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import get_default_config
from src.data.dataset import SummaryDataset, summary_collate_fn
from src.data.loader import load_cnn_dailymail
from src.data.preprocessing import preprocess_data
from src.data.sampling import stratified_sampling
from src.data.extractive import prepare_source_target
from src.training.trainer import train
from src.training.tester import test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Gaussian Positional Encoding for Extractive Summarization"
    )
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--s_max", type=float, default=2.0)
    parser.add_argument("--s_min", type=float, default=0.3)
    parser.add_argument("--mu", type=float, default=260.0)
    parser.add_argument("--sigma", type=float, default=65.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to preprocessed .pt data file")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_or_prepare_data(args, config):
    """Load preprocessed data or run full preprocessing pipeline."""
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading preprocessed data from {args.data_path}")
        data = torch.load(args.data_path)
        dataset = SummaryDataset(data["source"], data["target"], data["labels"])
    else:
        print("Loading dataset from scratch...")
        articles, highlights = load_cnn_dailymail()
        sampled_articles, sampled_highlights = stratified_sampling(
            articles, highlights, sample_size=config.sample_size
        )
        train_src, train_tgt, combined_array = preprocess_data(
            sampled_articles, sampled_highlights
        )
        mysource, mytarget, _ = prepare_source_target(combined_array)

        # Build dataset from source-target pairs
        flat_source = [row_data for _, row_data in mysource]
        flat_target = [row_data for _, row_data in mytarget]
        # Initialize empty labels (to be filled during embedding)
        labels = [[0] * len(src) for src in flat_source]
        dataset = SummaryDataset(flat_source, flat_target, labels)

    return dataset


def main():
    args = parse_args()
    config = get_default_config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        s_max=args.s_max,
        s_min=args.s_min,
        mu=args.mu,
        sigma=args.sigma,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    torch.manual_seed(config.random_seed)

    # Prepare data splits
    dataset = load_or_prepare_data(args, config)
    train_data, test_data = train_test_split(
        dataset, test_size=config.test_split, random_state=config.random_seed
    )
    train_data, val_data = train_test_split(
        train_data, test_size=config.val_split, random_state=config.random_seed
    )

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True,
        collate_fn=summary_collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False,
        collate_fn=summary_collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False,
        collate_fn=summary_collate_fn
    )

    # Training
    if args.mode in ("train", "both"):
        print("\n" + "=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)
        train_results = train(train_loader, val_loader, config)
        save_path = train_results["save_path"]
        print(f"\nTraining complete. Best model saved to: {save_path}")

    # Testing
    if args.mode in ("test", "both"):
        print("\n" + "=" * 60)
        print("TESTING PHASE")
        print("=" * 60)
        if args.mode == "test":
            save_path = os.path.join(config.checkpoint_dir, f"roberta_epoch{config.num_epochs}.pth")
        test_results = test(test_loader, config, save_path)
        print(f"\nTest Results:")
        print(f"  ROUGE-1: {test_results['avg_rouge1']:.4f}")
        print(f"  ROUGE-2: {test_results['avg_rouge2']:.4f}")
        print(f"  ROUGE-L: {test_results['avg_rougeL']:.4f}")
        print(f"  ROUGE-Lsum: {test_results['avg_rougeLsum']:.4f}")
        print(f"  BERTScore: {test_results['avg_bert']:.4f}")


if __name__ == "__main__":
    main()
