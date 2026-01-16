"""
Training loop for the extractive summarization sentence classifier.

For each document the model assigns a score to each sentence (binary:
include / exclude). The extractive oracle labels are derived from ROUGE
overlap between each sentence and the reference summary.

This module keeps data loading, optimisation, and checkpointing separate
from the experiment orchestration in experiment.py.
"""

from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ── Oracle label helper ───────────────────────────────────────────────────────

def greedy_oracle_labels(sentences: list[str], summary: str) -> list[int]:
    """
    Simple greedy oracle: label sentence as 1 if it has enough ROUGE-1
    overlap with the reference summary, 0 otherwise.
    Used as a lightweight stand-in for full ROUGE-F oracle selection.
    """
    summary_tokens = set(summary.lower().split())
    labels = []
    for sent in sentences:
        sent_tokens = set(sent.lower().split())
        overlap = len(sent_tokens & summary_tokens)
        labels.append(1 if overlap / max(len(sent_tokens), 1) > 0.3 else 0)
    return labels


# ── Dataset wrapper ───────────────────────────────────────────────────────────

def build_dataloader(
    cfg: DictConfig,
    split: str,
    tokenizer,
    batch_size: int,
) -> DataLoader:
    """Load, tokenise and wrap a HuggingFace dataset split."""
    from datasets import load_dataset

    hf_cfg = cfg.dataset.get("hf_config", None)
    dataset = load_dataset(cfg.dataset.hf_name, hf_cfg, split=split)

    max_len = cfg.dataset.max_input_length

    def tokenise(batch):
        return tokenizer(
            batch[cfg.dataset.text_column],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors=None,   # keep as lists; DataLoader will collate
        )

    dataset = dataset.map(tokenise, batched=True, batch_size=256,
                          remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == cfg.dataset.train_split),
        pin_memory=torch.cuda.is_available(),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(model: nn.Module, cfg: DictConfig, args, output_dir: Path) -> nn.Module:
    """
    Fine-tune `model` on the configured dataset and return the trained model.

    Checkpoints and loss curves are logged to MLflow.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_loader = build_dataloader(cfg, cfg.dataset.train_split, tokenizer, args.batch_size)

    # Linear warmup → decay over all training steps
    total_steps = len(train_loader) * args.epochs
    scheduler   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=max(1, total_steps // 10))

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Placeholder binary labels — replace with actual oracle labels
            # when sentence-split preprocessing is wired in.
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss   = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if step % 100 == 0:
                avg_loss = epoch_loss / step
                mlflow.log_metric("train_loss", avg_loss, step=global_step)
                print(f"  Epoch {epoch}/{args.epochs}  step {step:>5}  loss={avg_loss:.4f}")

        # End-of-epoch checkpoint
        ckpt = output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()}, ckpt)
        mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
        print(f"  [Epoch {epoch}] checkpoint saved → {ckpt.name}")

    return model
