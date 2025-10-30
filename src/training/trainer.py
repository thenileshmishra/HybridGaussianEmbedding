"""Training loop for the extractive summarization model."""

import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from transformers import RobertaConfig

from src.models.embedder import RoBertaDocumentEmbedder
from src.models.roberta_encoder import RoBERTaEncoder
from src.models.transformer_layers import TransformerInterEncoder
from src.models.summary_generator import SummaryGenerator
from src.evaluation.rouge import compute_rouge, init_rouge_results, average_rouge
from src.evaluation.bertscore import compute_bert_score, init_bert_results
from src.utils.device import move_to_device, get_device
from src.utils.gpu import monitor_gpu_usage, log_gpu_metrics
from src.training.validator import validate


def train(train_loader, val_loader, config):
    """Execute the full training pipeline.

    Initializes model components, handles checkpoint resumption,
    and runs the training loop with periodic validation.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: ModelConfig instance with all hyperparameters.

    Returns:
        Dict containing training results (save_path, losses, metrics).
    """
    r1, r2, rL, rLs = init_rouge_results()
    bert_results = init_bert_results()

    d = config.d_model
    seq_len = config.seq_len
    num_inter_layers = config.num_inter_layers
    heads = config.num_heads
    dropout = config.dropout

    best_val_rougeL = -1
    os.makedirs(config.checkpoint_dir, exist_ok=True)  # Ensure dirs exist
    os.makedirs(config.metrics_dir, exist_ok=True)
    save_path = os.path.join(config.checkpoint_dir, f"roberta_epoch{config.num_epochs}.pth")
    resume_doc_idx = 0

    print(f"\nTraining with: s_max={config.s_max}, s_min={config.s_min}, "
          f"mu={config.mu}, sigma={config.sigma}")

    # Initialize model components
    roberta_config = RobertaConfig.from_pretrained(
        "roberta-base", output_attentions=True,
        max_position_embeddings=config.max_position_embeddings
    )
    embedder = RoBertaDocumentEmbedder(seq_len=seq_len, d=d)
    interencoder = TransformerInterEncoder(d, seq_len, num_inter_layers, dropout, heads)
    logit_layer = nn.Linear(d, 1)

    device = get_device()
    print(f"Using device: {device}")

    embedder.to(device)
    roberta_model = RoBERTaEncoder(roberta_config, embedder)
    roberta_model.to(device)
    interencoder.to(device)
    logit_layer.to(device)

    # Optimizer, scheduler, loss
    optimizer = optim.AdamW(
        list(roberta_model.parameters()) + list(logit_layer.parameters()),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
    )
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_losses = []
    val_losses = []
    start_epoch = 0

    # Resume from checkpoint if available
    checkpoint_pattern = os.path.join(config.checkpoint_dir, "roberta_temp_checkpoint_epoch*.pth")
    checkpoints = sorted(glob.glob(checkpoint_pattern), reverse=True)
    if checkpoints:
        latest = checkpoints[0]
        checkpoint = torch.load(latest, map_location=device)
        roberta_model.load_state_dict(checkpoint["model_state_dict"])
        interencoder.load_state_dict(checkpoint["interencoder_state_dict"])
        logit_layer.load_state_dict(checkpoint["logit_layer_state_dict"])

        optimizer = optim.AdamW(
            list(roberta_model.parameters()) + list(logit_layer.parameters()),
            lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay,
        )
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("Optimizer state mismatch, reinitializing from scratch.")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_rougeL = checkpoint.get("val_rougeL", -1)
        resume_doc_idx = checkpoint.get("doc_idx", 0)
        print(f"Resuming from epoch {start_epoch}, doc {resume_doc_idx}, "
              f"best ROUGE-L: {best_val_rougeL:.4f}")

    summary_gen = SummaryGenerator(top_k=3)
    monitor_gpu_usage()
    print(f"Training for {config.num_epochs} epochs...")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        monitor_gpu_usage()

        roberta_model.train()
        embedder.train()
        interencoder.train()
        logit_layer.train()

        epoch_loss = 0
        total_sentences = 0

        for batch_id, (batch_sources, batch_targets, batch_labels) in enumerate(tqdm(train_loader)):
            for doc_idx in range(len(batch_sources)):
                input_text = batch_sources[doc_idx]
                target_text = batch_targets[doc_idx]
                tarlabel = batch_labels[doc_idx]

                if len(input_text) == 0 or len(tarlabel) == 0:
                    continue

                absolute_doc_index = batch_id * len(batch_sources) + doc_idx
                if absolute_doc_index < resume_doc_idx:
                    continue

                # Embedding
                embeddings, attention_mask, sep_indices = embedder(
                    input_text, config.s_max, config.s_min, config.mu, config.sigma
                )
                cls_indices = [0] + [x + 1 for x in sep_indices][:-1]
                cls_indices = [idx for idx in cls_indices
                              if idx < roberta_config.max_position_embeddings]

                embeddings = move_to_device(embeddings, device)
                attention_mask = move_to_device(attention_mask, device)
                cls_indices_tensor = torch.tensor(cls_indices).to(device)

                optimizer.zero_grad()

                # Forward pass through encoder pipeline
                cls_embeddings, _ = roberta_model(embeddings, cls_indices_tensor, attention_mask)
                cls_reshaped = cls_embeddings.permute(1, 0, 2)
                result, cls_idx, x = interencoder(cls_reshaped, attention_mask, cls_indices_tensor, d)

                # Summary generation
                generated_summary = summary_gen(result, cls_idx, input_text)

                # Loss computation
                num_sents = x.size(1)
                target_tensor = torch.tensor(tarlabel[:num_sents]).float().to(device).unsqueeze(0)
                x = move_to_device(x, device)
                logits = logit_layer(x).squeeze(-1)

                loss = loss_fn(logits, target_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(roberta_model.parameters(), config.max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item() * target_tensor.numel()
                total_sentences += target_tensor.numel()

                # Evaluation metrics
                r1, r2, rL, rLs = compute_rouge(
                    generated_summary, target_text, r1, r2, rL, rLs
                )
                bert_results = compute_bert_score(generated_summary, target_text, bert_results)

                # Periodic checkpointing
                if absolute_doc_index % config.checkpoint_interval == 0:
                    ckpt_path = os.path.join(
                        config.checkpoint_dir,
                        f"roberta_temp_checkpoint_epoch{epoch + 1}_doc{doc_idx + 1}.pth"
                    )
                    torch.save({
                        "epoch": epoch,
                        "doc_idx": absolute_doc_index,
                        "model_state_dict": roberta_model.state_dict(),
                        "interencoder_state_dict": interencoder.state_dict(),
                        "logit_layer_state_dict": logit_layer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_rougeL": best_val_rougeL,
                    }, ckpt_path)

        scheduler.step()
        avg_epoch_loss = epoch_loss / total_sentences if total_sentences > 0 else 0.0
        epoch_losses.append(avg_epoch_loss)

        # Validation
        val_results = validate(
            roberta_config, roberta_model, embedder, interencoder, logit_layer,
            val_losses, val_loader, config
        )
        val_losses = val_results["val_losses"]
        val_avg_rougeL = val_results["avg_rougeL"]

        # Save best model
        if val_avg_rougeL > best_val_rougeL:
            best_val_rougeL = val_avg_rougeL
            torch.save({
                "model_state_dict": roberta_model.state_dict(),
                "interencoder_state_dict": interencoder.state_dict(),
                "logit_layer_state_dict": logit_layer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "val_rougeL": val_avg_rougeL,
            }, save_path)
            print(f"Best model saved with ROUGE-L: {val_avg_rougeL:.4f}")

    return {
        "save_path": save_path,
        "epoch_losses": epoch_losses,
        "avg_rouge1": average_rouge(r1),
        "avg_rouge2": average_rouge(r2),
        "avg_rougeL": average_rouge(rL),
        "avg_rougeLsum": average_rouge(rLs),
        "val_losses": val_losses,
        "val_avg_rougeL": best_val_rougeL,
    }
