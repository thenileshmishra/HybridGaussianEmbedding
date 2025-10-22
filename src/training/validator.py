"""Validation loop for the extractive summarization model."""

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.summary_generator import SummaryGenerator
from src.evaluation.rouge import compute_rouge, init_rouge_results, average_rouge
from src.evaluation.bertscore import compute_bert_score, init_bert_results
from src.utils.device import move_to_device, get_device
from src.utils.gpu import monitor_gpu_usage


def validate(config, roberta_model, embedder, interencoder, logit_layer,
             val_losses, val_loader, model_config):
    """Run validation over the validation dataset.

    Args:
        config: RobertaConfig instance.
        roberta_model: Trained RoBERTa encoder model.
        embedder: Document embedder model.
        interencoder: Inter-sentence transformer encoder.
        logit_layer: Linear layer for logit projection.
        val_losses: List of accumulated validation losses.
        val_loader: DataLoader for validation data.
        model_config: ModelConfig instance.

    Returns:
        Dict with validation results (losses, ROUGE, BERTScore).
    """
    r1, r2, rL, rLs = init_rouge_results()
    bert_results = init_bert_results()
    d = model_config.d_model

    device = get_device()
    loss_fn = nn.BCEWithLogitsLoss()
    summary_gen = SummaryGenerator(top_k=3)

    roberta_model.to(device)
    embedder.to(device)
    interencoder.to(device)
    logit_layer.to(device)

    roberta_model.eval()
    embedder.eval()
    interencoder.eval()
    logit_layer.eval()

    val_loss = 0
    val_sentences = 0

    print("\nValidation starting...")
    monitor_gpu_usage()

    with torch.no_grad():
        for batch_id, (batch_sources, batch_targets, batch_labels) in enumerate(val_loader):
            for doc_idx in range(len(batch_sources)):
                input_text = batch_sources[doc_idx]
                target_text = batch_targets[doc_idx]
                tarlabel = batch_labels[doc_idx]

                if len(input_text) == 0 or len(tarlabel) == 0:
                    continue

                # Embedding
                embeddings, attention_mask, sep_indices = embedder(
                    input_text, model_config.s_max, model_config.s_min,
                    model_config.mu, model_config.sigma
                )
                cls_indices = [0] + [x + 1 for x in sep_indices][:-1]
                cls_indices = [idx for idx in cls_indices
                              if idx < config.max_position_embeddings]

                embeddings = move_to_device(embeddings, device)
                attention_mask = move_to_device(attention_mask, device)
                cls_indices_tensor = torch.tensor(cls_indices).to(device)

                # Forward pass
                cls_embeddings, _ = roberta_model(embeddings, cls_indices_tensor, attention_mask)
                cls_reshaped = cls_embeddings.permute(1, 0, 2)
                result, cls_idx, x = interencoder(cls_reshaped, attention_mask, cls_indices_tensor, d)

                # Summary and loss
                generated_summary = summary_gen(result, cls_idx, input_text)
                x = x.to(device)
                logits = logit_layer(x).squeeze(-1)
                target_tensor = torch.tensor(tarlabel[:x.size(1)]).float().to(device).unsqueeze(0)
                loss = loss_fn(logits, target_tensor)

                val_loss += loss.item() * target_tensor.numel()
                val_sentences += target_tensor.numel()

                # Metrics
                r1, r2, rL, rLs = compute_rouge(
                    generated_summary, target_text, r1, r2, rL, rLs
                )
                bert_results = compute_bert_score(generated_summary, target_text, bert_results)

    avg_val_loss = val_loss / val_sentences if val_sentences > 0 else 0.0
    val_losses.append(avg_val_loss)

    print("Validation completed.")
    return {
        "val_losses": val_losses,
        "avg_rouge1": average_rouge(r1),
        "avg_rouge2": average_rouge(r2),
        "avg_rougeL": average_rouge(rL),
        "avg_rougeLsum": average_rouge(rLs),
        "avg_bert": average_rouge(bert_results),
    }
