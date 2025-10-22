"""Testing loop for the extractive summarization model."""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaConfig

from src.models.embedder import RoBertaDocumentEmbedder
from src.models.roberta_encoder import RoBERTaEncoder
from src.models.transformer_layers import TransformerInterEncoder
from src.models.summary_generator import SummaryGenerator
from src.evaluation.rouge import compute_rouge, init_rouge_results, average_rouge
from src.evaluation.bertscore import compute_bert_score, init_bert_results
from src.utils.device import get_device
from src.utils.gpu import monitor_gpu_usage


def test(test_loader, config, save_path: str):
    """Run evaluation on the test dataset using the best saved model.

    Args:
        test_loader: DataLoader for test data.
        config: ModelConfig instance.
        save_path: Path to the saved model checkpoint.

    Returns:
        Dict with test results (losses, ROUGE scores, BERTScore).
    """
    d = config.d_model
    seq_len = config.seq_len
    num_inter_layers = config.num_inter_layers
    heads = config.num_heads
    dropout = config.dropout

    r1, r2, rL, rLs = init_rouge_results()
    bert_results = init_bert_results()

    # Initialize model components
    roberta_config = RobertaConfig.from_pretrained(
        "roberta-base", output_attentions=True,
        max_position_embeddings=config.max_position_embeddings,
    )
    embedder = RoBertaDocumentEmbedder(seq_len=seq_len, d=d)
    interencoder = TransformerInterEncoder(d, seq_len, num_inter_layers, dropout, heads)
    logit_layer = nn.Linear(d, 1)

    device = get_device()
    roberta_model = RoBERTaEncoder(roberta_config, embedder)

    # Load checkpoint
    checkpoint = torch.load(save_path, map_location=device)
    roberta_model.load_state_dict(checkpoint["model_state_dict"])
    interencoder.load_state_dict(checkpoint["interencoder_state_dict"])
    logit_layer.load_state_dict(checkpoint["logit_layer_state_dict"])

    roberta_model.to(device).eval()
    embedder.to(device).eval()
    interencoder.to(device).eval()
    logit_layer.to(device).eval()

    loss_fn = nn.BCEWithLogitsLoss()
    summary_gen = SummaryGenerator(top_k=3)

    test_loss = 0
    test_sentences = 0

    monitor_gpu_usage()
    print("Testing starting...")

    with torch.no_grad():
        for batch_id, (batch_sources, batch_targets, batch_labels) in enumerate(test_loader):
            for doc_idx in range(len(batch_sources)):
                input_text = batch_sources[doc_idx]
                target_text = batch_targets[doc_idx]
                tarlabel = batch_labels[doc_idx]

                if len(input_text) == 0 or len(tarlabel) == 0:
                    continue

                print(f"Processing test doc {batch_id * len(batch_sources) + doc_idx + 1}")

                # Embedding
                embeddings, attention_mask, sep_indices = embedder(
                    input_text, config.s_max, config.s_min, config.mu, config.sigma
                )
                cls_indices = [0] + [x + 1 for x in sep_indices][:-1]
                cls_indices = [idx for idx in cls_indices
                              if idx < roberta_config.max_position_embeddings]

                embeddings = embeddings.to(device)
                attention_mask = attention_mask.to(device)
                cls_indices_tensor = torch.tensor(cls_indices).to(device)

                # Forward pass
                cls_embeddings, _ = roberta_model(embeddings, cls_indices_tensor, attention_mask)
                cls_reshaped = cls_embeddings.permute(1, 0, 2)
                result, cls_idx, x = interencoder(cls_reshaped, attention_mask, cls_indices_tensor, d)

                # Summary and loss
                generated_summary = summary_gen(result, cls_idx, input_text)
                x = x.to(device)
                logits = logit_layer(x).squeeze(-1)
                num_sents = x.size(1)
                target_tensor = torch.tensor(tarlabel[:num_sents]).float().to(device).unsqueeze(0)
                loss = loss_fn(logits, target_tensor)

                test_loss += loss.item() * target_tensor.numel()
                test_sentences += target_tensor.numel()

                # Metrics
                r1, r2, rL, rLs = compute_rouge(
                    generated_summary, target_text, r1, r2, rL, rLs
                )
                bert_results = compute_bert_score(generated_summary, target_text, bert_results)

    print("Testing completed.")
    return {
        "test_loss": test_loss / test_sentences if test_sentences > 0 else 0.0,
        "avg_rouge1": average_rouge(r1),
        "avg_rouge2": average_rouge(r2),
        "avg_rougeL": average_rouge(rL),
        "avg_rougeLsum": average_rouge(rLs),
        "avg_bert": average_rouge(bert_results),
        "rouge1_results": r1,
        "rouge2_results": r2,
        "rougeL_results": rL,
        "rougeLsum_results": rLs,
    }
