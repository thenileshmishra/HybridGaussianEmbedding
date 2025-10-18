"""Summary generation from sentence scores."""

import torch
import torch.nn as nn


class SummaryGenerator(nn.Module):
    """Generates extractive summaries by selecting top-scored sentences.

    Given sentence scores from the inter-sentence transformer,
    selects the top-k sentences to form the extractive summary.

    Args:
        top_k: Number of sentences to select (default: 3).
    """

    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k

    def forward(self, sent_scores, cls_indices, input_text):
        """Generate summary from sentence scores.

        Args:
            sent_scores: Sentence importance scores (batch x n_sents).
            cls_indices: Indices of CLS tokens.
            input_text: Original list of sentences.

        Returns:
            Generated summary as a single string.
        """
        actual_sentences = [s for s in input_text if s.strip() != ""]
        actual_len = len(actual_sentences)

        if actual_len == 0:
            return ""

        # Trim scores to actual number of sentences
        trimmed_scores = sent_scores[:, :actual_len]
        k = min(self.top_k, actual_len)
        top_sents = torch.topk(trimmed_scores, k=k, dim=1)

        sen_indices = top_sents.indices.squeeze().tolist()
        if isinstance(sen_indices, int):
            sen_indices = [sen_indices]

        selected_sentences = [actual_sentences[i] for i in sen_indices]
        return " ".join(selected_sentences)
