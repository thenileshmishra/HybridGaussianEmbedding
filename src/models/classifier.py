"""Classifier module for sentence-level scoring."""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Binary classifier for sentence extraction.

    Projects hidden states to a single score per sentence,
    applying sigmoid activation masked by CLS token positions.

    Args:
        hidden_size: Dimension of input hidden states.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        """Compute sentence extraction scores.

        Args:
            x: Hidden states from the inter-sentence encoder.
            mask_cls: Binary mask for valid CLS positions.

        Returns:
            Sentence scores (batch x n_sents).
        """
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores
