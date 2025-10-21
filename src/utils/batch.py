"""Batch processing utilities for embedding and target tensors."""

import torch
from torch.nn.utils.rnn import pad_sequence


def batchify_embeddings(embeddings_list, attention_masks, cls_indices_list, device):
    """Pad and batch multiple document embeddings.

    Args:
        embeddings_list: List of embedding tensors.
        attention_masks: List of attention mask tensors.
        cls_indices_list: List of CLS index lists.
        device: Target device.

    Returns:
        Tuple of (padded_embeddings, padded_masks, padded_cls_indices) on device.
    """
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True)

    max_cls_len = max(len(cls) for cls in cls_indices_list)
    padded_cls_indices = torch.zeros(len(cls_indices_list), max_cls_len, dtype=torch.long)
    for i, cls in enumerate(cls_indices_list):
        padded_cls_indices[i, :len(cls)] = torch.tensor(cls)

    return (
        padded_embeddings.to(device),
        padded_attention_masks.to(device),
        padded_cls_indices.to(device),
    )


def build_target_tensor(batch_labels, max_num_sentences, device):
    """Build a padded target tensor from variable-length label lists.

    Args:
        batch_labels: List of label lists (one per document).
        max_num_sentences: Maximum number of sentences for padding.
        device: Target device.

    Returns:
        Padded target tensor of shape (batch_size x max_num_sentences).
    """
    batch_size = len(batch_labels)
    target_tensor = torch.zeros((batch_size, max_num_sentences), dtype=torch.float)
    for i, label_list in enumerate(batch_labels):
        length = min(len(label_list), max_num_sentences)
        target_tensor[i, :length] = torch.tensor(label_list[:length], dtype=torch.float)
    return target_tensor.to(device)
