from __future__ import annotations
import torch
from torch.nn import functional as F


def batch_to_labeled_samples(batch: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
    Converts a batch of sequences into input and target pairs for language modeling.
    """
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    return inputs, labels


def compute_loss(logits, gold_labels):
    """
    Computes the cross-entropy loss for language modeling.
    """
    # logits size is (batch, seq_len, vocab_size)
    # gold_labels size is (batch, seq_len)
    B, T, V = logits.shape

    # --- THIS IS THE FIX ---
    # Use .reshape() instead of .view() to handle non-contiguous tensors.
    # It's safer and recommended by the error message itself.
    logits_flat = logits.reshape(B * T, V)
    labels_flat = gold_labels.reshape(B * T)
    # --- END OF FIX ---

    # Compute loss, ignoring the padding token (assuming pad_id is 0)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)

    return loss