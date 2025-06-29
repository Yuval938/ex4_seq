from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads=1):
    """Creates a linear layer for a single attention head."""
    assert input_vector_dim % n_heads == 0, "Input dimension must be divisible by number of heads"
    head_dim = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, 3 * head_dim, bias=False)


def kqv(x, linear):
    """Computes Q, K, and V from the input x using a single linear projection."""
    projected = linear(x)
    d_head = projected.shape[-1] // 3
    k, q, v = torch.split(projected, [d_head, d_head, d_head], dim=-1)
    return k, q, v


def attention_scores(q, k):
    """Computes scaled dot-product attention scores."""
    d_k = k.size(-1)
    return (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d_k))


def create_causal_mask(max_context_len, **kwargs):
    """Creates a causal mask to prevent attention to future tokens."""
    mask = torch.tril(torch.ones(max_context_len, max_context_len))
    return mask.unsqueeze(0)


def self_attention(v, A, mask=None, dropout_layer=None):
    """Computes the self-attention output, applying dropout to attention weights if provided."""
    if mask is not None:
        N = A.shape[1]
        mask_slice = mask[:, :N, :N]
        A = A.masked_fill(mask_slice == 0, float('-inf'))

    probs = F.softmax(A, dim=-1)

    if dropout_layer is not None:
        probs = dropout_layer(probs)

    return probs @ v


def self_attention_layer(x, kqv_matrix, attention_mask, dropout_layer=None):
    """Orchestrates a single head of self-attention."""
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(q, k)
    sa = self_attention(v, att, attention_mask, dropout_layer)
    return sa


def multi_head_attention_layer(x, kqv_matrices, mask, dropout_layer=None):
    """Computes multi-head attention by concatenating the outputs of individual heads."""
    heads = [self_attention_layer(x, kqv_matrix, mask, dropout_layer) for kqv_matrix in kqv_matrices]
    sa = torch.cat(heads, dim=-1)
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    """The high-level module for causal self-attention, which USES the functions above."""

    def __init__(self, embed_dim, n_heads, max_context_len, dropout_rate):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for _ in range(n_heads)])
        mask = create_causal_mask(max_context_len=max_context_len)
        self.register_buffer("mask", mask)
        self.proj = nn.Linear(embed_dim, embed_dim)
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # This now calls the modular function, passing the dropout layer to it
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask, self.attn_dropout)
        # Apply final projection and residual dropout
        sa = self.resid_dropout(self.proj(sa))
        return sa