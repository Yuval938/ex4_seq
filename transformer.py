from torch import nn
import torch
import torch.nn.functional as F
import math
# --- CORRECTLY IMPORTING THE MODULES ---
import attention
import mlp

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len: int, dropout_rate: float):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N = x.shape
        position_ids = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(position_ids)
        embeddings = tok_embeddings + pos_embeddings
        return self.dropout(embeddings)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len: int, dropout_rate: float):
        super().__init__()
        # --- USING THE IMPORTED ATTENTION MODULE ---
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len, dropout_rate)
        # --- USING THE IMPORTED MLP MODULE ---
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # Pre-LayerNorm architecture with residual connections
        x = inputs + self.causal_attention(self.layer_norm_1(inputs))
        x = x + self.resid_dropout(self.mlp(self.layer_norm_2(x)))
        return x

class TransformerLM(nn.Module):
    def __init__(
            self, n_layers: int, n_heads: int, embed_size: int, max_context_len: int,
            vocab_size: int, mlp_hidden_size: int, dropout_rate: float):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len, dropout_rate)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, dropout_rate) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.max_context_len = max_context_len
        self.init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameter count: {n_params/1e6:.2f}M")

    def init_weights(self):
        for pn, p in self.named_parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float = 1.0, top_k: int = 0) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        device = next(self.parameters()).device
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                current_input = feed_to_lm[-self.max_context_len:]
                input_tensor = torch.tensor([current_input], dtype=torch.long, device=device)
                logits = self(input_tensor)
                logits_for_last = logits[0, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_for_last, k=min(top_k, logits_for_last.size(-1)))
                    logits_for_last[logits_for_last < v[-1]] = -float('Inf')
                probs = F.softmax(logits_for_last, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated