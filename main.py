from __future__ import annotations
import torch
from torch import optim
from transformer import TransformerLM
import data
import lm
import os
import math

# --- Configuration (remains the same) ---
PROFILE = 'en'

if PROFILE == 'en':
    data_path, out_dir = 'data/', 'out-en'
    n_layers, n_heads, embed_size = 6, 6, 384
    batch_size, seq_len = 64, 256
elif PROFILE == 'heb':
    data_path, out_dir = 'heb-data/', 'out-heb'
    n_layers, n_heads, embed_size = 6, 6, 384
    batch_size, seq_len = 64, 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval, log_interval, num_iters = 200, 20, 5000
learning_rate, warmup_iters, lr_decay_iters = 6e-4, 200, 5000
weight_decay, dropout_rate, gradient_clipping = 0.1, 0.1, 1.0
sampling_temperature, top_k = 0.8, 5


# --- Learning Rate Scheduler (remains the same) ---
def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > lr_decay_iters: return learning_rate / 10
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (learning_rate / 10) + coeff * (learning_rate - (learning_rate / 10))


if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)

    # --- Data Loading ---
    print(f"Loading data for profile: {PROFILE}...")
    tokenizer, tokenized_data = data.load_data(data_path)

    # --- THIS IS THE FIX ---
    # 1. Create the iterator that yields single examples (lists of ints)
    single_item_iter = data.RandomOrderDataIterator(tokenized_data, seq_len + 1)
    # 2. Wrap it with `batch_items` to create an iterator that yields BATCHES (torch.Tensors)
    batch_iter = iter(data.batch_items(single_item_iter, batch_size))
    # --- END OF FIX ---

    tokenizer.save(os.path.join(out_dir, 'tokenizer.json'))

    # --- Model and Optimizer Initialization (remains the same) ---
    model_args = dict(n_layers=n_layers, n_heads=n_heads, embed_size=embed_size, max_context_len=seq_len,
                      vocab_size=tokenizer.vocab_size(), mlp_hidden_size=embed_size * 4, dropout_rate=dropout_rate)
    model = TransformerLM(**model_args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=[0.9, 0.95])

    # --- Checkpoint Loading (remains the same) ---
    iter_num, best_val_loss = 0, 1e9
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        print(f"Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']

    # --- Training Loop ---
    print(f"Starting training on {device}...")
    model.train()
    while iter_num < num_iters:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        # --- THIS IS THE SECOND PART OF THE FIX ---
        # Get the next BATCH from the correct iterator
        batch_data = next(batch_iter).to(device)
        # --- END OF FIX ---

        batch_x, batch_y = lm.batch_to_labeled_samples(batch_data)

        # The line below was simplified in my previous version, I've expanded it for clarity
        logits = model(batch_x)
        loss = lm.compute_loss(logits, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if iter_num % log_interval == 0:
            print(f"Iter {iter_num}/{num_iters}, Loss: {loss.item():.4f}, LR: {lr:.6f}")

        if iter_num > 0 and iter_num % eval_interval == 0:
            model.eval()
            print("--- Evaluating and Saving Checkpoint ---")
            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': model_args,
                            'iter_num': iter_num, 'best_val_loss': best_val_loss}, ckpt_path)
            prompt = "The " if PROFILE == 'en' else "שלום"
            tokenized_prompt = tokenizer.tokenize(prompt)
            continuation = model.better_sample_continuation(tokenized_prompt, 150, sampling_temperature, top_k)
            print(f"Sample:\n{tokenizer.detokenize(tokenized_prompt + continuation)}\n---")
            model.train()
        iter_num += 1

    print("Training finished.")