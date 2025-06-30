from __future__ import annotations
import torch
from torch import optim
from transformer import TransformerLM
import data
import lm
import os
import math
import logging  # For file logging
import matplotlib.pyplot as plt  # For plotting

# --- Configuration (remains the same) ---
PROFILE = 'en'
if PROFILE == 'en':
    data_path, out_dir = 'data/', 'out-en'
elif PROFILE == 'heb':
    data_path, out_dir = 'heb-data/', 'out-heb'

n_layers, n_heads, embed_size = 6, 6, 384
batch_size, seq_len = 128, 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval, log_interval, num_iters, eval_iters = 200, 20, 50000, 100
learning_rate, warmup_iters, lr_decay_iters = 5e-4, 200, 5000
weight_decay, dropout_rate, gradient_clipping = 0.01, 0.1, 1.0
sampling_temperature, top_k = 0.8, 5


# --- Learning Rate Scheduler (remains the same) ---
def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > lr_decay_iters: return learning_rate / 10
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (learning_rate / 10) + coeff * (learning_rate - (learning_rate / 10))


# --- Evaluation Function (remains the same) ---
@torch.no_grad()
def estimate_loss(model, train_batch_iter, val_batch_iter):
    out = {}
    model.eval()
    for split, iterator in [('train', train_batch_iter), ('val', val_batch_iter)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                batch = next(iterator).to(device)
            except StopIteration:  # Handle case where iterator runs out
                break
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --- NEW: Plotting Function ---
def plot_losses(out_dir, history_iters, history_train_loss, history_val_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(history_iters, history_train_loss, label='Training Loss')
    plt.plot(history_iters, history_val_loss, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(out_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    logger.info(f"Loss plot saved to {plot_path}")


if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)

    # --- NEW: Setup Logging ---
    log_file_path = os.path.join(out_dir, 'training_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    logger = logging.getLogger()
    # --- END OF LOGGING SETUP ---

    logger.info(f"Starting run with profile: {PROFILE}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Using device: {device}")

    # --- Data Loading with Train/Val Split ---
    tokenizer, train_data, val_data = data.load_data(data_path, val_split=0.1)
    train_item_iter = data.RandomOrderDataIterator(train_data, seq_len + 1)
    val_item_iter = data.RandomOrderDataIterator(val_data, seq_len + 1)
    train_batch_iter = iter(data.batch_items(train_item_iter, batch_size))
    val_batch_iter = iter(data.batch_items(val_item_iter, batch_size))
    tokenizer.save(os.path.join(out_dir, 'tokenizer.json'))

    # --- Model and Optimizer Initialization ---
    model_args = dict(n_layers=n_layers, n_heads=n_heads, embed_size=embed_size, max_context_len=seq_len,
                      vocab_size=tokenizer.vocab_size(), mlp_hidden_size=embed_size * 4, dropout_rate=dropout_rate)
    model = TransformerLM(**model_args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=[0.9, 0.95])

    # --- Checkpoint Loading and History Initialization ---
    iter_num, best_val_loss = 0, 1e9
    history_iters, history_train_loss, history_val_loss = [], [], []
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model']);
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']

    # --- Training Loop ---
    model.train()
    while iter_num < num_iters:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        batch_data = next(train_batch_iter).to(device)
        batch_x, batch_y = lm.batch_to_labeled_samples(batch_data)

        logits = model(batch_x)
        loss = lm.compute_loss(logits, batch_y)
        optimizer.zero_grad(set_to_none=True);
        loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping);
        optimizer.step()

        if iter_num % log_interval == 0:
            logger.info(f"Iter {iter_num}/{num_iters}, Train Loss: {loss.item():.4f}, LR: {lr:.6f}")

        if iter_num > 0 and iter_num % eval_interval == 0:
            losses = estimate_loss(model, train_batch_iter, val_batch_iter)
            logger.info(
                f"--- Eval at Iter {iter_num}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f} ---")

            # Append data for plotting
            history_iters.append(iter_num)
            history_train_loss.append(losses['train'])
            history_val_loss.append(losses['val'])

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': model_args,
                            'iter_num': iter_num, 'best_val_loss': best_val_loss}, ckpt_path)
                logger.info(f"Saved new best checkpoint to {ckpt_path}")

            prompt = "The " if PROFILE == 'en' else "שלום"
            continuation = model.better_sample_continuation(tokenizer.tokenize(prompt), 150, sampling_temperature,
                                                            top_k)
            logger.info(f"Sample:\n{tokenizer.detokenize(tokenizer.tokenize(prompt) + continuation)}\n---")

        iter_num += 1

    logger.info("Training finished.")

    # --- Plotting at the end of training ---
    if history_iters:
        plot_losses(out_dir, history_iters, history_train_loss, history_val_loss)