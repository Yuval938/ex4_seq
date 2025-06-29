import torch

from transformer import Embed, TransformerDecoderBlock, TransformerLM
from lm import batch_to_labeled_samples, compute_loss


def test_lm_functions():
    print("Running test_lm_functions...")
    seq_len, batch_size = 10, 4
    batch = torch.randint(1, 100, (batch_size, seq_len + 1))

    inputs, labels = batch_to_labeled_samples(batch)

    assert inputs.shape == (batch_size, seq_len)
    assert labels.shape == (batch_size, seq_len)
    assert torch.equal(inputs, batch[:, :-1])
    assert torch.equal(labels, batch[:, 1:])
    print("batch_to_labeled_samples PASSED!")

    vocab_size = 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels[0, 0] = 0

    loss = compute_loss(logits, labels)
    assert loss.item() > 0 and not torch.isnan(loss).any()
    print("compute_loss PASSED!")


def test_model_components():
    print("Running test_model_components...")
    vocab_size, embed_size, n_heads, max_context_len = 100, 32, 4, 50
    dropout_rate = 0.1
    batch_size, seq_len = 4, 20

    embed_layer = Embed(vocab_size, embed_size, max_context_len, dropout_rate)
    input_indices = torch.randint(1, vocab_size, (batch_size, seq_len))
    output = embed_layer(input_indices)
    assert output.shape == (batch_size, seq_len, embed_size)
    print("Embed layer PASSED!")

    block = TransformerDecoderBlock(n_heads, embed_size, embed_size * 4, dropout_rate)
    input_tensor = torch.randn(batch_size, seq_len, embed_size)
    output = block(input_tensor)
    assert output.shape == (batch_size, seq_len, embed_size)
    assert not torch.allclose(output, input_tensor)
    print("TransformerDecoderBlock PASSED!")


def test_full_model_pass():
    print("Running test_full_model_pass...")
    vocab_size, embed_size, n_layers, n_heads, max_context_len = 100, 32, 2, 4, 50
    dropout_rate = 0.1
    batch_size, seq_len = 4, 20

    model = TransformerLM(n_layers, n_heads, embed_size, max_context_len, vocab_size, embed_size * 4, dropout_rate)
    input_indices = torch.randint(1, vocab_size, (batch_size, seq_len))
    logits = model(input_indices)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("test_full_model_pass PASSED!")


if __name__ == "__main__":
    test_lm_functions()
    print("-" * 40)
    test_model_components()
    print("-" * 40)
    test_full_model_pass()
    print("\nAll tests passed!")