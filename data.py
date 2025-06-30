from __future__ import annotations
from typing import Iterator
import torch
import random
import glob
import json


class CharTokenizer:
    # This class remains unchanged
    def __init__(self):
        self.symbols = ["<PAD>"]
        self.tokens = set()
        self.vocab = list(self.symbols)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}

    def pad_id(self):
        return self.stoi["<PAD>"]

    def get_id(self, tok: str):
        return self.stoi.get(tok)

    def vocab_size(self):
        return len(self.vocab)

    def train(self, text: str) -> None:
        for symbol in self._tokenize_to_symbols(text): self.tokens.add(symbol)
        self.vocab = list(self.symbols) + list(sorted(self.tokens))
        self.stoi = {s: i for i, s in enumerate(self.vocab)}

    def _tokenize_to_symbols(self, text: str) -> list[str]:
        return list(text)

    def tokenize(self, text: str) -> list[int]:
        return [self.stoi.get(s, self.pad_id()) for s in self._tokenize_to_symbols(text)]

    def detokenize(self, tokens: list[int], keep_symbols=True) -> str:
        strs = [self.vocab[t] for t in tokens]
        if not keep_symbols: strs = [s for s in strs if s not in self.symbols]
        return "".join(strs)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f: json.dump({'vocab': self.vocab}, f)

    @staticmethod
    def load(path: str) -> CharTokenizer:
        tokenizer = CharTokenizer()
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        tokenizer.vocab, tokenizer.stoi = data['vocab'], {s: i for i, s in enumerate(data['vocab'])}
        tokenizer.tokens = set(t for t in tokenizer.vocab if t not in tokenizer.symbols)
        return tokenizer


class RandomOrderDataIterator:
    # This class remains unchanged
    def __init__(self, data, desired_length):
        self.desired_length = desired_length
        self.data: list[list[int]] = [seq for seq in data if len(seq) > self.desired_length]

    def __iter__(self):
        if not self.data: return
        while True:
            seq = random.choice(self.data)
            idx = random.randint(0, len(seq) - self.desired_length)
            yield seq[idx:idx + self.desired_length]


# --- MODIFIED FUNCTION ---
def load_data(path: str, val_split: float = 0.1) -> [CharTokenizer, list[list[int]], list[list[int]]]:
    """
    Loads data and splits it into training and validation sets.
    Returns: (tokenizer, train_data, val_data)
    """
    tokenizer = CharTokenizer()
    all_text = ""
    for fname in glob.glob(f"{path}/*.txt"):
        with open(fname, encoding='utf-8') as fh:
            all_text += fh.read()

    tokenizer.train(all_text)

    # Tokenize the entire corpus
    tokenized_corpus = tokenizer.tokenize(all_text)

    # Split the tokenized data
    split_idx = int(len(tokenized_corpus) * (1 - val_split))
    train_data = [tokenized_corpus[:split_idx]]
    val_data = [tokenized_corpus[split_idx:]]

    print(
        f"Data loaded. Total tokens: {len(tokenized_corpus)}. Train tokens: {len(train_data[0])}. Val tokens: {len(val_data[0])}.")
    return tokenizer, train_data, val_data


# --- END OF MODIFICATION ---

def batch_items(data_iter: Iterator[list[int]], batch_size: int = 2) -> Iterator[torch.LongTensor]:
    # This function remains unchanged
    batch = []
    for seq in data_iter:
        batch.append(seq)
        if len(batch) >= batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
    if len(batch) > 0:
        yield torch.tensor(batch, dtype=torch.long)