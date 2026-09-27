import numpy as np
import torch


def take_fraction(text, frac):
    return text[: int(len(text) * frac)]


def build_char_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    """Vectorized char -> id encoding into a uint8 tensor (vocab must be <= 256)."""
    assert len(stoi) <= 256, f"encode() packs ids into a byte; vocab_size={len(stoi)} is too large"
    table = {ord(ch): chr(idx) for ch, idx in stoi.items()}
    raw = text.translate(table).encode('latin-1')
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())


def decode(ids, itos):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return ''.join(itos[i] for i in ids)


def train_val_split(data, train_frac=0.9):
    """Split a 1D token stream by position (no shuffling)."""
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size, generator=None):
    """Random contiguous windows: x = data[i:i+T], y = data[i+1:i+T+1]."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).long()
    return x, y
