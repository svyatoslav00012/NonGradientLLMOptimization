import random

import torch


def load_words(path="names.txt"):
    return open(path).read().splitlines()


def build_vocab(words):
    """stoi/itos mapping, with '.' reserved as index 0 (start/end token)."""
    chars = sorted(list(set(''.join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, context_len):
    """
    Slides a window of `context_len` previous characters across each word
    (padded with '.' at the start, terminated with '.') to build
    (context -> next char) training pairs.
    """
    X, Y = [], []
    for w in words:
        context = [stoi['.']] * context_len
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def train_dev_test_split(words, train_frac=0.8, dev_frac=0.1, seed=42):
    rng = random.Random(seed)
    shuffled = words[:]
    rng.shuffle(shuffled)
    n1 = int(train_frac * len(shuffled))
    n2 = int((train_frac + dev_frac) * len(shuffled))
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]
