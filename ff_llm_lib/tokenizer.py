"""Char-level and byte-level BPE tokenizers with a shared interface:
.vocab_size, .encode(text) -> tensor, .decode(ids) -> str.
"""
import os

import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from ff_llm_lib.text_data import build_char_vocab
from ff_llm_lib.text_data import decode as char_decode
from ff_llm_lib.text_data import encode as char_encode


class CharTokenizer:
    def __init__(self, text):
        self.stoi, self.itos = build_char_vocab(text)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return char_encode(text, self.stoi)

    def decode(self, ids):
        return char_decode(ids, self.itos)


class BpeTokenizer:
    """Byte-level BPE trained on the corpus itself (optionally cached to `cache_path`)."""

    def __init__(self, text, vocab_size=4096, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            self._tok = HFTokenizer.from_file(cache_path)
            return

        tok = HFTokenizer(BPE(unk_token=None))
        tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, show_progress=False)
        tok.train_from_iterator([text], trainer=trainer)
        if cache_path:
            tok.save(cache_path)
        self._tok = tok

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return torch.tensor(self._tok.encode(text).ids, dtype=torch.long)

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self._tok.decode(ids)


def build_tokenizer(text, config):
    kind = config["kind"]
    if kind == "char":
        return CharTokenizer(text)
    if kind == "bpe":
        return BpeTokenizer(text, vocab_size=config.get("vocab_size", 4096), cache_path=config.get("cache_path"))
    raise ValueError(f"unknown tokenizer kind: {kind!r}")
