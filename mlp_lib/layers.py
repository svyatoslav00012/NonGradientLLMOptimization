import torch

from mlp_lib.ops import manual_relu


class Embedding:
    """
    Lookup table C: (vocab_size, embed_dim). Forward indexes one row per
    context position and flattens the window into a single vector per
    example: (B, context_len) -> (B, context_len * embed_dim).
    """

    def __init__(self, vocab_size, embed_dim, generator=None):
        self.C = torch.randn((vocab_size, embed_dim), generator=generator)
        self._input_ix = None
        self.dC = None

    def forward(self, ix):
        self._input_ix = ix                   # (B, context_len)
        emb = self.C[ix]                      # (B, context_len, embed_dim)
        return emb.view(ix.shape[0], -1)      # (B, context_len * embed_dim)

    def backward(self, dout):
        """
        Undo the flatten, then scatter-add each row's gradient back into
        the embedding rows that were actually looked up. Must accumulate
        (index_add_), not overwrite: the same character can appear more
        than once across a batch/context window.
        """
        B, context_len = self._input_ix.shape
        dout = dout.view(B, context_len, -1)

        self.dC = torch.zeros_like(self.C)
        self.dC.index_add_(0, self._input_ix.reshape(-1), dout.reshape(-1, dout.shape[-1]))

    def parameters(self):
        return [self.C]

    def grads(self):
        return [self.dC]


class Linear:
    """y = x @ W + b"""

    def __init__(self, fan_in, fan_out, generator=None, weight_scale=None):
        # Default: Kaiming init for a ReLU-fed layer, std = sqrt(2 / fan_in).
        scale = weight_scale if weight_scale is not None else (2 / fan_in) ** 0.5
        self.W = torch.randn((fan_in, fan_out), generator=generator) * scale
        self.b = torch.randn(fan_out, generator=generator) * 0.01
        self._input = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self._input = x
        return x @ self.W + self.b

    def backward(self, dout):
        """
        y = x @ W + b, so:
          dW = x^T @ dout   (sums the outer product over the batch dim)
          db = sum_b dout   (bias is added to every row)
          dx = dout @ W^T   (gradient pushed back to this layer's input)
        """
        self.dW = self._input.T @ dout
        self.db = dout.sum(0)
        return dout @ self.W.T

    def parameters(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU:
    """h = max(z, 0). Local derivative is 1 where active, 0 where clipped."""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = (x > 0).float()
        return manual_relu(x)

    def backward(self, dout):
        return dout * self._mask

    def parameters(self):
        return []

    def grads(self):
        return []


class Tanh:
    """h = tanh(z). d(tanh(z))/dz = 1 - tanh(z)^2 = 1 - h^2."""

    def __init__(self):
        self._out = None

    def forward(self, x):
        self._out = torch.tanh(x)
        return self._out

    def backward(self, dout):
        return dout * (1 - self._out ** 2)

    def parameters(self):
        return []

    def grads(self):
        return []
