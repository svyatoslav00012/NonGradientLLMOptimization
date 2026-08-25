import torch

from mlp_lib.layers import Embedding, Linear, ReLU
from mlp_lib.ops import manual_cross_entropy, manual_cross_entropy_backward


class MLP:
    """
    Character-level MLP, all forward/backward math implemented manually
    (no autograd): embedding lookup -> N x (Linear -> activation) hidden
    blocks -> output Linear producing raw logits over the vocabulary.

    hidden_sizes=[200] reproduces a single-hidden-layer network.
    Pass more sizes, e.g. [200, 100], for a deeper perceptron.
    activation: a layer class with forward/backward/parameters/grads,
    e.g. ReLU (default) or Tanh from mlp_lib.layers.
    """

    def __init__(self, vocab_size, embed_dim, context_len, hidden_sizes, generator=None, activation=ReLU):
        self.embedding = Embedding(vocab_size, embed_dim, generator=generator)

        self.layers = []
        fan_in = embed_dim * context_len
        for size in hidden_sizes:
            self.layers.append(Linear(fan_in, size, generator=generator))
            self.layers.append(activation())
            fan_in = size

        # Output layer: no activation. Small weight scale keeps initial
        # logits near 0 (uniform-ish softmax) instead of huge, which is
        # what previously caused exp() underflow -> inf loss at init.
        self.layers.append(Linear(fan_in, vocab_size, generator=generator, weight_scale=0.01))

    def forward(self, X):
        out = self.embedding.forward(X)
        for layer in self.layers:
            out = layer.forward(out)
        return out  # logits

    def backward(self, dlogits):
        """Walk the forward graph in reverse, layer by layer."""
        dout = dlogits
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self.embedding.backward(dout)

    def parameters(self):
        params = self.embedding.parameters()
        for layer in self.layers:
            params += layer.parameters()
        return params

    def grads(self):
        grads = self.embedding.grads()
        for layer in self.layers:
            grads += layer.grads()
        return grads

    def step(self, lr):
        """Plain SGD: p -= lr * dp for every parameter."""
        for p, dp in zip(self.parameters(), self.grads()):
            p.data -= lr * dp

    def loss(self, X, Y):
        logits = self.forward(X)
        return manual_cross_entropy(logits, Y)

    def train(self, Xtr, Ytr, epochs, batch_size, lr_schedule, log_every=1000):
        """
        lr_schedule: a float (constant lr) or a callable epoch -> lr.
        """
        loss_history = []
        get_lr = lr_schedule if callable(lr_schedule) else (lambda epoch: lr_schedule)

        for epoch in range(epochs):
            batch_ix = torch.randint(0, Xtr.shape[0], (batch_size,))
            X_batch, Y_batch = Xtr[batch_ix], Ytr[batch_ix]

            logits = self.forward(X_batch)
            loss, probs = manual_cross_entropy(logits, Y_batch)
            loss_history.append(loss.item())

            dlogits = manual_cross_entropy_backward(probs, Y_batch)
            self.backward(dlogits)
            self.step(get_lr(epoch))

            if epoch % log_every == 0:
                print(f"epoch {epoch:6d} loss {loss.item():.4f}")

        return loss_history
