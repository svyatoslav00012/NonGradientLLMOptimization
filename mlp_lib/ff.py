import torch

from mlp_lib.ops import manual_cross_entropy, manual_softmax


def manual_softplus(x):
    """log(1 + exp(x)), computed in a numerically stable way (no overflow
    for large positive/negative x)."""
    return torch.clamp(x, min=0) + torch.log1p(torch.exp(-torch.abs(x)))


def normalize_rows(h, eps=1e-4):
    """
    Divide each row by its own L2 norm. Used between FF layers so the next
    layer only sees the *direction* of the previous layer's activity, not
    its magnitude (goodness = squared magnitude) -- otherwise a layer could
    "cheat" by just passing along a big/small vector instead of learning
    anything, and the next layer's goodness would trivially inherit it.
    """
    return h / (h.norm(dim=1, keepdim=True) + eps)


def make_ff_batch(X_batch, Y_batch, vocab_size, generator=None):
    """
    Builds positive and negative token-id sequences for one training step.

    Positive sequence = context followed by the TRUE next char.
    Negative sequence = context followed by a uniformly random WRONG char
    (never the true one -- the "rand >= Y" shift skips over it so every
    other vocab entry stays equally likely).
    """
    pos_ids = torch.cat([X_batch, Y_batch.unsqueeze(1)], dim=1)

    rand = torch.randint(0, vocab_size - 1, (X_batch.shape[0],), generator=generator)
    neg_labels = rand + (rand >= Y_batch).long()
    neg_ids = torch.cat([X_batch, neg_labels.unsqueeze(1)], dim=1)

    return pos_ids, neg_ids


class FFNet:
    """
    Forward-Forward network for next-char prediction.

    Instead of "context -> logits over next char" (backprop's setup), FF
    reframes the problem as "context + candidate next char -> is this a
    real (context, next-char) pair or a fake one?". The candidate char is
    embedded as an extra token appended to the context, so every input
    sequence has length (context_len + 1).

    Each hidden layer is trained by its OWN local objective (goodness),
    with no error signal flowing backward across layers -- that's the
    entire point of FF: it replaces global backprop with local learning.
    """

    def __init__(self, vocab_size, embed_dim, context_len, hidden_sizes, generator=None, threshold=4.0):
        self.C = torch.randn((vocab_size, embed_dim), generator=generator)
        self.vocab_size = vocab_size
        self.context_len = context_len          # length of the context alone (label token added on top)
        self.threshold = threshold               # goodness "decision boundary" between real/fake

        self.blocks = []                         # each block: {'W': ..., 'b': ...}
        fan_in = embed_dim * (context_len + 1)   # +1 for the appended candidate-char token
        for size in hidden_sizes:
            W = torch.randn((fan_in, size), generator=generator) * (2 / fan_in) ** 0.5
            b = torch.randn(size, generator=generator) * 0.01
            self.blocks.append({'W': W, 'b': b})
            fan_in = size

    def embed(self, ids):
        """ids: (B, context_len + 1) -> flattened embedding (B, (context_len+1)*embed_dim)."""
        emb = self.C[ids]
        return emb.view(ids.shape[0], -1)

    def block_forward(self, x, block):
        z = x @ block['W'] + block['b']
        h = torch.tanh(z)
        return z, h

    def goodness(self, h):
        """Sum of squared activations per example -- the FF 'confidence' signal."""
        return (h ** 2).sum(dim=1)

    def total_goodness(self, ids):
        """Sums goodness across all hidden blocks -- used at inference time
        to rank candidate next chars (higher total goodness = more 'real')."""
        x = self.embed(ids)
        total = torch.zeros(ids.shape[0])
        for block in self.blocks:
            z, h = self.block_forward(x, block)
            total = total + self.goodness(h)
            x = normalize_rows(h)
        return total

    def _local_grad(self, x, z, h, W, y):
        """
        Local gradient of one block's goodness-based loss w.r.t. its own
        W, b, and its input x -- NOT w.r.t. any earlier block's weights.

        loss = softplus(-y * (goodness - threshold))
          y = +1 for a positive (real) example  -> loss shrinks as goodness rises
          y = -1 for a negative (fake) example   -> loss shrinks as goodness falls

        Chain rule, entirely local to this block:
          dL/dg = -y * (1 - sigmoid(y*(g - threshold)))      (g = goodness)
          dg/dh = 2h                                          (g = sum(h^2))
          dh/dz = 1 - tanh(z)^2 = 1 - h^2
          dL/dz = dL/dg * dg/dh * dh/dz
          dW = x.T @ dL/dz / B ,  db = mean(dL/dz) ,  dx = dL/dz @ W.T
        `dx` is only used to also route a gradient into the embedding
        table when this is the first block (see FFNet.train_step).
        """
        g = self.goodness(h)
        s = torch.sigmoid(y * (g - self.threshold))
        dL_dg = -y * (1 - s)
        dL_dh = dL_dg.unsqueeze(1) * (2 * h)
        dh_dz = 1 - h ** 2
        dL_dz = dL_dh * dh_dz

        B = x.shape[0]
        dW = x.T @ dL_dz / B
        db = dL_dz.mean(0)
        dx = dL_dz @ W.T
        loss = manual_softplus(-y * (g - self.threshold)).mean()

        return dW, db, dx, loss

    def train_step(self, pos_ids, neg_ids, lr):
        """
        One FF training step: forward the positive and negative sequences
        through every block, updating each block IMMEDIATELY from its own
        local loss before the (already-updated) activations are normalized
        and handed to the next block. No global loss, no cross-layer
        gradient -- each block only ever sees its own goodness objective.
        """
        x_pos = self.embed(pos_ids)
        x_neg = self.embed(neg_ids)

        demb = torch.zeros_like(self.C)
        losses, goodness_pos, goodness_neg = [], [], []

        for i, block in enumerate(self.blocks):
            z_pos, h_pos = self.block_forward(x_pos, block)
            z_neg, h_neg = self.block_forward(x_neg, block)

            dW_pos, db_pos, dx_pos, loss_pos = self._local_grad(x_pos, z_pos, h_pos, block['W'], y=1.0)
            dW_neg, db_neg, dx_neg, loss_neg = self._local_grad(x_neg, z_neg, h_neg, block['W'], y=-1.0)

            block['W'] -= lr * (dW_pos + dW_neg)
            block['b'] -= lr * (db_pos + db_neg)

            if i == 0:
                # Route the local gradient one step further back, into the
                # embedding rows that were looked up -- embedding + block 0
                # are trained together as a single local unit.
                B, n_tokens = pos_ids.shape
                dx_pos_r = dx_pos.view(B, n_tokens, -1)
                dx_neg_r = dx_neg.view(B, n_tokens, -1)
                demb.index_add_(0, pos_ids.reshape(-1), dx_pos_r.reshape(-1, dx_pos_r.shape[-1]))
                demb.index_add_(0, neg_ids.reshape(-1), dx_neg_r.reshape(-1, dx_neg_r.shape[-1]))

            losses.append(((loss_pos + loss_neg) / 2).item())
            goodness_pos.append(self.goodness(h_pos).mean().item())
            goodness_neg.append(self.goodness(h_neg).mean().item())

            # Detach-by-construction: x_pos/x_neg for the NEXT block are
            # just numbers derived from this block's (now-updated-weight)
            # forward pass -- no gradient ever flows back into this block
            # from later blocks, because later blocks never see z_pos/h_pos.
            x_pos = normalize_rows(h_pos)
            x_neg = normalize_rows(h_neg)

        self.C -= lr * demb

        return losses, goodness_pos, goodness_neg

    def train(self, Xtr, Ytr, epochs, batch_size, lr, log_every=1000, generator=None,
              eval_sets=None, eval_every=None):
        """
        eval_sets: optional dict {name: (X, Y)} (e.g. {'train': (...), 'dev': (...)}).
        If given, cross_entropy_readout is computed on each every `eval_every`
        epochs (default: log_every) and tracked in history['ce'][name] /
        history['ce_epochs'] -- this is the comparable-to-backprop metric,
        distinct from history['loss'] (the local goodness-margin loss, a
        different, non-comparable scale).
        """
        history = {'loss': [], 'goodness_pos': [], 'goodness_neg': [], 'ce_epochs': [], 'ce': {}}
        if eval_sets:
            history['ce'] = {name: [] for name in eval_sets}
        eval_every = eval_every or log_every

        for epoch in range(epochs):
            batch_ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=generator)
            X_batch, Y_batch = Xtr[batch_ix], Ytr[batch_ix]

            pos_ids, neg_ids = make_ff_batch(X_batch, Y_batch, self.vocab_size, generator=generator)
            losses, g_pos, g_neg = self.train_step(pos_ids, neg_ids, lr)

            history['loss'].append(sum(losses) / len(losses))
            history['goodness_pos'].append(sum(g_pos) / len(g_pos))
            history['goodness_neg'].append(sum(g_neg) / len(g_neg))

            if eval_sets and epoch % eval_every == 0:
                history['ce_epochs'].append(epoch)
                for name, (X, Y) in eval_sets.items():
                    ce_loss, _ = self.cross_entropy_readout(X, Y)
                    history['ce'][name].append(ce_loss.item())

            if epoch % log_every == 0:
                print(f"epoch {epoch:6d}  local_loss {history['loss'][-1]:.4f}  "
                      f"goodness+ {history['goodness_pos'][-1]:.2f}  goodness- {history['goodness_neg'][-1]:.2f}")

        return history

    def cross_entropy_readout(self, X, Y):
        """
        FF has no softmax output -- to get a number comparable to backprop's
        cross-entropy, score EVERY candidate next char by total goodness
        (one forward pass per candidate) and treat those goodness scores as
        unnormalized logits, then run the usual softmax + NLL on them.
        """
        B = X.shape[0]
        V = self.vocab_size

        X_expanded = X.unsqueeze(1).expand(B, V, self.context_len).reshape(B * V, self.context_len)
        candidates = torch.arange(V).unsqueeze(0).expand(B, V).reshape(B * V, 1)
        ids_flat = torch.cat([X_expanded, candidates], dim=1)

        totals = self.total_goodness(ids_flat).view(B, V)
        loss, probs = manual_cross_entropy(totals, Y)
        return loss, probs
