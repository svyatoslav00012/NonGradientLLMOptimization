"""Forward-Forward (Hinton, 2022) for next-token prediction.

For every position t, K negative tokens run as side queries next to the real
sequence. A side query attends to the real prefix x_<t and to itself only, so
a positive/negative pair differs only in the token at t.

Goodness is the mean square of each block's update (out - in). Per-block loss
is InfoNCE over log-goodness (loss_kind="nce") or Hinton's threshold loss
(loss_kind="hinton"). Negatives come from a bigram model with the true token
excluded; n_mine > neg_k keeps the candidates the current model scores highest.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_llm_lib.local_learning import (
    BlockStack, block_param_groups, clip_per_group, eval_generator, make_optimizer, run_local_training,
)
from ff_llm_lib.model import apply_rope, rotate_half
from ff_llm_lib.text_data import get_batch


class FFModel(nn.Module):
    def __init__(self, config, alpha=1.0):
        super().__init__()
        self.config = config
        self.stack = BlockStack(config)
        self.alpha = alpha
        V = config.vocab_size
        # row V: unigram, used for position 0
        self.register_buffer("bigram", torch.full((V + 1, V), 1.0 / V))
        self.register_buffer("score_temp", torch.tensor(1.0))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def fit_bigram(self, data, max_tokens=5_000_000, smoothing=1.0):
        V = self.config.vocab_size
        d = data[:max_tokens].long()
        counts = torch.full((V + 1, V), smoothing)
        counts[:V].index_put_((d[:-1], d[1:]), torch.ones(len(d) - 1), accumulate=True)
        counts[V] += torch.bincount(d, minlength=V).float()
        self.bigram.copy_((counts / counts.sum(-1, keepdim=True)).to(self.bigram.device))

    def sample_negatives(self, x, side_pos, k):
        """(B, P, k) tokens ~ bigram(. | x_{t-1}) with the true x_t excluded."""
        V = self.config.vocab_size
        B = x.shape[0]
        prev = torch.where(side_pos > 0, x[:, (side_pos - 1).clamp(min=0)], torch.full_like(x[:, side_pos], V))
        probs = self.bigram[prev].clone()
        probs.scatter_(-1, x[:, side_pos].unsqueeze(-1), 0.0)
        return torch.multinomial(probs.view(-1, V), k, replacement=True).view(B, len(side_pos), k)

    def embed_side(self, cand, side_pos):
        s = self.stack.token_emb(cand)
        if self.stack.pos_emb is not None:
            s = s + self.stack.pos_emb(side_pos)[None, :, None, :]
        return s


def block_forward_with_side(block, h, s=None, side_pos=None):
    """Run a Block on the real stream h (B, T, C) and side queries s (B, P, K, C) at side_pos (P,).

    Side query (p, k) attends to real keys j < side_pos[p] and to its own key.
    No attention dropout.
    """
    attn = block.attn
    B, T, C = h.shape
    nh, hd = attn.n_head, attn.head_dim

    q, k, v = attn.qkv_proj(block.norm1(h)).split(C, dim=-1)
    q, k, v = (t.view(B, T, nh, hd).transpose(1, 2) for t in (q, k, v))
    if attn.use_rope:
        q, k = apply_rope(q, k, attn.rope_cos[:T], attn.rope_sin[:T])
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h1 = h + attn.out_proj(y.transpose(1, 2).reshape(B, T, C))
    h_out = h1 + block.mlp(block.norm2(h1))
    if s is None:
        return h_out, None

    P, K = s.shape[1], s.shape[2]
    qs, ks, vs = attn.qkv_proj(block.norm1(s)).split(C, dim=-1)
    qs, ks, vs = (t.view(B, P, K, nh, hd).permute(0, 3, 1, 2, 4) for t in (qs, ks, vs))
    if attn.use_rope:
        cos = attn.rope_cos[side_pos][None, None, :, None, :]
        sin = attn.rope_sin[side_pos][None, None, :, None, :]
        qs, ks = qs * cos + rotate_half(qs) * sin, ks * cos + rotate_half(ks) * sin
    scale = hd ** -0.5
    sc_real = torch.einsum("bhpkd,bhjd->bhpkj", qs, k) * scale
    visible = torch.arange(T, device=h.device)[None, :] < side_pos[:, None]
    sc_real = sc_real.masked_fill(~visible[None, None, :, None, :], float("-inf"))
    sc_self = (qs * ks).sum(-1, keepdim=True) * scale
    att = torch.softmax(torch.cat([sc_real, sc_self], dim=-1), dim=-1)
    ys = torch.einsum("bhpkj,bhjd->bhpkd", att[..., :T], v) + att[..., T:] * vs
    s1 = s + attn.out_proj(ys.permute(0, 2, 3, 1, 4).reshape(B, P, K, C))
    s_out = s1 + block.mlp(block.norm2(s1))
    return h_out, s_out


def _goodness(out, inp):
    return (out - inp).pow(2).mean(dim=-1)


def _block_loss(g_pos, g_neg, alpha, loss_kind, threshold):
    """g_pos (B, P), g_neg (B, P, K)."""
    if loss_kind == "hinton":
        return F.softplus(threshold - g_pos).mean() + F.softplus(g_neg - threshold).mean()
    logits = alpha * torch.log(torch.cat([g_pos.unsqueeze(-1), g_neg], dim=-1) + 1e-8)
    return -F.log_softmax(logits, dim=-1)[..., 0].mean()


def _score_from_goodness(g, alpha, loss_kind):
    return g if loss_kind == "hinton" else alpha * torch.log(g + 1e-8)


@torch.no_grad()
def score_candidates(model, x, cand, side_pos, loss_kind="nce", batch_chunk=8, cand_chunk=32):
    """Total FF score (B, P, K) of candidate tokens cand (B, P, K) at side_pos."""
    was_training = model.training
    model.eval()
    B, P, K = cand.shape
    out = torch.empty(B, P, K, device=x.device)
    for b0 in range(0, B, batch_chunk):
        xb = x[b0:b0 + batch_chunk]
        for c0 in range(0, K, cand_chunk):
            cb = cand[b0:b0 + batch_chunk, :, c0:c0 + cand_chunk]
            h, s = model.stack.embed(xb), model.embed_side(cb, side_pos)
            total = torch.zeros(cb.shape, device=x.device)
            for block in model.stack.blocks:
                h_new, s_new = block_forward_with_side(block, h, s, side_pos)
                total += _score_from_goodness(_goodness(s_new, s), model.alpha, loss_kind)
                h, s = h_new, s_new
            out[b0:b0 + batch_chunk, :, c0:c0 + cand_chunk] = total
    model.train(was_training)
    return out


def ff_train_step(model, optimizer, x, neg_k, n_mine, loss_kind, threshold, grad_clip):
    T = x.shape[1]
    side_pos = torch.arange(1, T, device=x.device)
    if n_mine and n_mine > neg_k:
        cand = model.sample_negatives(x, side_pos, n_mine)
        scores = score_candidates(model, x, cand, side_pos, loss_kind)
        neg = cand.gather(-1, scores.topk(neg_k, dim=-1).indices)
    else:
        neg = model.sample_negatives(x, side_pos, neg_k)

    optimizer.zero_grad(set_to_none=True)
    h, s = model.stack.embed(x), model.embed_side(neg, side_pos)
    losses = []
    for block in model.stack.blocks:
        h_out, s_out = block_forward_with_side(block, h, s, side_pos)
        g_pos = _goodness(h_out, h)[:, side_pos]
        g_neg = _goodness(s_out, s)
        loss = _block_loss(g_pos, g_neg, model.alpha, loss_kind, threshold)
        loss.backward()
        losses.append(loss.detach())
        h, s = h_out.detach(), s_out.detach()
    clip_per_group(optimizer, grad_clip)
    optimizer.step()
    return torch.stack(losses).mean()


def _fit_temperature(scores, targets):
    """Grid search for the temperature minimizing CE of softmax(scores / T)."""
    best_t, best_ce = 1.0, float("inf")
    for t in torch.logspace(-2, 2, 41).tolist():
        ce = F.cross_entropy(scores / t, targets).item()
        if ce < best_ce:
            best_t, best_ce = t, ce
    return best_t


@torch.no_grad()
def ff_evaluate(model, data, block_size, device, n_windows=16, neg_k=4, loss_kind="nce", threshold=2.0,
                generator=None):
    """dev_loss: FF loss with bigram negatives.
    cross_entropy: next-token CE over all V candidates' scores, with a temperature
    fit on the other half of the windows. accuracy: top-1 over all V candidates.
    """
    V = model.config.vocab_size
    x, _ = get_batch(data, block_size, n_windows, generator=generator)
    x = x.to(device)
    T = x.shape[1]
    side_pos = torch.arange(1, T, device=device)

    model.eval()
    h, s = model.stack.embed(x), model.embed_side(model.sample_negatives(x, side_pos, neg_k), side_pos)
    dev_losses = []
    for block in model.stack.blocks:
        h_out, s_out = block_forward_with_side(block, h, s, side_pos)
        dev_losses.append(_block_loss(_goodness(h_out, h)[:, side_pos], _goodness(s_out, s),
                                      model.alpha, loss_kind, threshold).item())
        h, s = h_out, s_out

    cand = torch.arange(V, device=device).expand(n_windows, T - 1, V)
    scores = score_candidates(model, x, cand, side_pos, loss_kind).reshape(-1, V)
    targets = x[:, 1:].reshape(-1)
    half = scores.shape[0] // 2
    temp = _fit_temperature(scores[:half], targets[:half])
    model.score_temp.fill_(temp)
    ce = F.cross_entropy(scores[half:] / temp, targets[half:]).item()
    acc = (scores.argmax(-1) == targets).float().mean().item()
    model.train()
    return {"dev_loss": sum(dev_losses) / len(dev_losses), "cross_entropy": ce, "accuracy": acc,
            "temperature": float(temp)}


def train_ff(model, train_data, val_data, block_size, device, steps=1000, batch_size=64,
             max_lr=2e-3, warmup_steps=200, weight_decay=0.01, grad_clip=1.0,
             neg_k=2, n_mine=0, loss_kind="nce", threshold=2.0,
             eval_every=200, eval_windows=16, generator=None, eval_seed=None):
    model.to(device)
    model.fit_bigram(train_data)
    groups = block_param_groups(model.stack)
    optimizer, scheduler = make_optimizer(groups, steps, max_lr, warmup_steps, weight_decay)
    return run_local_training(
        step_fn=lambda x, y: ff_train_step(model, optimizer, x, neg_k, n_mine, loss_kind, threshold, grad_clip),
        eval_fn=lambda: ff_evaluate(model, val_data, block_size, device, n_windows=eval_windows,
                                    loss_kind=loss_kind, threshold=threshold,
                                    generator=eval_generator(generator, eval_seed)),
        optimizer=optimizer, scheduler=scheduler, train_data=train_data, block_size=block_size,
        batch_size=batch_size, steps=steps, eval_every=eval_every, device=device, generator=generator,
    )


@torch.no_grad()
def generate_ff(model, tokenizer, device, prompt="\n", max_new_tokens=200, temperature=1.0, top_k=None,
                loss_kind="nce", generator=None):
    """Each new token: score all V candidates as side queries at the next position."""
    V = model.config.vocab_size
    block_size = model.config.block_size
    idx = tokenizer.encode(prompt).long().to(device)
    cand = torch.arange(V, device=device).view(1, 1, V)
    for _ in range(max_new_tokens):
        ctx = idx[-(block_size - 1):]
        x = torch.cat([ctx, ctx.new_zeros(1)]).unsqueeze(0)  # placeholder at the scored position
        side_pos = torch.tensor([ctx.shape[0]], device=device)
        logits = score_candidates(model, x, cand, side_pos, loss_kind)[0, 0] / (model.score_temp * temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, V))
            logits[logits < v[-1]] = -float("inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        idx = torch.cat([idx, next_id])
    return tokenizer.decode(idx)
