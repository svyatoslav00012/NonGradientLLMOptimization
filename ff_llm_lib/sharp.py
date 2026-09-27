"""SHARP: shared-readout residual propagation.

Each block group g is trained locally with next-token CE through one shared
readout (the tied token embedding) plus a per-exit norm. Inputs are detached,
so group g only learns its residual update.

Optionally mixes in the skip-path gradient dL/dh_L of the final loss:
    grad at h_g = (1 - beta) * dCE_g/dh_g + beta * dL/dh_L
group_size: backprop only inside groups of consecutive blocks.
group_size = n_layer with beta = 0 is exactly backprop.
lens_negatives: sampled softmax at non-final exits (true token + top-k of the
final head + random tokens), for large vocabularies.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_llm_lib.local_learning import (
    BlockStack, block_param_groups, clip_per_group, eval_ce_all_positions, eval_generator, make_optimizer,
    run_local_training,
)
from ff_llm_lib.model import build_norm
from ff_llm_lib.text_data import get_batch


class SharpModel(nn.Module):
    def __init__(self, config, group_size=1, beta=0.0, readout_grad="all", lens_negatives=None,
                 n_random_negatives=8):
        super().__init__()
        assert readout_grad in ("all", "final")
        self.config = config
        self.stack = BlockStack(config)
        L = config.n_layer
        self.groups = [list(range(i, min(i + group_size, L))) for i in range(0, L, group_size)]
        # one norm per exit; the last one acts as the final norm
        self.lens_norms = nn.ModuleList([build_norm(config.use_rmsnorm, config.n_embd) for _ in self.groups])
        self.beta = beta
        self.readout_grad = readout_grad
        self.lens_negatives = lens_negatives
        self.n_random_negatives = n_random_negatives

    @property
    def n_exits(self):
        return len(self.groups)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def readout_weight(self, g):
        W = self.stack.token_emb.weight
        if self.readout_grad == "final" and g < self.n_exits - 1:
            W = W.detach()
        return W

    def lens_logits(self, h, g):
        return self.lens_norms[g](h) @ self.readout_weight(g).T

    def run_group(self, h, g):
        for i in self.groups[g]:
            h = self.stack.blocks[i](h)
        return h

    @torch.no_grad()
    def exit_logits(self, x, up_to=None):
        up_to = self.n_exits - 1 if up_to is None else up_to
        h = self.stack.embed(x)
        out = []
        for g in range(up_to + 1):
            h = self.run_group(h, g)
            out.append(self.lens_logits(h, g))
        return out

    @torch.no_grad()
    def logits(self, x, exit_group=None):
        exit_group = self.n_exits - 1 if exit_group is None else exit_group
        h = self.stack.embed(x)
        for g in range(exit_group + 1):
            h = self.run_group(h, g)
        return self.lens_logits(h, exit_group)

    def exit_depth(self, g):
        return (self.groups[g][-1] + 1) / self.config.n_layer


def _sampled_lens_ce(model, h, g, y, hard):
    V = model.config.vocab_size
    B, T = y.shape
    rand = torch.randint(0, V, (B, T, model.n_random_negatives), device=y.device)
    cand = torch.cat([y.unsqueeze(-1), hard, rand], dim=-1)
    z = model.lens_norms[g](h)
    logits = torch.einsum("btc,btkc->btk", z, model.readout_weight(g)[cand])
    dup = (cand[..., 1:] == y.unsqueeze(-1))
    logits[..., 1:] = logits[..., 1:].masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), torch.zeros(B * T, dtype=torch.long, device=y.device))


def _full_lens_ce(model, h, g, y):
    logits = model.lens_logits(h, g)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def sharp_train_step(model, optimizer, x, y, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    G = model.n_exits
    needs_global = model.beta > 0 or model.lens_negatives
    losses = []

    if not needs_global:
        h = model.stack.embed(x)
        for g in range(G):
            h = model.run_group(h if g == 0 else h.detach(), g)
            loss = _full_lens_ce(model, h, g, y)
            loss.backward()  # frees this group's graph right away
            losses.append(loss.detach())
        clip_per_group(optimizer, grad_clip)
        optimizer.step()
        return losses[-1]

    # pass 1 (no grad): group-boundary states and dL/dh_L
    was_training = model.training
    model.eval()
    with torch.no_grad():
        bounds = [model.stack.embed(x)]
        for g in range(G):
            bounds.append(model.run_group(bounds[-1], g))
    model.train(was_training)
    hL = bounds[-1].detach().requires_grad_(True)
    with torch.enable_grad():
        logits_L = model.lens_logits(hL, G - 1)
        ce_L = F.cross_entropy(logits_L.reshape(-1, logits_L.size(-1)), y.reshape(-1))
        g_L = torch.autograd.grad(ce_L, hL)[0]
    hard = None
    if model.lens_negatives:
        hard = logits_L.detach().topk(model.lens_negatives, dim=-1).indices

    # pass 2: recompute each group with grad, one at a time
    for g in range(G):
        h_in = model.stack.embed(x) if g == 0 else bounds[g]
        h = model.run_group(h_in, g)
        if g == G - 1:
            loss = _full_lens_ce(model, h, g, y)  # final exit: local loss is the true loss
            total = loss
        else:
            loss = _sampled_lens_ce(model, h, g, y, hard) if hard is not None else _full_lens_ce(model, h, g, y)
            total = (1 - model.beta) * loss
            if model.beta > 0:
                total = total + model.beta * (h * g_L).sum()
        total.backward()
        losses.append(loss.detach())
    clip_per_group(optimizer, grad_clip)
    optimizer.step()
    return losses[-1]


@torch.no_grad()
def evaluate(model, data, block_size, device, n_windows=64, generator=None):
    """cross_entropy/accuracy: final exit. dev_loss: mean CE over exits. ce_exit_g: each exit."""
    model.eval()
    per_exit = eval_ce_all_positions(model.exit_logits, data, block_size, device, n_windows=n_windows,
                                     generator=generator)
    model.train()
    out = {"dev_loss": sum(c for c, _ in per_exit) / len(per_exit),
           "cross_entropy": per_exit[-1][0], "accuracy": per_exit[-1][1]}
    out.update({f"ce_exit_{g}": c for g, (c, _) in enumerate(per_exit)})
    return out


def _param_groups(model):
    last_of_group = {grp[-1]: g for g, grp in enumerate(model.groups)}
    extra = [model.lens_norms[last_of_group[i]] if i in last_of_group else None
             for i in range(model.config.n_layer)]
    return block_param_groups(model.stack, extra_per_block=extra)


def train_sharp(model, train_data, val_data, block_size, device, steps=1000, batch_size=64,
                max_lr=3e-3, warmup_steps=200, weight_decay=0.01, grad_clip=1.0,
                eval_every=200, eval_windows=64, generator=None, eval_seed=None):
    model.to(device)
    optimizer, scheduler = make_optimizer(_param_groups(model), steps, max_lr, warmup_steps, weight_decay)
    return run_local_training(
        step_fn=lambda x, y: sharp_train_step(model, optimizer, x, y, grad_clip),
        eval_fn=lambda: evaluate(model, val_data, block_size, device, n_windows=eval_windows,
                                 generator=eval_generator(generator, eval_seed)),
        optimizer=optimizer, scheduler=scheduler, train_data=train_data, block_size=block_size,
        batch_size=batch_size, steps=steps, eval_every=eval_every, device=device, generator=generator,
    )


def _probs(logits, temperature, top_k):
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[..., [-1]], -float("inf"))
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def early_exit_eval(model, data, block_size, device, thresholds=(0.5, 0.7, 0.9), n_windows=64, generator=None):
    """Confidence-based early exit: per token, use the first exit with max prob >= threshold.

    Returns CE, accuracy and mean fraction of blocks run for each threshold.
    """
    model.eval()
    x, y = get_batch(data, block_size, n_windows, generator=generator)
    x, y = x.to(device), y.to(device)
    logps = torch.stack([F.log_softmax(l.float(), -1) for l in model.exit_logits(x)])
    depths = torch.tensor([model.exit_depth(g) for g in range(model.n_exits)], device=device)
    conf = logps.exp().amax(-1)
    results = []
    for thr in thresholds:
        ok = conf >= thr
        ok[-1] = True
        first = ok.float().argmax(0)
        chosen = logps.gather(0, first[None, ..., None].expand(1, *first.shape, logps.size(-1)))[0]
        ce = F.nll_loss(chosen.reshape(-1, chosen.size(-1)), y.reshape(-1)).item()
        acc = (chosen.argmax(-1) == y).float().mean().item()
        results.append({"threshold": thr, "cross_entropy": ce, "accuracy": acc,
                        "compute_fraction": depths[first].mean().item()})
    model.train()
    return results


@torch.no_grad()
def generate_sharp(model, tokenizer, device, prompt="\n", max_new_tokens=200, temperature=0.8, top_k=None,
                   generator=None, exit_group=None):
    model.eval()
    block_size = model.config.block_size
    idx = tokenizer.encode(prompt).long().unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        p = _probs(model.logits(idx[:, -block_size:], exit_group=exit_group)[:, -1, :], temperature, top_k)
        idx = torch.cat([idx, torch.multinomial(p, 1, generator=generator)], dim=1)
    model.train()
    return tokenizer.decode(idx[0])


@torch.no_grad()
def speculative_generate(model, tokenizer, device, prompt="\n", max_new_tokens=200, draft_exit=0, n_draft=4,
                         temperature=0.8, top_k=None, generator=None):
    """Self-speculative decoding: exit `draft_exit` drafts n_draft tokens, the full model verifies them.

    Standard speculative-sampling acceptance, so outputs follow the full model's
    distribution. No KV cache. Returns (text, stats).
    """
    model.eval()
    bs = model.config.block_size
    idx = tokenizer.encode(prompt).long().unsqueeze(0).to(device)
    start_len = idx.shape[1]
    stats = {"proposed": 0, "accepted": 0, "full_passes": 0, "draft_passes": 0}
    t0 = time.time()

    def rand():
        return torch.rand(1, generator=generator, device=device).item()

    while idx.shape[1] - start_len < max_new_tokens:
        ctx, qs, drafts = idx, [], []
        for _ in range(n_draft):
            q = _probs(model.logits(ctx[:, -bs:], exit_group=draft_exit)[:, -1, :], temperature, top_k)
            d = torch.multinomial(q, 1, generator=generator)
            qs.append(q[0]); drafts.append(d.item())
            ctx = torch.cat([ctx, d], dim=1)
        stats["draft_passes"] += n_draft

        win = ctx[:, -bs:]
        W = win.shape[1]
        p_all = _probs(model.logits(win)[0], temperature, top_k)
        stats["full_passes"] += 1
        new = []
        for j, (d, q) in enumerate(zip(drafts, qs)):
            p = p_all[W - n_draft + j - 1]
            stats["proposed"] += 1
            if rand() < min(1.0, (p[d] / q[d].clamp_min(1e-12)).item()):
                new.append(d); stats["accepted"] += 1
            else:
                resid = (p - q).clamp_min(0)
                resid = resid / resid.sum() if resid.sum() > 0 else p
                new.append(torch.multinomial(resid, 1, generator=generator).item())
                break
        else:
            new.append(torch.multinomial(p_all[W - 1], 1, generator=generator).item())
        idx = torch.cat([idx, torch.tensor([new], device=device)], dim=1)

    idx = idx[:, :start_len + max_new_tokens]
    stats["seconds"] = time.time() - t0
    stats["acceptance_rate"] = stats["accepted"] / max(1, stats["proposed"])
    stats["tokens_per_full_pass"] = max_new_tokens / max(1, stats["full_passes"])
    d = model.exit_depth(draft_exit)
    stats["relative_block_compute"] = (stats["full_passes"] + d * stats["draft_passes"]) / max_new_tokens
    model.train()
    return tokenizer.decode(idx[0]), stats
