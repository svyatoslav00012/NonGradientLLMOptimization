"""Shared pieces for the layer-local training methods (FF, SigProp, DFA, SHARP).

Every block's input is detached, so no gradient crosses a block boundary.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_llm_lib.model import Block, build_norm
from ff_llm_lib.text_data import get_batch
from ff_llm_lib.train import make_cosine_warmup_scheduler


def init_gpt_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


class BlockStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        init_gpt_weights(self)

    def embed(self, x):
        h = self.token_emb(x)
        if self.pos_emb is not None:
            h = h + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        return h

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def block_param_groups(stack, extra_per_block=None, shared=None):
    """One param group per block (block 0 also owns the embeddings).

    extra_per_block: optional per-block module trained with that block.
    shared: optional modules trained by every block, in their own group.
    Tied parameters are listed once.
    """
    seen = set()

    def fresh(params):
        out = [p for p in params if id(p) not in seen]
        seen.update(id(p) for p in out)
        return out

    groups = []
    if shared:
        groups.append({"params": fresh(p for m in shared for p in m.parameters())})
    for i, block in enumerate(stack.blocks):
        params = list(block.parameters())
        if i == 0:
            params += list(stack.token_emb.parameters())
            if stack.pos_emb is not None:
                params += list(stack.pos_emb.parameters())
        if extra_per_block is not None and extra_per_block[i] is not None:
            params += list(extra_per_block[i].parameters())
        groups.append({"params": fresh(params)})
    return [g for g in groups if g["params"]]


def make_optimizer(param_groups, steps, max_lr, warmup_steps, weight_decay):
    optimizer = torch.optim.AdamW(param_groups, lr=max_lr, weight_decay=weight_decay)
    scheduler = make_cosine_warmup_scheduler(optimizer, max(1, warmup_steps), steps, max_lr)
    return optimizer, scheduler


def clip_per_group(optimizer, max_norm):
    # per group, so clipping one block never depends on another block's gradients
    if max_norm is None:
        return
    for g in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(g["params"], max_norm)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def eval_generator(generator, eval_seed):
    """Fixed-seed generator for evaluation, so eval never consumes the training RNG."""
    return torch.Generator().manual_seed(eval_seed) if eval_seed is not None else generator


@torch.no_grad()
def eval_ce_all_positions(logits_fn, data, block_size, device, n_windows=256, batch_size=32, generator=None):
    """Next-token CE and top-1 accuracy over all positions of `n_windows` windows.

    `logits_fn(x)` returns (B, T, V) logits or a list of them (one per exit);
    the result is (ce, acc) or a list of (ce, acc) accordingly.
    """
    totals, total_n = None, 0
    remaining = n_windows
    single = False
    while remaining > 0:
        bs = min(batch_size, remaining)
        x, y = get_batch(data, block_size, bs, generator=generator)
        x, y = x.to(device), y.to(device)
        out = logits_fn(x)
        if torch.is_tensor(out):
            out, single = [out], True
        if totals is None:
            totals = [[0.0, 0] for _ in out]
        for t, logits in zip(totals, out):
            t[0] += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum").item()
            t[1] += (logits.argmax(-1) == y).sum().item()
        total_n += y.numel()
        remaining -= bs
    res = [(ce / total_n, c / total_n) for ce, c in totals]
    return res[0] if single else res


def run_local_training(step_fn, eval_fn, optimizer, scheduler, train_data, block_size, batch_size,
                       steps, eval_every, device, generator=None):
    """Training loop shared by the local methods.

    step_fn(x, y) returns the train loss as a tensor (not synced every step).
    eval_fn() returns a dict with at least "dev_loss" and "cross_entropy".
    Returns (history, training seconds excluding eval time).
    """
    history = {"step": [], "train_loss": [], "dev_loss": [], "lr": [], "cross_entropy": [],
               "accuracy": [], "elapsed": []}
    train_time = 0.0
    t_last = time.time()
    for step in range(steps):
        x, y = get_batch(train_data, block_size, batch_size, generator=generator)
        x, y = x.to(device), y.to(device)
        train_loss = step_fn(x, y)
        scheduler.step()

        if step % eval_every == 0 or step == steps - 1:
            sync(device)
            train_time += time.time() - t_last
            ev = eval_fn()
            lr = scheduler.get_last_lr()[0]
            history["step"].append(step)
            history["train_loss"].append(float(train_loss))
            history["dev_loss"].append(ev["dev_loss"])
            history["lr"].append(lr)
            history["cross_entropy"].append(ev["cross_entropy"])
            history["accuracy"].append(ev.get("accuracy"))
            history["elapsed"].append(train_time)
            extra = " ".join(f"| {k} {v:.4f}" for k, v in ev.items()
                             if k not in ("dev_loss", "cross_entropy", "accuracy") and isinstance(v, float))
            print(f"step {step:6d} | train loss {float(train_loss):.4f} | dev loss {ev['dev_loss']:.4f} "
                  f"| val CE {ev['cross_entropy']:.4f} | val acc {ev.get('accuracy', float('nan')):.1%} "
                  f"| lr {lr:.5f} {extra}")
            t_last = time.time()

    sync(device)
    train_time += time.time() - t_last
    return history, train_time


class LensHead(nn.Module):
    """Norm + linear readout to the vocabulary."""

    def __init__(self, config):
        super().__init__()
        self.norm = build_norm(config.use_rmsnorm, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, h):
        return self.proj(self.norm(h))
