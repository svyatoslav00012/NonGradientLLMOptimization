"""Local cross-entropy per block (Signal Propagation / Mono-Forward style).

Each block has its own readout trained with next-token CE on its detached
output, at every position.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_llm_lib.local_learning import (
    BlockStack, LensHead, block_param_groups, clip_per_group, eval_ce_all_positions, eval_generator,
    make_optimizer, run_local_training,
)


class SigPropModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stack = BlockStack(config)
        self.readouts = nn.ModuleList([LensHead(config) for _ in range(config.n_layer)])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def logits(self, x, exit_block=None):
        exit_block = self.config.n_layer - 1 if exit_block is None else exit_block
        h = self.stack.embed(x)
        for block in self.stack.blocks[:exit_block + 1]:
            h = block(h)
        return self.readouts[exit_block](h)

    @torch.no_grad()
    def all_exit_logits(self, x):
        h = self.stack.embed(x)
        out = []
        for block, readout in zip(self.stack.blocks, self.readouts):
            h = block(h)
            out.append(readout(h))
        return out


def train_step(model, optimizer, x, y, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    h = model.stack.embed(x)
    losses = []
    for block, readout in zip(model.stack.blocks, model.readouts):
        h = block(h)
        logits = readout(h)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()  # frees this block's graph right away
        losses.append(loss.detach())
        h = h.detach()
    clip_per_group(optimizer, grad_clip)
    optimizer.step()
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, data, block_size, device, n_windows=256, generator=None):
    """cross_entropy/accuracy: last block. dev_loss: mean CE over blocks. ce_block_i: each block."""
    model.eval()
    per_block = eval_ce_all_positions(model.all_exit_logits, data, block_size, device,
                                      n_windows=n_windows, generator=generator)
    model.train()
    out = {"dev_loss": sum(c for c, _ in per_block) / len(per_block),
           "cross_entropy": per_block[-1][0], "accuracy": per_block[-1][1]}
    out.update({f"ce_block_{i}": c for i, (c, _) in enumerate(per_block)})
    return out


def train_sigprop(model, train_data, val_data, block_size, device, steps=1000, batch_size=64,
                  max_lr=3e-3, warmup_steps=200, weight_decay=0.01, grad_clip=1.0,
                  eval_every=200, eval_windows=64, generator=None, eval_seed=None):
    model.to(device)
    groups = block_param_groups(model.stack, extra_per_block=list(model.readouts))
    optimizer, scheduler = make_optimizer(groups, steps, max_lr, warmup_steps, weight_decay)
    return run_local_training(
        step_fn=lambda x, y: train_step(model, optimizer, x, y, grad_clip),
        eval_fn=lambda: evaluate(model, val_data, block_size, device, n_windows=eval_windows,
                                 generator=eval_generator(generator, eval_seed)),
        optimizer=optimizer, scheduler=scheduler, train_data=train_data, block_size=block_size,
        batch_size=batch_size, steps=steps, eval_every=eval_every, device=device, generator=generator,
    )


@torch.no_grad()
def generate_sigprop(model, tokenizer, device, prompt="\n", max_new_tokens=200,
                     temperature=0.8, top_k=None, generator=None, exit_block=None):
    model.eval()
    block_size = model.config.block_size
    idx = tokenizer.encode(prompt).long().unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits = model.logits(idx[:, -block_size:], exit_block=exit_block)[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return tokenizer.decode(idx[0])
