"""Block-level Direct Feedback Alignment (Nokland 2016; Launay et al. 2020).

The output error e = softmax(logits) - onehot(y) reaches each hidden block's
output through a fixed random matrix B_i; backprop runs only inside a block.
The last block and the head get the true gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_llm_lib.local_learning import (
    BlockStack, block_param_groups, clip_per_group, eval_ce_all_positions, eval_generator, make_optimizer,
    run_local_training,
)
from ff_llm_lib.model import build_norm


class DFAModel(nn.Module):
    def __init__(self, config, shared_feedback=False, feedback_scale=1.0):
        super().__init__()
        self.config = config
        self.stack = BlockStack(config)
        self.norm_f = build_norm(config.use_rmsnorm, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.stack.token_emb.weight
        n_fb = 1 if shared_feedback else config.n_layer - 1
        fb = torch.randn(max(n_fb, 1), config.vocab_size, config.n_embd) * feedback_scale / config.vocab_size ** 0.5
        self.register_buffer("feedback", fb)  # fixed, never trained
        self.shared_feedback = shared_feedback

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def head(self, h):
        return self.lm_head(self.norm_f(h))

    @torch.no_grad()
    def logits(self, x):
        h = self.stack.embed(x)
        for block in self.stack.blocks:
            h = block(h)
        return self.head(h)


def dfa_train_step(model, optimizer, x, y, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    blocks = model.stack.blocks
    h = model.stack.embed(x)
    hidden_outs = []
    for i, block in enumerate(blocks):
        h_in = h if i == 0 else h.detach()
        h = block(h_in)
        if i < len(blocks) - 1:
            hidden_outs.append(h)

    logits = model.head(h)
    V = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    with torch.no_grad():
        e = F.softmax(logits.float(), dim=-1)
        e.scatter_add_(-1, y.unsqueeze(-1), -torch.ones_like(y, dtype=e.dtype).unsqueeze(-1))
        e /= y.numel()  # d(mean CE)/d(logits)

    surrogate = 0.0
    for i, h_i in enumerate(hidden_outs):
        B_i = model.feedback[0 if model.shared_feedback else i]
        # Feedback goes through a parameter-free RMS norm of h_i. Without it the
        # surrogate is linear in h_i and blocks minimize it by growing the
        # residual stream without bound.
        surrogate = surrogate + (F.rms_norm(h_i, (h_i.size(-1),)) * (e @ B_i)).sum()
    (loss + surrogate).backward()
    clip_per_group(optimizer, grad_clip)
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(model, data, block_size, device, n_windows=64, generator=None):
    model.eval()
    ce, acc = eval_ce_all_positions(model.logits, data, block_size, device, n_windows=n_windows, generator=generator)
    model.train()
    return {"dev_loss": ce, "cross_entropy": ce, "accuracy": acc}


def train_dfa(model, train_data, val_data, block_size, device, steps=1000, batch_size=64,
              max_lr=3e-3, warmup_steps=200, weight_decay=0.01, grad_clip=1.0,
              eval_every=200, eval_windows=64, generator=None, eval_seed=None):
    model.to(device)
    groups = block_param_groups(model.stack, shared=[model.norm_f])
    optimizer, scheduler = make_optimizer(groups, steps, max_lr, warmup_steps, weight_decay)
    return run_local_training(
        step_fn=lambda x, y: dfa_train_step(model, optimizer, x, y, grad_clip),
        eval_fn=lambda: evaluate(model, val_data, block_size, device, n_windows=eval_windows,
                                 generator=eval_generator(generator, eval_seed)),
        optimizer=optimizer, scheduler=scheduler, train_data=train_data, block_size=block_size,
        batch_size=batch_size, steps=steps, eval_every=eval_every, device=device, generator=generator,
    )


@torch.no_grad()
def generate_dfa(model, tokenizer, device, prompt="\n", max_new_tokens=200, temperature=0.8, top_k=None,
                 generator=None):
    model.eval()
    block_size = model.config.block_size
    idx = tokenizer.encode(prompt).long().unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits = model.logits(idx[:, -block_size:])[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return tokenizer.decode(idx[0])
