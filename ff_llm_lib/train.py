"""Backprop baseline: AdamW with linear warmup, cosine decay and gradient clipping."""
import time

import torch

from ff_llm_lib.text_data import get_batch


def make_cosine_warmup_scheduler(optimizer, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
    """Linear warmup, then cosine decay to min_lr_ratio * max_lr."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / warmup_steps, end_factor=1.0, total_iters=warmup_steps
    )
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=max_lr * min_lr_ratio
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


@torch.no_grad()
def evaluate_lm(model, data, block_size, device, batch_size=64, generator=None):
    model.eval()
    xb, yb = get_batch(data, block_size, batch_size, generator=generator)
    xb, yb = xb.to(device), yb.to(device)
    _, loss = model(xb, yb)
    model.train()
    return loss.item()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_lm(model, train_data, val_data, block_size, device, steps=5000, batch_size=64,
             max_lr=3e-3, warmup_steps=200, weight_decay=0.01, grad_clip=1.0,
             eval_every=500, generator=None, eval_seed=None, eval_batch_size=64):
    """Returns (history, training seconds excluding eval time)."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = make_cosine_warmup_scheduler(optimizer, warmup_steps, steps, max_lr)
    eval_gen = torch.Generator().manual_seed(eval_seed) if eval_seed is not None else generator

    history = {"step": [], "train_loss": [], "dev_loss": [], "lr": [], "elapsed": []}
    train_time = 0.0
    t_last = time.time()
    for step in range(steps):
        xb, yb = get_batch(train_data, block_size, batch_size, generator=generator)
        xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == steps - 1:
            _sync(device)
            train_time += time.time() - t_last
            dev_loss = evaluate_lm(model, val_data, block_size, device, batch_size=eval_batch_size,
                                   generator=eval_gen)
            history["elapsed"].append(train_time)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["dev_loss"].append(dev_loss)
            history["lr"].append(scheduler.get_last_lr()[0])
            print(f"step {step:6d} | train loss {loss.item():.4f} | dev loss {dev_loss:.4f} "
                  f"| lr {scheduler.get_last_lr()[0]:.5f}")
            t_last = time.time()

    _sync(device)
    train_time += time.time() - t_last
    return history, train_time
