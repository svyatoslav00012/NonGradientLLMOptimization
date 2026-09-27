import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_from_tokenizer(model, tokenizer, device, prompt="\n", max_new_tokens=200,
                            temperature=0.8, top_k=None, generator=None):
    """Continue `prompt` by `max_new_tokens` tokens with a GPT model."""
    model.eval()
    block_size = model.config.block_size
    idx = tokenizer.encode(prompt).long().unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        next_ix = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_ix], dim=1)

    model.train()
    return tokenizer.decode(idx[0])
