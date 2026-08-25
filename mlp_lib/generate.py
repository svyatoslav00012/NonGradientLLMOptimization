import torch

from mlp_lib.ops import manual_softmax


def predict_next_ff(ff_model, context, generator=None):
    """
    FF equivalent of predict_next: score every candidate next char by total
    goodness (context + candidate, summed goodness across blocks), treat
    those scores as logits, softmax + sample.
    """
    V = ff_model.vocab_size
    context_t = torch.tensor(context).unsqueeze(0).expand(V, -1)
    candidates = torch.arange(V).unsqueeze(1)
    ids = torch.cat([context_t, candidates], dim=1)

    totals = ff_model.total_goodness(ids)          # (V,) -- one per candidate char
    probs = manual_softmax(totals.unsqueeze(0))     # (1, V)
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


def predict_word_ff(ff_model, stoi, itos, context_len, max_len=20, generator=None):
    context = [stoi['.']] * context_len
    out = []
    for _ in range(max_len):
        ix = predict_next_ff(ff_model, context, generator=generator)
        if ix == stoi['.']:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    return ''.join(out)


def sample_words_ff(ff_model, stoi, itos, context_len, n=10, generator=None):
    return [predict_word_ff(ff_model, stoi, itos, context_len, generator=generator) for _ in range(n)]


def predict_next(model, context, generator=None):
    """
    context: list of `context_len` integer token ids.
    Returns one sampled next token id, drawn from the model's softmax
    distribution (torch.multinomial, not argmax) so generated names vary.
    """
    x = torch.tensor([context])
    logits = model.forward(x)
    probs = manual_softmax(logits)
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


def predict_word(model, stoi, itos, context_len, max_len=20, generator=None):
    """
    Repeatedly samples predict_next(), feeding each sampled char back in
    as context, until it samples the end token '.' or hits max_len.
    """
    context = [stoi['.']] * context_len
    out = []
    for _ in range(max_len):
        ix = predict_next(model, context, generator=generator)
        if ix == stoi['.']:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    return ''.join(out)


def sample_words(model, stoi, itos, context_len, n=10, generator=None):
    return [predict_word(model, stoi, itos, context_len, generator=generator) for _ in range(n)]
