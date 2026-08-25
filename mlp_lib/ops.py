import torch


def manual_relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def manual_softmax(logits):
    """
    p_i = exp(z_i - max) / sum_j exp(z_j - max)

    Subtracting the row max doesn't change the result (cancels in the
    ratio) but keeps exp() from overflowing on large logits.
    """
    logits_max = torch.max(logits, dim=1, keepdim=True).values  # (B, 1)
    norm_logits = logits - logits_max                            # (B, C)

    counts = torch.exp(norm_logits)              # (B, C)
    counts_sum = counts.sum(1, keepdim=True)     # (B, 1)
    return counts / counts_sum                   # (B, C)


def manual_log_softmax(logits):
    """
    log(p_i) = (z_i - max) - log(sum_j exp(z_j - max))

    Computed as a subtraction, not log(manual_softmax(logits)): if a
    probability underflows to exact 0.0, log(0) = -inf. Doing the
    subtraction in log-space stays finite, since counts_sum always
    includes exp(0) = 1 from the max element itself.
    """
    logits_max = torch.max(logits, dim=1, keepdim=True).values
    norm_logits = logits - logits_max

    counts = torch.exp(norm_logits)
    counts_sum = counts.sum(1, keepdim=True)
    return norm_logits - torch.log(counts_sum)


def manual_cross_entropy(logits, targets):
    """
    logits: (B, C) raw unnormalized scores
    targets: (B,) ground-truth class indices
    returns (loss, probs) -- probs is needed by the backward pass
    """
    B = logits.shape[0]

    probs = manual_softmax(logits)
    log_probs = manual_log_softmax(logits)

    correct_log_probs = log_probs[torch.arange(B), targets]
    loss = -correct_log_probs.mean()

    return loss, probs


def manual_cross_entropy_backward(probs, targets):
    """
    dL/dlogits = (probs - one_hot(targets)) / B

    L = -log(softmax(z)_t). Differentiating softmax and log together,
    the Jacobian collapses to dL/dz_i = p_i - 1{i == t}. The /B comes
    from the forward pass using .mean() over the batch, not .sum().
    """
    B = probs.shape[0]

    dlogits = probs.clone()
    dlogits[range(B), targets] -= 1.0
    dlogits /= B

    return dlogits
