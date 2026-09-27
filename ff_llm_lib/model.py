from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True


def build_norm(use_rmsnorm, dim):
    return nn.RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)


def precompute_rope(head_dim, block_size, base=10000.0):
    """Rotary position embedding tables, each (block_size, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(block_size).float()
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = config.use_rope

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

        if config.use_rope:
            cos, sin = precompute_rope(self.head_dim, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, k, self.rope_cos[:T], self.rope_sin[:T])

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    """GPT-2 style MLP: Linear -> GELU -> Linear, 4x expansion."""

    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden)
        self.fc2 = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, config):
        super().__init__()
        hidden = int(4 * config.n_embd * 2 / 3)  # keep param count ~= 4x GELU MLP
        self.gate_proj = nn.Linear(config.n_embd, hidden)
        self.up_proj = nn.Linear(config.n_embd, hidden)
        self.down_proj = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Pre-norm transformer decoder block: x + Attn(Norm(x)), x + MLP(Norm(x))."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = build_norm(config.use_rmsnorm, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = build_norm(config.use_rmsnorm, config.n_embd)
        self.mlp = SwiGLU(config) if config.use_swiglu else MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # learned positions only when RoPE is off
        self.pos_emb = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = build_norm(config.use_rmsnorm, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss
