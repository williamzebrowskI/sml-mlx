import mlx.core as mx
import mlx.nn   as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = mx.ones(dim)

    def __call__(self, x):
        return x * mx.rsqrt((x**2).mean(-1, keepdims=True) + self.eps) * self.scale


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn  = nn.MultiHeadAttention(
            cfg.hidden_size,                 # dims
            cfg.n_head                       # num_heads
        ) 
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp   = nn.Sequential(
            nn.Linear(cfg.hidden_size,       cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size,       bias=False),
        )

    def __call__(self, x, mask=None):
        qkv = self.norm1(x)
        x   = x + self.attn(qkv, qkv, qkv, mask=mask)   # ← pass q, k, v
        x   = x + self.mlp(self.norm2(x))
        return x


class TransformerConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb    = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        self.norm   = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.out_proj        = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.out_proj.weight = self.emb.weight

    def __call__(self, x, mask=None):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.out_proj(self.norm(h))