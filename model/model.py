
import mlx
import mlx.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(mlx.ones(dim))

    def forward(self, x):
        return x * mlx.rsqrt((x ** 2).mean(-1, keepdims=True) + self.eps) * self.scale

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = nn.MultiHeadAttention(
            cfg.hidden_size,
            cfg.n_head,
            cfg.hidden_size // cfg.n_head
        )
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        )

    def forward(self, x, mask=None):
        h = self.attn(self.norm1(x), mask=mask)
        x = x + h
        h = self.mlp(self.norm2(x))
        x = x + h
        return x

class TransformerConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # weight tying
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.out_proj.weight = self.emb.weight

    def forward(self, x, mask=None):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.norm(h)
        return self.out_proj(h)
