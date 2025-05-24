import mlx.core as mx
import mlx.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = mx.ones(dim)

    def __call__(self, x):
        return x * mx.rsqrt((x ** 2).mean(-1, keepdims=True) + self.eps) * self.scale

class TransformerConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        # ensure required fields are present
        self.checkpoint = getattr(self, 'checkpoint', False)
        self.dropout = getattr(self, 'dropout', 0.0)

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # token + positional embeddings
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_enc = nn.SinusoidalPositionalEncoding(cfg.hidden_size)

        # transformer stack
        # signature: TransformerEncoder(num_layers, dims, num_heads, norm_first=True, dropout=0.0, checkpoint=False)
        self.encoder = nn.TransformerEncoder(
            cfg.n_layer,
            cfg.hidden_size,
            cfg.n_head,
            norm_first=True,
            dropout=cfg.dropout,
            checkpoint=cfg.checkpoint,
        )

        # output projection (weight-tied)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.out_proj.weight = self.emb.weight

    def __call__(self, x: mx.array):
        # x: (B, L)
        B, L = x.shape
        # causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        # embeddings + positional encoding
        h = self.emb(x) + self.pos_enc(mx.arange(L, dtype=mx.int32))

        # transformer
        h = self.encoder(h, mask=mask)

        # project back to vocab
        return self.out_proj(h)
