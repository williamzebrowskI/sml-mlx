# import mlx.core as mx
# import mlx.nn as nn

# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps   = eps
#         self.scale = mx.ones(dim)

#     def __call__(self, x):
#         return x * mx.rsqrt((x ** 2).mean(-1, keepdims=True) + self.eps) * self.scale

# class TransformerConfig:
#     def __init__(self, **kw):
#         self.__dict__.update(kw)
#         # ensure required fields are present
#         self.checkpoint = getattr(self, 'checkpoint', False)
#         self.dropout = getattr(self, 'dropout', 0.0)

# class Transformer(nn.Module):
#     def __init__(self, cfg: TransformerConfig):
#         super().__init__()
#         self.cfg = cfg

#         # token + positional embeddings
#         self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
#         self.pos_enc = nn.SinusoidalPositionalEncoding(cfg.hidden_size)

#         # transformer stack
#         # signature: TransformerEncoder(num_layers, dims, num_heads, norm_first=True, dropout=0.0, checkpoint=False)
#         self.encoder = nn.TransformerEncoder(
#             cfg.n_layer,
#             cfg.hidden_size,
#             cfg.n_head,
#             norm_first=True,
#             dropout=cfg.dropout,
#             checkpoint=cfg.checkpoint,
#         )

#         # output projection (weight-tied)
#         self.out_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
#         self.out_proj.weight = self.emb.weight

#     def __call__(self, x: mx.array):
#         # x: (B, L)
#         B, L = x.shape
#         # causal mask
#         mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

#         # embeddings + positional encoding
#         h = self.emb(x) + self.pos_enc(mx.arange(L, dtype=mx.int32))

#         # transformer
#         h = self.encoder(h, mask=mask)

#         # project back to vocab
#         return self.out_proj(h)
# model/model.py
import math
import mlx.core as mx
import mlx.nn   as nn


# ─────────────────────────────────────────
# 1. RMSNorm (γ kept trainable)
# ─────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = mx.ones(dim)          # trainable parameter

    def __call__(self, x: mx.array):
        return x * mx.rsqrt((x ** 2).mean(-1, keepdims=True) + self.eps) * self.scale


# ─────────────────────────────────────────
# 2. RoPE-aware wrapper that *preserves*
#    the original param names so old
#    checkpoints still load.
# ─────────────────────────────────────────
class MultiHeadAttentionRoPE(nn.MultiHeadAttention):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.rope = nn.RoPE(self.head_dim)

    def __call__(self, x: mx.array, *, mask: mx.array | None = None):
        B, L, _ = x.shape

        # standard QKV projections (weights keep same names)
        q = self.query_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.key_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.value_proj(x).reshape(B, L, self.num_heads, self.head_dim)

        # apply rotary to Q and K in-place
        q = self.rope(q)
        k = self.rope(k)

        # scaled dot-product attention (reuse parent helper)
        attn_out = self.scaled_dot_product_attention(q, k, v, mask)
        return self.out_proj(attn_out)


# Monkey-patch BEFORE any encoder is built
nn.MultiHeadAttention = MultiHeadAttentionRoPE


# ─────────────────────────────────────────
# 3. Config helper (unchanged except dropout default)
# ─────────────────────────────────────────
class TransformerConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.dropout    = getattr(self, "dropout", 0.0)   # set 0.1 in config.json
        self.checkpoint = getattr(self, "checkpoint", False)


# ─────────────────────────────────────────
# 4. Main Transformer
#    (identical weight names → checkpoints load)
# ─────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # token embeddings
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # transformer stack (uses patched RoPE attention)
        self.encoder = nn.TransformerEncoder(
            num_layers   = cfg.n_layer,
            dims         = cfg.hidden_size,
            num_heads    = cfg.n_head,
            norm_first   = True,
            dropout      = cfg.dropout,     # enable with "dropout": 0.1
            checkpoint   = cfg.checkpoint,
            norm_factory = lambda _: RMSNorm(cfg.hidden_size),
        )

        # weight-tied output head
        self.out_proj      = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.out_proj.weight = self.emb.weight

    # --------------------------------------------------
    def __call__(self, x: mx.array):
        """
        x: (B, L) int32 token IDs
        returns logits: (B, L, vocab_size)
        """
        B, L = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        h = self.emb(x)               # (B, L, D)
        h = self.encoder(h, mask=mask)
        return self.out_proj(h)