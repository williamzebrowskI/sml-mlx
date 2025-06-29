
# import math
# import dataclasses
# from typing import List

# import mlx.core as mx
# import mlx.nn   as nn


# # ───────────────────────────────────
# # 1.  Config (mirrors HF config keys)
# # ───────────────────────────────────
# @dataclasses.dataclass
# class SMLMConfig:
#     vocab_size         : int
#     model_dim          : int                # hidden size
#     n_layers           : int
#     head_dim           : int
#     num_q_heads        : List[int]          # len = n_layers
#     num_kv_heads       : List[int]          # len = n_layers
#     ffn_multipliers    : List[float]        # len = n_layers
#     ffn_dim_divisor    : int = 256
#     rope_freq_constant : int = 10_000
#     rope_max_length    : int = 4096
#     dropout            : float  = 0.0
#     use_swiglu         : bool   = True
#     tie_embeddings     : bool   = True


# # ───────────────────────────────────
# # 2.  RMSNorm wrapper (MLX has one)
# # ───────────────────────────────────
# RMSNorm = nn.RMSNorm   # alias for brevity

# def repeat_kv(x: mx.array, n_rep: int, axis: int = 2) -> mx.array:
#     """Tile (repeat) KV heads along 'axis' to match query heads."""
#     return mx.repeat(x, n_rep, axis=axis)

# # ───────────────────────────────────
# # 3.  FFN (with optional SwiGLU)
# # ───────────────────────────────────
# class FeedForward(nn.Module):
#     def __init__(self, cfg: SMLMConfig, layer_idx: int):
#         super().__init__()
#         mult = cfg.ffn_multipliers[layer_idx]
#         inter = int(
#             math.ceil(mult * cfg.model_dim / cfg.ffn_dim_divisor)
#             * cfg.ffn_dim_divisor
#         )
#         self.use_glu = cfg.use_swiglu
#         out_features = 2 * inter if self.use_glu else inter

#         self.proj_in  = nn.Linear(cfg.model_dim, out_features, bias=False)
#         self.proj_out = nn.Linear(inter, cfg.model_dim, bias=False)
#         self.act      = nn.silu if self.use_glu else nn.gelu

#     def __call__(self, x: mx.array) -> mx.array:
#         y = self.proj_in(x)
#         if self.use_glu:
#             y1, y2 = mx.split(y, 2, axis=-1)
#             y = self.act(y1) * y2
#         else:
#             y = self.act(y)
#         return self.proj_out(y)


# # ───────────────────────────────────
# # 4.  Grouped-Query Attention w/ RoPE
# # ───────────────────────────────────
# class GQAttention(nn.Module):
#     def __init__(self, cfg: SMLMConfig, layer_idx: int):
#         super().__init__()
#         self.q_heads = cfg.num_q_heads[layer_idx]
#         self.kv_heads = cfg.num_kv_heads[layer_idx]
#         self.head_dim = cfg.head_dim
#         self.groups   = self.q_heads // self.kv_heads

#         out_dim = (self.q_heads + 2 * self.kv_heads) * self.head_dim
#         self.qkv = nn.Linear(cfg.model_dim, out_dim, bias=False)
#         self.o_proj = nn.Linear(self.q_heads * self.head_dim,
#                                 cfg.model_dim, bias=False)

#         self.rope = nn.RoPE(self.head_dim)

#     def __call__(self, x: mx.array, *, mask: mx.array):
#         """
#         x   : (B, L, D)
#         mask: (L, L) additive causal mask from MLX helper
#         returns (B, L, D)
#         """
#         B, L, _ = x.shape

#         # 1) project to Q-K-V and reshape
#         qkv = self.qkv(x).reshape(
#             B, L, self.q_heads + 2 * self.kv_heads, self.head_dim
#         )                                    # (B, L, Htot, D)

#         # 2) put heads first for MLX attention
#         qkv = qkv.transpose(0, 2, 1, 3)      # (B, Htot, L, D)

#         # 3) split into query / key / value along head axis
#         q, k, v = mx.split(
#             qkv,
#             [self.q_heads, self.q_heads + self.kv_heads],
#             axis=1                            # heads axis
#         )                                     # each (B, H, L, D)

#         # 4) rotary position embedding on Q & K
#         q = self.rope(q)
#         k = self.rope(k)

#         # 5) repeat KV heads for GQA (now along axis=1)
#         k = repeat_kv(k, self.groups, axis=1)
#         v = repeat_kv(v, self.groups, axis=1)

#         # 6) scaled-dot-product attention (MLX kernel)
#         attn = mx.fast.scaled_dot_product_attention(
#             q, k, v,
#             scale=1.0 / math.sqrt(self.head_dim),
#             mask=mask,                         # (L, L) → broadcast to (B, H, L, L)
#         )                                      # (B, H, L, D)

#         # 7) merge heads back to (B, L, D_model)
#         attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
#         return self.o_proj(attn)


# # ───────────────────────────────────
# # 5.  Decoder layer
# # ───────────────────────────────────
# class DecoderLayer(nn.Module):
#     def __init__(self, cfg: SMLMConfig, layer_idx: int):
#         super().__init__()
#         self.norm1 = RMSNorm(cfg.model_dim)
#         self.attn  = GQAttention(cfg, layer_idx)
#         self.norm2 = RMSNorm(cfg.model_dim)
#         self.ffn   = FeedForward(cfg, layer_idx)

#     def __call__(self, x: mx.array, *, mask: mx.array):
#         x = x + self.attn(self.norm1(x), mask=mask)
#         x = x + self.ffn(self.norm2(x))
#         return x


# # ───────────────────────────────────
# # 6.  Complete OpenELM-style decoder
# # ───────────────────────────────────
# class OpenELM(nn.Module):
#     def __init__(self, cfg: SMLMConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)

#         self.layers = [DecoderLayer(cfg, i) for i in range(cfg.n_layers)]
#         self.final_norm = RMSNorm(cfg.model_dim)

#         self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
#         if cfg.tie_embeddings:
#             self.lm_head.weight = self.emb.weight  # weight tying

#     # --------------------------------------------------------
#     def __call__(self, tokens: mx.array) -> mx.array:
#         """
#         tokens: (B, L) int32
#         returns logits: (B, L, vocab)
#         """
#         B, L = tokens.shape
#         mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

#         h = self.emb(tokens)
#         for layer in self.layers:
#             h = layer(h, mask=mask)
#         h = self.final_norm(h)
#         return self.lm_head(h)

# model/model.py
import math, json, pathlib, dataclasses
from typing import List

import mlx.core as mx
import mlx.nn   as nn

# ───────────────────────────────────
# 1) Config (all args pulled from JSON)
# ───────────────────────────────────
@dataclasses.dataclass
class SMLMConfig:
    # transformer
    tokenizer_path           : str
    checkpoint_dir           : str
    vocab_size               : int
    model_dim                : int
    num_transformer_layers   : int
    head_dim                 : int
    num_query_heads          : List[int]
    num_kv_heads             : List[int]
    num_gqa_groups           : int
    normalize_qk_projections : bool
    ffn_multipliers          : List[float]
    ffn_dim_divisor          : int
    ffn_with_glu             : bool
    rope_freq_constant       : int
    rope_max_length          : int
    normalization_layer_name : str
    activation_fn_name       : str
    initializer_range        : float
    share_input_output_layers: bool

    # training-only hyperparams (wired through train.py)
    context_size : int = 512
    train_batch_size : int = 16
    max_iterations   : int = 350_000
    warmup_iterations: int = 5_000
    max_lr           : float = 5.3e-3
    min_lr           : float = 5.3e-4
    weight_decay     : float = 0.1
    grad_clip        : float = 1.0
    eval_every       : int = 10_000
    torch_dtype      : str   = "bfloat16"
    dropout          : float  = 0.0

    @classmethod
    def from_json(cls, path: str) -> "SMLMConfig":
        raw = json.loads(pathlib.Path(path).read_text())
        return cls(
            # core model args
            tokenizer_path            = raw["tokenizer_path"],
            checkpoint_dir            = raw["checkpoint_dir"],
            vocab_size                = raw["vocab_size"],
            model_dim                 = raw["model_dim"],
            num_transformer_layers    = raw["num_transformer_layers"],
            head_dim                  = raw["head_dim"],
            num_query_heads           = raw["num_query_heads"],
            num_kv_heads              = raw["num_kv_heads"],
            num_gqa_groups            = raw["num_gqa_groups"],
            normalize_qk_projections  = raw["normalize_qk_projections"],
            ffn_multipliers           = raw["ffn_multipliers"],
            ffn_dim_divisor           = raw["ffn_dim_divisor"],
            ffn_with_glu              = raw["ffn_with_glu"],
            rope_freq_constant        = raw["rope_freq_constant"],
            rope_max_length           = raw["rope_max_length"],
            normalization_layer_name  = raw["normalization_layer_name"],
            activation_fn_name        = raw["activation_fn_name"],
            initializer_range         = raw["initializer_range"],
            share_input_output_layers = raw["share_input_output_layers"],
            dropout                   = raw["dropout"],

            # training args
            context_size      = raw["context_size"],
            train_batch_size  = raw["train_batch_size"],
            max_iterations    = raw["max_iterations"],
            warmup_iterations = raw["warmup_iterations"],
            max_lr            = raw["max_lr"],
            min_lr            = raw["min_lr"],
            weight_decay      = raw["weight_decay"],
            grad_clip         = raw["grad_clip"],
            eval_every        = raw.get("eval_every", 10_000),
            torch_dtype       = raw.get("torch_dtype", "bfloat16"),
        )

# ───────────────────────────────────
# 2) Helpers
# ───────────────────────────────────
RMSNorm = nn.RMSNorm
def repeat_kv(x: mx.array, n: int, axis: int = 2) -> mx.array:
    return mx.repeat(x, n, axis=axis)

# ───────────────────────────────────
# 3) Feed-Forward with optional SwiGLU + activation lookup
# ───────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, cfg: SMLMConfig, idx: int):
        super().__init__()
        mult = cfg.ffn_multipliers[idx]
        inter = int(math.ceil(mult * cfg.model_dim / cfg.ffn_dim_divisor)
                    * cfg.ffn_dim_divisor)
        self.use_glu = cfg.ffn_with_glu
        out_feats = 2*inter if self.use_glu else inter

        self.proj_in  = nn.Linear(cfg.model_dim, out_feats, bias=False)
        self.proj_out = nn.Linear(inter,     cfg.model_dim, bias=False)

        # pick activation
        acts = {
            "swish": nn.SiLU(),
            "gelu":  nn.GELU(),
        }
        self.act = acts[cfg.activation_fn_name]

        # Optional dropout for regularization
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.proj_in(x)
        if self.use_glu:
            y1, y2 = mx.split(y, 2, axis=-1)
            y = self.act(y1) * y2
        else:
            y = self.act(y)
        y = self.dropout(y) 
        return self.proj_out(y)

# ───────────────────────────────────
# 4) Grouped-Query Attention w/ RoPE
# ───────────────────────────────────
class GQAttention(nn.Module):
    def __init__(self, cfg: SMLMConfig, idx: int):
        super().__init__()
        Hq, Hkv, D = cfg.num_query_heads[idx], cfg.num_kv_heads[idx], cfg.head_dim
        self.q_heads, self.kv_heads, self.head_dim = Hq, Hkv, D
        self.groups = cfg.num_gqa_groups

        tot_head = (Hq + 2*Hkv) * D
        self.qkv    = nn.Linear(cfg.model_dim, tot_head, bias=False)
        self.o_proj = nn.Linear(Hq*D, cfg.model_dim, bias=False)
        self.rope   = nn.RoPE(D, base=cfg.rope_freq_constant)

    def __call__(self, x: mx.array, *, mask: mx.array):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, self.q_heads+2*self.kv_heads, self.head_dim)
        qkv = qkv.transpose(0,2,1,3)  # (B, Htot, L, D)

        q, k, v = mx.split(qkv,
                           [self.q_heads, self.q_heads+self.kv_heads],
                           axis=1)
        q, k = self.rope(q), self.rope(k)
        k = repeat_kv(k, self.groups, axis=1)
        v = repeat_kv(v, self.groups, axis=1)

        attn = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=1.0/math.sqrt(self.head_dim),
            mask=mask
        )
        out = attn.transpose(0,2,1,3).reshape(B, L, -1)
        return self.o_proj(out)

# ───────────────────────────────────
# 5) Transformer decoder layer
# ───────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, cfg: SMLMConfig, idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.model_dim, eps=1e-6)
        self.attn  = GQAttention(cfg, idx)
        self.norm2 = RMSNorm(cfg.model_dim, eps=1e-6)
        self.ffn   = FeedForward(cfg, idx)

    def __call__(self, x: mx.array, *, mask: mx.array):
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.ffn(self.norm2(x))

# ───────────────────────────────────
# 6) Full OpenELM Decoder
# ───────────────────────────────────
class OpenELM(nn.Module):
    def __init__(self, cfg: SMLMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = [DecoderLayer(cfg, i)
                       for i in range(cfg.num_transformer_layers)]
        self.final_norm = RMSNorm(cfg.model_dim, eps=1e-6)
        self.lm_head = nn.Linear(cfg.model_dim,
                                 cfg.vocab_size,
                                 bias=False)
        if cfg.share_input_output_layers:
            self.lm_head.weight = self.emb.weight

    def __call__(self, tokens: mx.array) -> mx.array:
        B, L = tokens.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        h = self.emb(tokens)
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.final_norm(h)
        return self.lm_head(h)