# # model/model.py
# import math, json, pathlib, dataclasses
# from typing import List

# import mlx.core as mx
# import mlx.nn   as nn

# # ───────────────────────────────────
# # 1) Config (all args pulled from JSON)
# # ───────────────────────────────────
# @dataclasses.dataclass
# class SMLMConfig:
#     # transformer
#     tokenizer_path           : str
#     checkpoint_dir           : str
#     vocab_size               : int
#     model_dim                : int
#     num_transformer_layers   : int
#     head_dim                 : int
#     num_query_heads          : List[int]
#     num_kv_heads             : List[int]
#     num_gqa_groups           : int
#     normalize_qk_projections : bool
#     ffn_multipliers          : List[float]
#     ffn_dim_divisor          : int
#     ffn_with_glu             : bool
#     rope_freq_constant       : int
#     rope_max_length          : int
#     normalization_layer_name : str
#     activation_fn_name       : str
#     initializer_range        : float
#     share_input_output_layers: bool

#     # training-only hyperparams (wired through train.py)
#     context_size : int = 512
#     train_batch_size : int = 16
#     max_iterations   : int = 350_000
#     warmup_iterations: int = 5_000
#     max_lr           : float = 5.3e-3
#     min_lr           : float = 5.3e-4
#     weight_decay     : float = 0.1
#     grad_clip        : float = 1.0
#     eval_every       : int = 10_000
#     torch_dtype      : str   = "bfloat16"
#     dropout          : float  = 0.0

#     @classmethod
#     def from_json(cls, path: str) -> "SMLMConfig":
#         raw = json.loads(pathlib.Path(path).read_text())
#         return cls(
#             # core model args
#             tokenizer_path            = raw["tokenizer_path"],
#             checkpoint_dir            = raw["checkpoint_dir"],
#             vocab_size                = raw["vocab_size"],
#             model_dim                 = raw["model_dim"],
#             num_transformer_layers    = raw["num_transformer_layers"],
#             head_dim                  = raw["head_dim"],
#             num_query_heads           = raw["num_query_heads"],
#             num_kv_heads              = raw["num_kv_heads"],
#             num_gqa_groups            = raw["num_gqa_groups"],
#             normalize_qk_projections  = raw["normalize_qk_projections"],
#             ffn_multipliers           = raw["ffn_multipliers"],
#             ffn_dim_divisor           = raw["ffn_dim_divisor"],
#             ffn_with_glu              = raw["ffn_with_glu"],
#             rope_freq_constant        = raw["rope_freq_constant"],
#             rope_max_length           = raw["rope_max_length"],
#             normalization_layer_name  = raw["normalization_layer_name"],
#             activation_fn_name        = raw["activation_fn_name"],
#             initializer_range         = raw["initializer_range"],
#             share_input_output_layers = raw["share_input_output_layers"],
#             dropout                   = raw["dropout"],

#             # training args
#             context_size      = raw["context_size"],
#             train_batch_size  = raw["train_batch_size"],
#             max_iterations    = raw["max_iterations"],
#             warmup_iterations = raw["warmup_iterations"],
#             max_lr            = raw["max_lr"],
#             min_lr            = raw["min_lr"],
#             weight_decay      = raw["weight_decay"],
#             grad_clip         = raw["grad_clip"],
#             eval_every        = raw.get("eval_every", 10_000),
#             torch_dtype       = raw.get("torch_dtype", "bfloat16"),
#         )

# # ───────────────────────────────────
# # 2) Helpers
# # ───────────────────────────────────
# RMSNorm = nn.RMSNorm
# def repeat_kv(x: mx.array, n: int, axis: int = 2) -> mx.array:
#     return mx.repeat(x, n, axis=axis)

# # ───────────────────────────────────
# # 3) Feed-Forward with optional SwiGLU + activation lookup
# # ───────────────────────────────────
# class FeedForward(nn.Module):
#     def __init__(self, cfg: SMLMConfig, idx: int):
#         super().__init__()
#         mult = cfg.ffn_multipliers[idx]
#         inter = int(math.ceil(mult * cfg.model_dim / cfg.ffn_dim_divisor)
#                     * cfg.ffn_dim_divisor)
#         self.use_glu = cfg.ffn_with_glu
#         out_feats = 2*inter if self.use_glu else inter

#         self.proj_in  = nn.Linear(cfg.model_dim, out_feats, bias=False)
#         self.proj_out = nn.Linear(inter,     cfg.model_dim, bias=False)

#         # pick activation
#         acts = {
#             "swish": nn.SiLU(),
#             "gelu":  nn.GELU(),
#         }
#         self.act = acts[cfg.activation_fn_name]

#         # Optional dropout for regularization
#         self.dropout = nn.Dropout(cfg.dropout)

#     def __call__(self, x: mx.array) -> mx.array:
#         y = self.proj_in(x)
#         if self.use_glu:
#             y1, y2 = mx.split(y, 2, axis=-1)
#             y = self.act(y1) * y2
#         else:
#             y = self.act(y)
#         y = self.dropout(y) 
#         return self.proj_out(y)

# # ───────────────────────────────────
# # 4) Grouped-Query Attention w/ RoPE
# # ───────────────────────────────────
# class GQAttention(nn.Module):
#     def __init__(self, cfg: SMLMConfig, idx: int):
#         super().__init__()
#         Hq, Hkv, D = cfg.num_query_heads[idx], cfg.num_kv_heads[idx], cfg.head_dim
#         self.q_heads, self.kv_heads, self.head_dim = Hq, Hkv, D
#         self.groups = cfg.num_gqa_groups

#         tot_head = (Hq + 2*Hkv) * D
#         self.qkv    = nn.Linear(cfg.model_dim, tot_head, bias=False)
#         self.o_proj = nn.Linear(Hq*D, cfg.model_dim, bias=False)
#         self.rope   = nn.RoPE(D, base=cfg.rope_freq_constant)

#     def __call__(self, x: mx.array, *, mask: mx.array):
#         B, L, _ = x.shape
#         qkv = self.qkv(x).reshape(B, L, self.q_heads+2*self.kv_heads, self.head_dim)
#         qkv = qkv.transpose(0,2,1,3)  # (B, Htot, L, D)

#         q, k, v = mx.split(qkv,
#                            [self.q_heads, self.q_heads+self.kv_heads],
#                            axis=1)
#         q, k = self.rope(q), self.rope(k)
#         k = repeat_kv(k, self.groups, axis=1)
#         v = repeat_kv(v, self.groups, axis=1)

#         attn = mx.fast.scaled_dot_product_attention(
#             q, k, v,
#             scale=1.0/math.sqrt(self.head_dim),
#             mask=mask
#         )
#         out = attn.transpose(0,2,1,3).reshape(B, L, -1)
#         return self.o_proj(out)

# # ───────────────────────────────────
# # 5) Transformer decoder layer
# # ───────────────────────────────────
# class DecoderLayer(nn.Module):
#     def __init__(self, cfg: SMLMConfig, idx: int):
#         super().__init__()
#         self.norm1 = RMSNorm(cfg.model_dim, eps=1e-6)
#         self.attn  = GQAttention(cfg, idx)
#         self.norm2 = RMSNorm(cfg.model_dim, eps=1e-6)
#         self.ffn   = FeedForward(cfg, idx)

#     def __call__(self, x: mx.array, *, mask: mx.array):
#         x = x + self.attn(self.norm1(x), mask=mask)
#         return x + self.ffn(self.norm2(x))

# # ───────────────────────────────────
# # 6) Full OpenELM Decoder
# # ───────────────────────────────────
# class OpenELM(nn.Module):
#     def __init__(self, cfg: SMLMConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
#         self.layers = [DecoderLayer(cfg, i)
#                        for i in range(cfg.num_transformer_layers)]
#         self.final_norm = RMSNorm(cfg.model_dim, eps=1e-6)
#         self.lm_head = nn.Linear(cfg.model_dim,
#                                  cfg.vocab_size,
#                                  bias=False)
#         if cfg.share_input_output_layers:
#             self.lm_head.weight = self.emb.weight

#     def __call__(self, tokens: mx.array) -> mx.array:
#         B, L = tokens.shape
#         mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
#         h = self.emb(tokens)
#         for layer in self.layers:
#             h = layer(h, mask=mask)
#         h = self.final_norm(h)
#         return self.lm_head(h)

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
    num_heads                : int            # changed: number of attention heads
    head_dim                 : int            # dimension per head
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
            num_heads                 = raw["num_heads"],
            head_dim                  = raw["head_dim"],
            normalize_qk_projections  = raw.get("normalize_qk_projections", False),
            ffn_multipliers           = raw["ffn_multipliers"],
            ffn_dim_divisor           = raw["ffn_dim_divisor"],
            ffn_with_glu              = raw["ffn_with_glu"],
            rope_freq_constant        = raw["rope_freq_constant"],
            rope_max_length           = raw["rope_max_length"],
            normalization_layer_name  = raw["normalization_layer_name"],
            activation_fn_name        = raw["activation_fn_name"],
            initializer_range         = raw["initializer_range"],
            share_input_output_layers = raw["share_input_output_layers"],
            dropout                   = raw.get("dropout", 0.0),

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
# 4) Standard Multi-Head Attention w/ RoPE
# ───────────────────────────────────
class StandardSelfAttention(nn.Module):
    def __init__(self, cfg: SMLMConfig):
        super().__init__()
        assert cfg.model_dim == cfg.num_heads * cfg.head_dim, "model_dim must equal num_heads * head_dim"
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        # projections for Q, K, V
        self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim, bias=False)
        self.o_proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, base=cfg.rope_freq_constant)

    def __call__(self, x: mx.array, *, mask: mx.array):
        B, L, _ = x.shape  # (batch, seq, model_dim)
        qkv = self.qkv(x)  # (B, L, 3*model_dim)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # (B, L, 3, H, D)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, D)

        # apply RoPE to q and k
        q = self.rope(q)
        k = self.rope(k)

        # scaled dot-product attention
        # mx.fast.scaled_dot_product_attention expects (B, H, L, D)
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask=mask
        )  # (B, H, L, D)

        # merge heads
        out = attn.transpose(0,2,1,3).reshape(B, L, -1)  # (B, L, model_dim)
        return self.o_proj(out)

# ───────────────────────────────────
# 5) Transformer decoder layer
# ───────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, cfg: SMLMConfig, idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.model_dim, eps=1e-6)
        self.attn  = StandardSelfAttention(cfg)
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