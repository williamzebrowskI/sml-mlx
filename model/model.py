

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
#     num_heads                : int            # changed: number of attention heads
#     head_dim                 : int            # dimension per head
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
#             num_heads                 = raw["num_heads"],
#             head_dim                  = raw["head_dim"],
#             normalize_qk_projections  = raw.get("normalize_qk_projections", False),
#             ffn_multipliers           = raw["ffn_multipliers"],
#             ffn_dim_divisor           = raw["ffn_dim_divisor"],
#             ffn_with_glu              = raw["ffn_with_glu"],
#             rope_freq_constant        = raw["rope_freq_constant"],
#             rope_max_length           = raw["rope_max_length"],
#             normalization_layer_name  = raw["normalization_layer_name"],
#             activation_fn_name        = raw["activation_fn_name"],
#             initializer_range         = raw["initializer_range"],
#             share_input_output_layers = raw["share_input_output_layers"],
#             dropout                   = raw.get("dropout", 0.0),

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
# # 4) Standard Multi-Head Attention w/ RoPE
# # ───────────────────────────────────
# class StandardSelfAttention(nn.Module):
#     def __init__(self, cfg: SMLMConfig):
#         super().__init__()
#         assert cfg.model_dim == cfg.num_heads * cfg.head_dim, "model_dim must equal num_heads * head_dim"
#         self.num_heads = cfg.num_heads
#         self.head_dim = cfg.head_dim
#         self.scale = 1.0 / math.sqrt(cfg.head_dim)

#         # projections for Q, K, V
#         self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim, bias=False)
#         self.o_proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
#         self.rope = nn.RoPE(cfg.head_dim, base=cfg.rope_freq_constant)

#     def __call__(self, x: mx.array, *, mask: mx.array):
#         B, L, _ = x.shape  # (batch, seq, model_dim)
#         qkv = self.qkv(x)  # (B, L, 3*model_dim)
#         qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # (B, L, 3, H, D)
#         qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, D)

#         # apply RoPE to q and k
#         q = self.rope(q)
#         k = self.rope(k)

#         # scaled dot-product attention
#         # mx.fast.scaled_dot_product_attention expects (B, H, L, D)
#         attn = mx.fast.scaled_dot_product_attention(
#             q, k, v,
#             scale=self.scale,
#             mask=mask
#         )  # (B, H, L, D)

#         # merge heads
#         out = attn.transpose(0,2,1,3).reshape(B, L, -1)  # (B, L, model_dim)
#         return self.o_proj(out)

# # ───────────────────────────────────
# # 5) Transformer decoder layer
# # ───────────────────────────────────
# class DecoderLayer(nn.Module):
#     def __init__(self, cfg: SMLMConfig, idx: int):
#         super().__init__()
#         self.norm1 = RMSNorm(cfg.model_dim, eps=1e-6)
#         self.attn  = StandardSelfAttention(cfg)
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

# model/model.py
# Tiny decoder-only LM in MLX with fp16-safe attention & loss.
# - Attention softmax runs in fp32 (Q/K/V + mask upcast), then cast back.
# - Cross-entropy computed in fp32.
# - Pre-norm Transformer blocks, RoPE positional encoding, tied embeddings.

import math, json, pathlib, dataclasses
from typing import List

import mlx.core as mx
import mlx.nn as nn

# ───────────────────────────────────
# Config (loaded from config.json)
# ───────────────────────────────────
@dataclasses.dataclass
class SMLMConfig:
    # model
    vocab_size: int
    model_dim: int
    num_transformer_layers: int
    num_heads: int
    head_dim: int
    ffn_multipliers: List[float] = None
    ffn_dim_divisor: int = 256
    ffn_with_glu: bool = True
    rope_freq_constant: int = 10000
    rope_max_length: int = 2048
    normalization_layer_name: str = "rmsnorm"
    activation_fn_name: str = "silu"
    context_size: int = 2048
    share_input_output_layers: bool = True
    dropout: float = 0.0

    # training knobs (used by train.py too)
    max_iterations: int = 1_000_000
    warmup_iterations: int = 20_000
    max_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    torch_dtype: str = "float16"  # "float16" or "bfloat16"
    local_bs: int = 4
    accum_steps: int = 16

    # optional metadata
    tokenizer_path: str = ""
    checkpoint_dir: str = "runs/sml-lm"

    @classmethod
    def from_json(cls, path: str) -> "SMLMConfig":
        raw = json.loads(pathlib.Path(path).read_text())
        cfg = cls(
            vocab_size=raw["vocab_size"],
            model_dim=raw["model_dim"],
            num_transformer_layers=raw["num_transformer_layers"],
            num_heads=raw["num_heads"],
            head_dim=raw["head_dim"],
            ffn_multipliers=raw.get(
                "ffn_multipliers",
                [4.0] * int(raw["num_transformer_layers"])
            ),
            ffn_dim_divisor=int(raw.get("ffn_dim_divisor", 256)),
            ffn_with_glu=bool(raw.get("ffn_with_glu", True)),
            rope_freq_constant=int(raw.get("rope_freq_constant", 10000)),
            rope_max_length=int(raw.get("rope_max_length", raw.get("context_size", 2048))),
            normalization_layer_name=str(raw.get("normalization_layer_name", "rmsnorm")),
            activation_fn_name=str(raw.get("activation_fn_name", "silu")),
            context_size=int(raw.get("context_size", 2048)),
            share_input_output_layers=bool(raw.get("share_input_output_layers", True)),
            dropout=float(raw.get("dropout", 0.0)),
            max_iterations=int(raw.get("max_iterations", 1_000_000)),
            warmup_iterations=int(raw.get("warmup_iterations", 20_000)),
            max_lr=float(raw.get("max_lr", 1e-4)),
            min_lr=float(raw.get("min_lr", 1e-6)),
            weight_decay=float(raw.get("weight_decay", 0.1)),
            grad_clip=float(raw.get("grad_clip", 1.0)),
            torch_dtype=str(raw.get("torch_dtype", raw.get("mlx_dtype", "float16"))),
            local_bs=int(raw.get("local_bs", 4)),
            accum_steps=int(raw.get("accum_steps", 16)),
            tokenizer_path=str(raw.get("tokenizer_path", "")),
            checkpoint_dir=str(raw.get("checkpoint_dir", "runs/sml-lm")),
        )
        return cfg


RMSNorm = nn.RMSNorm


class FeedForward(nn.Module):
    """FFN with optional (Swi)GLU and dropout."""
    def __init__(self, cfg: SMLMConfig, idx: int):
        super().__init__()
        mult = cfg.ffn_multipliers[idx]
        inter = int(math.ceil(mult * cfg.model_dim / cfg.ffn_dim_divisor) * cfg.ffn_dim_divisor)
        self.use_glu = cfg.ffn_with_glu
        out_feats = 2 * inter if self.use_glu else inter

        self.proj_in = nn.Linear(cfg.model_dim, out_feats, bias=False)
        self.proj_out = nn.Linear(inter, cfg.model_dim, bias=False)

        acts = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        self.act = acts[cfg.activation_fn_name]
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


class StandardSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE; numerically safe under fp16."""
    def __init__(self, cfg: SMLMConfig):
        super().__init__()
        assert cfg.model_dim == cfg.num_heads * cfg.head_dim, "model_dim == num_heads * head_dim"
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim, bias=False)
        self.o_proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, base=cfg.rope_freq_constant)

    def __call__(self, x: mx.array, *, mask: mx.array):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)

        # upcast numerically sensitive parts to fp32
        base = q.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        v = v.astype(mx.float32)
        if mask.dtype != mx.float32:
            mask = mask.astype(mx.float32)

        # RoPE in fp32
        q = self.rope(q)
        k = self.rope(k)

        # scaled dot-product attention (fp32)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = attn.transpose(0, 2, 1, 3).reshape(B, L, -1).astype(base)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    """Pre-norm block: x + Attn(Norm(x)) ; x + FFN(Norm(x))"""
    def __init__(self, cfg: SMLMConfig, idx: int):
        super().__init__()
        Norm = RMSNorm if cfg.normalization_layer_name == "rmsnorm" else nn.LayerNorm
        self.norm1 = Norm(cfg.model_dim, eps=1e-6)
        self.norm2 = Norm(cfg.model_dim, eps=1e-6)
        self.attn = StandardSelfAttention(cfg)
        self.ffn = FeedForward(cfg, idx)

    def __call__(self, x: mx.array, *, mask: mx.array):
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.ffn(self.norm2(x))


class OpenELM(nn.Module):
    """Full decoder-only LM with tied embeddings."""
    def __init__(self, cfg: SMLMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = [DecoderLayer(cfg, i) for i in range(cfg.num_transformer_layers)]
        self.final_norm = RMSNorm(cfg.model_dim, eps=1e-6)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        if cfg.share_input_output_layers:
            self.lm_head.weight = self.emb.weight

    def __call__(self, tokens: mx.array) -> mx.array:
        B, L = tokens.shape
        # use additive causal mask; keep it fp32 for stability
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(mx.float32)
        h = self.emb(tokens)
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.final_norm(h)
        return self.lm_head(h)