#!/usr/bin/env python3
"""MLX-only transformer model for small causal language modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    bias: bool = False
    attention_impl: str = "fast"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.mlp_ratio <= 1.0:
            raise ValueError("mlp_ratio must be > 1.0")
        if self.attention_impl not in {"fast", "vanilla"}:
            raise ValueError("attention_impl must be one of: fast, vanilla")


def _tree_leaves(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _tree_leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _tree_leaves(v)
    else:
        yield tree


def count_parameters(model: nn.Module) -> int:
    total = 0
    for leaf in _tree_leaves(model.parameters()):
        if isinstance(leaf, mx.array):
            n = 1
            for d in leaf.shape:
                n *= int(d)
            total += n
    return total


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.max_seq_len = cfg.max_seq_len

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.rope = nn.RoPE(self.head_dim, base=cfg.rope_base)
        self.causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(
            cfg.max_seq_len
        )
        self.attention_impl = cfg.attention_impl

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        def split_heads(t: mx.array) -> mx.array:
            t = t.reshape(bsz, seqlen, self.n_heads, self.head_dim)
            return t.transpose(0, 2, 1, 3)

        q = self.rope(split_heads(q))
        k = self.rope(split_heads(k))
        v = split_heads(v)

        if cache is not None:
            k_cache, v_cache = cache
            if k_cache is not None and v_cache is not None:
                k = mx.concatenate([k_cache, k], axis=2)
                v = mx.concatenate([v_cache, v], axis=2)

        q_len = q.shape[2]
        k_len = k.shape[2]
        if q_len > self.max_seq_len or k_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {q_len}/{k_len} exceeds max_seq_len={self.max_seq_len}"
            )
        mask = self.causal_mask[:q_len, :k_len]

        scale = 1.0 / math.sqrt(self.head_dim)
        if self.attention_impl == "fast":
            attn = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mask,
            )
        else:
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = scores + mask[None, None, :, :]
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
            attn = mx.matmul(probs, v)
        out = attn.transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
        out = self.proj(out)
        return out, (k, v)


class FeedForward(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        hidden_dim = int(cfg.d_model * cfg.mlp_ratio)
        self.up = nn.Linear(cfg.d_model, hidden_dim, bias=cfg.bias)
        self.act = nn.SiLU()
        self.down = nn.Linear(hidden_dim, cfg.d_model, bias=cfg.bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(self.act(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model)
        self.output_bias = mx.zeros((cfg.vocab_size,))

    def logits(
        self,
        input_ids: mx.array,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        x = self.embed(input_ids)
        if caches is None:
            caches = [None] * len(self.blocks)

        new_caches = []
        for block, block_cache in zip(self.blocks, caches):
            x, next_cache = block(x, cache=block_cache)
            new_caches.append(next_cache)

        x = self.norm(x)
        weight = self.embed.weight
        logits = mx.matmul(x, weight.transpose(1, 0)) + self.output_bias
        return logits, new_caches

    def __call__(
        self,
        input_ids: mx.array,
        targets: Optional[mx.array] = None,
        ignore_index: int = -100,
    ) -> dict:
        logits, _ = self.logits(input_ids, caches=None)
        out = {"logits": logits}
        if targets is None:
            return out

        # Keep CE math in fp32 for stability when model weights/logits are fp16.
        per_token = losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
        if ignore_index >= 0:
            mask = (targets != ignore_index).astype(per_token.dtype)
            denom = mask.sum() + 1e-6
            loss = (per_token * mask).sum() / denom
        else:
            loss = per_token.mean()
        out["loss"] = loss
        return out

    def step(
        self,
        token_ids: mx.array,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        # token_ids shape: [B, 1]
        logits, new_caches = self.logits(token_ids, caches=caches)
        return logits[:, -1, :], new_caches
