#!/usr/bin/env python3
"""GPT-2 style MLX language model for the streaming nanoGPT path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    attention_impl: str = "fast"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be > 0")
        if self.n_head <= 0:
            raise ValueError("n_head must be > 0")
        if self.n_embd <= 0 or self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be > 0 and divisible by n_head")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_impl not in {"fast", "vanilla"}:
            raise ValueError("attention_impl must be one of: fast, vanilla")


def _count_leaves(tree: Any) -> int:
    if isinstance(tree, dict):
        return sum(_count_leaves(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_count_leaves(v) for v in tree)
    if isinstance(tree, mx.array):
        size = 1
        for dim in tree.shape:
            size *= int(dim)
        return size
    return 0


def count_parameters(model: nn.Module, *, non_embedding: bool = True) -> int:
    total = _count_leaves(model.parameters())
    if non_embedding:
        total -= int(model.wpe.weight.size)
    return total


def _map_named(tree: Any, fn, prefix: str = ""):
    if isinstance(tree, dict):
        return {k: _map_named(v, fn, f"{prefix}.{k}" if prefix else k) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_map_named(v, fn, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(tree)]
    if isinstance(tree, tuple):
        return tuple(_map_named(v, fn, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(tree))
    if isinstance(tree, mx.array):
        return fn(prefix, tree)
    return tree


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.attention_impl = cfg.attention_impl
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(cfg.block_size)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, channels = x.shape
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        def split_heads(t: mx.array) -> mx.array:
            t = t.reshape(bsz, seqlen, self.n_head, self.head_dim)
            return t.transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        if self.attention_impl == "fast" and self.dropout == 0.0:
            attn = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=self.causal_mask[:seqlen, :seqlen],
            )
        else:
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = scores + self.causal_mask[:seqlen, :seqlen][None, None, :, :]
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
            probs = self.attn_dropout(probs)
            attn = mx.matmul(probs, v)

        y = attn.transpose(0, 2, 1, 3).reshape(bsz, seqlen, channels)
        y = self.c_proj(y)
        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LM(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        proj_std = 0.02 / math.sqrt(2.0 * self.config.n_layer)

        def init(path: str, value: mx.array) -> mx.array:
            if path.endswith("bias"):
                return mx.zeros(value.shape, dtype=value.dtype)
            if path.endswith("ln_1.weight") or path.endswith("ln_2.weight") or path.endswith("ln_f.weight"):
                return mx.ones(value.shape, dtype=value.dtype)
            std = proj_std if path.endswith("c_proj.weight") else 0.02
            return (mx.random.normal(shape=value.shape) * std).astype(value.dtype)

        self.update(_map_named(self.parameters(), init))
        mx.eval(self.parameters())

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("Cannot increase block_size with crop_block_size")
        if block_size == self.config.block_size:
            return
        self.config.block_size = block_size
        self.wpe.weight = self.wpe.weight[:block_size]
        for block in self.h:
            block.attn.causal_mask = block.attn.causal_mask[:block_size, :block_size]
        mx.eval(self.parameters())

    def logits(self, idx: mx.array) -> mx.array:
        batch, seq = idx.shape
        if seq > self.config.block_size:
            raise ValueError(f"Cannot forward length {seq}, block_size is {self.config.block_size}")
        pos = mx.arange(seq, dtype=mx.int32)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)[None, :, :]
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return mx.matmul(x, self.wte.weight.transpose(1, 0))

    def __call__(
        self,
        idx: mx.array,
        targets: Optional[mx.array] = None,
        ignore_index: int = -1,
    ) -> Dict[str, mx.array]:
        logits = self.logits(idx)
        out: Dict[str, mx.array] = {"logits": logits}
        if targets is None:
            out["logits"] = logits[:, [-1], :]
            return out
        per_token = losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
        if ignore_index >= 0:
            mask = (targets != ignore_index).astype(per_token.dtype)
            denom = mask.sum() + 1e-6
            loss = (per_token * mask).sum() / denom
        else:
            loss = per_token.mean()
        out["loss"] = loss
        return out
