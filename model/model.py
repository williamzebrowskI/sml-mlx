#!/usr/bin/env python
# model/model.py
# A ~100M-parameter LM in MLX with HF streaming + ring DDP,
# using a SentencePiece BPE tokenizer ONLY (no byte-tokenizer fallback).
# Upgrades:
# - RMSNorm + SiLU + RoPE
# - Background prefetch for batches
# - KV-cache for fast inference
# - safetensors save/load + resume with rank-0 broadcast
# - Runtime PAD/MASK ids follow the active SPM tokenizer

import os
os.environ.setdefault("MX_MAX_INFLIGHT_CMDS", "1")

import math, time, json, gzip, requests, argparse, threading, queue
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from safetensors.numpy import load_file as safetensors_load  # loading .safetensors
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim
from mlx.utils import tree_map

# ---------------------------
# SentencePiece tokenizer (BPE) wrapper
# ---------------------------

class SPMTokenizer:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SPM model not found: {model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.vocab_size()
        if self.pad_id is None or self.pad_id < 0:
            self.pad_id = 0

    def encode(self, s: str) -> List[int]:
        ids = [self.bos_id]
        ids += self.sp.encode(s, out_type=int)
        ids += [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        core = [t for t in ids if t not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(core)

# ------- global tokenizer (SPM only) -------
TOK: SPMTokenizer | None = None
VOCAB_SIZE: int = 0
PAD_ID_RUNTIME: int = 0
MASK_ID_FOR_LOSS: int = 0

def set_tokenizer(spm_path: str):
    """Require an SPM model and initialize globals."""
    global TOK, VOCAB_SIZE, PAD_ID_RUNTIME, MASK_ID_FOR_LOSS
    TOK = SPMTokenizer(spm_path)
    VOCAB_SIZE = TOK.vocab_size
    PAD_ID_RUNTIME = TOK.pad_id
    MASK_ID_FOR_LOSS = PAD_ID_RUNTIME
    print(f"[tok] Using SentencePiece BPE: {spm_path} (vocab={VOCAB_SIZE}, pad_id={PAD_ID_RUNTIME})", flush=True)

# ---------------------------
# RoPE helper (per-head dim)
# ---------------------------

def make_rope(head_dim: int, base: float = 10000.0):
    return nn.RoPE(head_dim, base=base)

# ---------------------------
# Transformer (fast SDPA + cached additive mask) with RoPE
# ---------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dh = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,    bias=False)
        self.max_seq = max_seq
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(max_seq)
        self.rope = make_rope(self.dh, base=rope_base)

    def __call__(self, x: mx.array, cache: Optional[Tuple[mx.array, mx.array]] = None) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        def split_heads(t):
            t = t.reshape(B, T, self.h, self.dh)
            return t.transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q = self.rope(q)
        k = self.rope(k)

        if cache is not None:
            k_cache, v_cache = cache
            if k_cache is not None and v_cache is not None:
                k = mx.concatenate([k_cache, k], axis=2)
                v = mx.concatenate([v_cache, v], axis=2)

        Tq = q.shape[2]
        Tk = k.shape[2]
        mask = self._mask[:Tq, :Tk]

        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.dh), mask=mask
        )
        ctx = attn.transpose(0, 2, 1, 3).reshape(B, Tq, D)
        out = self.proj(ctx)
        return out, (k, v)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq: int,
        mlp_mult: int = 4,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.ln1  = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq, rope_base=rope_base)
        self.ln2  = nn.RMSNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_mult * d_model, d_model, bias=False),
        )

    def __call__(self, x: mx.array, cache: Optional[Tuple[mx.array, mx.array]] = None) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        a, new_cache = self.attn(self.ln1(x), cache=cache)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, new_cache

@dataclass
class TinyGPConfig:
    vocab_size: int
    d_model: int = 768        # 100M config: 768-dim model
    n_heads: int = 12         # 12 heads → 64-dim per head
    n_layers: int = 12        # 12 transformer blocks
    max_seq: int = 256
    max_grad_norm: float = 1.0
    mlp_mult: int = 4
    rope_base: float = 10000.0

def load_tinygplm_config(path: str, vocab_size: int, max_seq: int) -> TinyGPConfig:
    """
    Load a TinyGPLM config from a JSON file and merge with runtime vocab_size/max_seq.
    """
    with open(path, "r") as f:
        cfg_json = json.load(f)

    return TinyGPConfig(
        vocab_size=vocab_size,
        d_model=cfg_json.get("d_model", 768),
        n_heads=cfg_json.get("n_heads", 12),
        n_layers=cfg_json.get("n_layers", 12),
        max_seq=max_seq,
        max_grad_norm=1.0,
        mlp_mult=cfg_json.get("mlp_mult", 4),
        rope_base=cfg_json.get("rope_base", 10000.0),
    )

class TinyGPLM(nn.Module):
    def __init__(self, cfg: TinyGPConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks  = [
            TransformerBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.max_seq,
                mlp_mult=cfg.mlp_mult,
                rope_base=cfg.rope_base,
            )
            for _ in range(cfg.n_layers)
        ]
        self.ln_f    = nn.RMSNorm(cfg.d_model)
        self.out_bias = mx.zeros((cfg.vocab_size,))
        self.pos_cache = None  # RoPE handles positions

    def logits(self, x_ids: mx.array) -> mx.array:
        B, T = x_ids.shape
        x = self.tok_emb(x_ids)
        for blk in self.blocks:
            x, _ = blk(x, cache=None)
        x = self.ln_f(x)
        W = self.tok_emb.weight
        return mx.matmul(x, W.transpose(1, 0)) + self.out_bias

    def __call__(self, x_ids: mx.array, targets: Optional[mx.array] = None):
        lg = self.logits(x_ids)
        if targets is None:
            return {"logits": lg}
        ce = losses.cross_entropy(lg, targets, reduction="none")
        mask = (targets != MASK_ID_FOR_LOSS).astype(mx.float32)
        loss = (ce * mask).sum() / (mask.sum() + 1e-6)
        return {"logits": lg, "loss": loss}

    def step(self, tok_id: mx.array, caches: Optional[List[Tuple[mx.array, mx.array]]]) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        B, L = tok_id.shape
        assert L == 1
        x = self.tok_emb(tok_id)
        if caches is None:
            caches = [None] * len(self.blocks)
        new_caches = []
        for blk, cache in zip(self.blocks, caches):
            x, new_cache = blk(x, cache=cache)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        W = self.tok_emb.weight
        logits = mx.matmul(x, W.transpose(1, 0)).squeeze(axis=1) + self.out_bias
        return logits, new_caches

# ---------------------------
# HF streaming helpers
# ---------------------------

def hf_text_iterator(name, config, split, field, world_size, rank, trust_remote_code=False):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=trust_remote_code)
    try:
        ds = ds.shard(num_shards=world_size, index=rank)
        for ex in ds:
            yield {"text": ex.get(field, "")}
    except Exception:
        i = 0
        for ex in ds:
            if i % world_size == rank:
                yield {"text": ex.get(field, "")}
            i += 1

# ---------------------------
# Distributed + utils
# ---------------------------

def init_dist(backend="ring", expected_world=None):
    group = mx.distributed.init(backend=backend)
    size = group.size() if callable(getattr(group,"size",None)) else int(getattr(group,"size",1))
    rank = group.rank() if callable(getattr(group,"rank",None)) else int(getattr(group,"rank",0))
    print(f"[boot] rank={rank} size={size} backend={backend}", flush=True)
    if expected_world is not None and size != expected_world:
        raise RuntimeError(f"Expected world size {expected_world}, got {size}")
    if rank == 0:
        print(f"[dist] backend={backend} world={size} (expected={expected_world})")
    return group, size, rank

def allreduce_grads(grads, world):
    if world == 1:
        return grads
    def _reduce(g):
        if isinstance(g, mx.array):
            try:
                return mx.distributed.all_sum(g, stream=mx.cpu) / world
            except TypeError:
                return mx.distributed.all_sum(g) / world
        return g
    return tree_map(_reduce, grads)

def grad_norm_from_tree(tree) -> float:
    # Temporarily disabled to avoid GPU-heavy norm computation in multi-host
    return 0.0

def clip_global(tree, max_norm):
    # No clipping requested
    if max_norm <= 0:
        return tree, 0.0

    # If we ever re-enable this, do it carefully (CPU or rank-0 only)
    gnorm = grad_norm_from_tree(tree)
    if gnorm <= max_norm:
        return tree, gnorm

    scale = max_norm / (gnorm + 1e-6)
    clipped = tree_map(
        lambda x: x * scale if isinstance(x, mx.array) else x,
        tree,
    )
    return clipped, gnorm

# ---------- safetensors save/load ----------

def flatten_params(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            flat.update(flatten_params(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            flat.update(flatten_params(v, key))
    elif isinstance(obj, mx.array):
        key = prefix if prefix else "param"
        flat[key] = obj
    return flat

def save_safetensors_model(path: str, model: nn.Module):
    flat = flatten_params(model.parameters())
    if not flat:
        print(f"[save] ⚠️ no tensors to save for {path}", flush=True)
        return False
    try:
        mx.save_safetensors(path, flat)
        print(f"[save] wrote {path} ({len(flat)} tensors)", flush=True)
        return True
    except Exception as e:
        print(f"[save] ❌ safetensors failed for {path}: {e!r}", flush=True)
        return False

def load_safetensors_model(path: str, model: nn.Module) -> bool:
    if not os.path.exists(path):
        print(f"[load] ❌ checkpoint not found: {path}", flush=True)
        return False
    try:
        np_flat = safetensors_load(path)
    except Exception as e:
        print(f"[load] ❌ failed to read {path}: {e!r}", flush=True)
        return False

    def assign(tree, prefix=""):
        if isinstance(tree, dict):
            out = {}
            for k, v in tree.items():
                key = f"{prefix}.{k}" if prefix else k
                out[k] = assign(v, key)
            return out
        elif isinstance(tree, (list, tuple)):
            out = []
            for i, v in enumerate(tree):
                key = f"{prefix}.{i}" if prefix else str(i)
                out.append(assign(v, key))
            return type(tree)(out)
        elif isinstance(tree, mx.array):
            key = prefix if prefix else "param"
            if key in np_flat:
                arr = np_flat[key]
                t = mx.array(arr)
                if tuple(t.shape) != tuple(tree.shape):
                    print(f"[load] ⚠️ shape mismatch for {key}: file {t.shape} vs model {tree.shape}", flush=True)
                return t.astype(tree.dtype)
            else:
                return tree
        else:
            return tree

    updated = assign(model.parameters())
    model.update(updated)
    mx.eval(model.parameters())
    print(f"[load] loaded {len(np_flat)} tensors from {path}", flush=True)
    return True

# ---------- broadcast-from-rank-0 helper ----------

def _broadcast_from_rank0(model: nn.Module, rank: int):
    # Helpful logs to see which rank is doing what
    print(f"[rank {rank}] starting _broadcast_from_rank0", flush=True)

    def push(x):
        if not isinstance(x, mx.array):
            return x
        x0 = x if rank == 0 else mx.zeros_like(x)
        # Broadcast on CPU stream to avoid Metal GPU timeout
        try:
            return mx.distributed.all_sum(x0, stream=mx.cpu)
        except TypeError:
            return mx.distributed.all_sum(x0)

    bcast = tree_map(push, model.parameters())
    model.update(bcast)
    mx.eval(model.parameters())

    print(f"[rank {rank}] finished _broadcast_from_rank0", flush=True)

# ---------- fwd+bwd (microstep) ----------

def _make_compiled_step(model: nn.Module):
    """
    Build a per-microstep function that returns (loss, grads) and
    matches the call pattern: step(X, Y).
    We intentionally do NOT mx.compile this, because combining
    mx.compile + nn.value_and_grad has been unstable in some setups.
    """
    def loss_fn(x, y):
        return model(x, y)["loss"]

    # This wires gradients w.r.t. model.trainable_parameters()
    step = nn.value_and_grad(model, loss_fn)
    return step

# ---------------------------
# Training
# ---------------------------

def train_hf_distributed(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    trust_remote_code: bool = False,
    max_steps: int = 100_000,
    seq_len: int = 256,
    batch_size: int = 1,
    accum_steps: int = 1,
    lr: float = 3e-4,
    wd: float = 0.1,
    backend: str = "ring",
    expected_world: int | None = None,
    log_every: int = 10,
    per_rank_logs: bool = False,
    save_dir: str = "model/checkpoints_spm",
    save_every: int = 5_000,
    eval_every: int = 0,
    eval_prompt: str = "Hello, my name is ",
    eval_tokens: int = 50,
    resume_ckpt: Optional[str] = None,
    config_path: Optional[str] = None,
    lr_decay_start_step: Optional[int] = None,
):
    _, world, rank = init_dist(backend=backend, expected_world=expected_world)
    print(f"[rank {rank}] entering train_hf_distributed", flush=True)

    # Rank 0 uses full seq_len; other ranks use a shorter local_seq_len
    if world > 1 and rank != 0:
        local_seq_len = max(1, seq_len // 4)  # e.g., 2048 -> 512
    else:
        local_seq_len = seq_len
    print(f"[rank {rank}] using local_seq_len={local_seq_len} (global seq_len={seq_len})", flush=True)

    # Build config from JSON (if provided) or use defaults
    if config_path is not None:
        cfg = load_tinygplm_config(config_path, vocab_size=VOCAB_SIZE, max_seq=seq_len)
    else:
        cfg = TinyGPConfig(
            vocab_size=VOCAB_SIZE,
            d_model=768,
            n_heads=12,
            n_layers=12,
            max_seq=seq_len,
            max_grad_norm=1.0,
        )

    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    print(f"[rank {rank}] model initialized (vocab={VOCAB_SIZE}, d_model={cfg.d_model}, layers={cfg.n_layers})", flush=True)

    # step bound to this model; it will be callable as step(X, Y)
    step = _make_compiled_step(model)

    update_step = 0
    if resume_ckpt and rank == 0:
        if load_safetensors_model(resume_ckpt, model):
            try:
                stem = os.path.splitext(os.path.basename(resume_ckpt))[0]
                update_step = int(stem.split("_")[-1])
            except Exception:
                update_step = 0
            print(f"[rank 0] resume from {resume_ckpt} at step {update_step}", flush=True)
        else:
            print(f"[rank 0] resume requested but failed, training from scratch", flush=True)
    mx.eval(model.parameters())
    if world > 1:
        print(f"[rank {rank}] broadcasting parameters from rank 0", flush=True)
        _broadcast_from_rank0(model, rank)

    # Approximate number of dataset examples to skip when resuming.
    # For world=1 and batch_size=1, each step uses exactly one example.
    skip_examples = update_step * batch_size if resume_ckpt else 0
    if rank == 0 and skip_examples > 0:
        print(
            f"[rank 0] will skip approximately {skip_examples} examples in the data stream "
            f"to align with resume step {update_step}",
            flush=True,
        )

    opt = optim.AdamW(lr, weight_decay=wd)
    base_lr = lr

    # Decide when to start LR decay
    if lr_decay_start_step is not None:
        decay_start = lr_decay_start_step
    else:
        decay_start = int(max_steps * 0.5)

    sample_iter = hf_text_iterator(dataset_name, dataset_config, split, text_field, world, rank, trust_remote_code)

    def batch_iterator():
        X = mx.zeros((batch_size, local_seq_len), dtype=mx.int32)
        Y = mx.zeros((batch_size, local_seq_len), dtype=mx.int32)
        filled = 0
        skipped = 0

        for ex in sample_iter:
            # Skip examples if resuming, before building batches
            if skipped < skip_examples:
                skipped += 1
                continue

            text = ex.get("text", "")
            if not text:
                continue
            ids = TOK.encode(text)
            if len(ids) < 2:
                continue
            ids = ids[: local_seq_len + 1]
            x_ids, y_ids = ids[:-1], ids[1:]
            if len(x_ids) < local_seq_len:
                pad = local_seq_len - len(x_ids)
                x_ids = x_ids + [PAD_ID_RUNTIME] * pad
                y_ids = y_ids + [PAD_ID_RUNTIME] * pad
            X[filled] = mx.array(x_ids, dtype=mx.int32)
            Y[filled] = mx.array(y_ids, dtype=mx.int32)
            filled += 1
            if filled == batch_size:
                yield X, Y
                X = mx.zeros((batch_size, local_seq_len), dtype=mx.int32)
                Y = mx.zeros((batch_size, local_seq_len), dtype=mx.int32)
                filled = 0

    def make_prefetcher(batch_iter, maxsize=8):
        q = queue.Queue(maxsize=maxsize)
        def _runner():
            for xy in batch_iter():
                q.put(xy)
            q.put(None)
        th = threading.Thread(target=_runner, daemon=True)
        th.start()
        def _get():
            item = q.get()
            if item is None:
                return None, None
            return item
        return _get

    next_batch = make_prefetcher(batch_iterator, maxsize=8)

    micro_accum = 0
    accum_grads = None
    last = time.time()

    for _ in range(max_steps * max(1, accum_steps) * 10**9):
        X, Y = next_batch()
        if X is None:
            break

        loss, grads = step(X, Y)
        mx.eval(loss, grads)

        # 1) NaN/Inf checks
        if bool(mx.isnan(loss)) or bool(mx.isinf(loss)):
            if rank == 0:
                print(f"[{update_step}] ⚠️ NaN/Inf loss; skipping micro-step", flush=True)
            continue

        def _bad(g):
            return isinstance(g, mx.array) and (bool(mx.isnan(g).any()) or bool(mx.isinf(g).any()))
        if any(_bad(g) for g in tree_map(lambda x: x, grads)):
            if rank == 0:
                print(f"[{update_step}] ⚠️ NaN/Inf grads; skipping micro-step", flush=True)
            continue

        # 2) Grad scaling & accumulation
        grads = tree_map(
            lambda g: g / accum_steps if isinstance(g, mx.array) else g,
            grads,
        )
        accum_grads = grads if accum_grads is None else tree_map(
            lambda a, g: a + g if isinstance(a, mx.array) else a,
            accum_grads,
            grads,
        )
        micro_accum += 1

        if micro_accum == accum_steps:
            global_grads = allreduce_grads(accum_grads, world)

            # Disable clipping in multi-host to avoid Metal GPU timeout
            max_norm = model.cfg.max_grad_norm
            if world > 1:
                max_norm = 0.0

            clipped, gnorm = clip_global(global_grads, max_norm)

            # ---- Cosine LR decay after decay_start ----
            if update_step >= decay_start:
                t = (update_step - decay_start) / max(1, max_steps - decay_start)
                t = min(1.0, max(0.0, t))
                factor = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
                opt.learning_rate = base_lr * factor
            else:
                opt.learning_rate = base_lr
            # -------------------------------------------

            opt.update(model, clipped)

            now = time.time()
            dt = now - last
            last = now
            if update_step % log_every == 0:
                toks_local_s = (batch_size * local_seq_len * accum_steps) / max(1e-9, dt)
                ppl = math.exp(min(20.0, float(loss.item())))
                lr_cur = getattr(opt, "learning_rate", None)
                lr_str = f" lr={lr_cur:.6e}" if lr_cur is not None else ""
                if per_rank_logs:
                    print(f"[{update_step}] rank={rank} loss={loss.item():.4f} "
                          f"ppl={ppl:.2f} grad_norm={gnorm:.3f}{lr_str} "
                          f"tok/s={toks_local_s:.0f}")
                if rank == 0 and not per_rank_logs:
                    print(f"[{update_step}] loss={loss.item():.4f} ppl={ppl:.2f} "
                          f"grad_norm={gnorm:.3f}{lr_str} "
                          f"tok/s≈{toks_local_s * world:.0f}")

            if rank == 0 and update_step > 0 and update_step % save_every == 0:
                path = os.path.join(save_dir, f"ckpt_{update_step:06d}.safetensors")
                ok = save_safetensors_model(path, model)
                if not ok:
                    print(f"[{update_step}] ⚠️ checkpoint NOT written: {path}", flush=True)

            if rank == 0 and eval_every and update_step % eval_every == 0:
                try:
                    run_eval_samples(model, update_step, eval_prompt, eval_tokens)
                except Exception as e:
                    print(f"[{update_step}] eval failed: {e}")

            update_step += 1
            micro_accum = 0
            accum_grads = None
            if update_step >= max_steps:
                break

    if rank == 0:
        final_path = os.path.join(save_dir, "ckpt_final.safetensors")
        ok = save_safetensors_model(final_path, model)
        if not ok:
            print(f"[final] ⚠️ final checkpoint NOT written: {final_path}", flush=True)

# ---------------------------
# Generators (KV-cache aware)
# ---------------------------

from collections import Counter

def _apply_repetition_penalty_np(
    logits_np: np.ndarray,
    generated_ids: List[int],
    repetition_penalty: float,
    penalty_window: Optional[int],
) -> np.ndarray:
    """
    Apply a count-based repetition penalty directly to logits (numpy array).

    - logits_np: shape (vocab_size,)
    - generated_ids: list of token ids generated so far (including prompt)
    - repetition_penalty > 1.0: stronger push away from repeats
    - penalty_window: consider only the last N tokens (None = all)
    """
    if not generated_ids or repetition_penalty is None or repetition_penalty <= 1.0:
        return logits_np

    if penalty_window is not None:
        window_ids = generated_ids[-penalty_window:]
    else:
        window_ids = generated_ids

    counts = Counter(window_ids)
    for tid, c in counts.items():
        if 0 <= tid < logits_np.shape[0]:
            # divide by penalty ** count to strongly discourage heavy repeats
            logits_np[tid] /= (repetition_penalty ** c)

    return logits_np

def _greedy_with_repetition_penalty(
    logits: mx.array,
    generated_ids: List[int],
    repetition_penalty: float = 1.8,
    penalty_window: int = 32,
) -> int:
    """
    Deterministic 'greedy' using a count-based repetition penalty in logit space.
    """
    logits_np = np.array(logits[0], dtype=np.float32)
    logits_np = _apply_repetition_penalty_np(
        logits_np,
        generated_ids,
        repetition_penalty=repetition_penalty,
        penalty_window=penalty_window,
    )
    best_id = int(np.argmax(logits_np))
    return best_id

def _sample_next_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
    penalty_window: Optional[int] = None,
) -> int:
    """
    Sample a next token with optional top-k/top-p and count-based repetition penalty
    applied in logit space.
    """
    # Convert logits to numpy
    logits_np = np.array(logits[0], dtype=np.float32)

    # Apply repetition penalty before softmax
    logits_np = _apply_repetition_penalty_np(
        logits_np,
        generated_ids or [],
        repetition_penalty=repetition_penalty,
        penalty_window=penalty_window,
    )

    # Temperature scaling + softmax
    logits_np = logits_np / max(1e-5, float(temperature))
    exp = np.exp(logits_np - np.max(logits_np))
    probs_np = exp / (exp.sum() + 1e-9)

    # ---- top-k filtering ----
    if top_k is not None and top_k > 0 and top_k < probs_np.shape[0]:
        idxs = np.argsort(probs_np)[::-1]
        keep = set(idxs[:top_k])
        mask = np.array([1.0 if i in keep else 0.0 for i in range(len(probs_np))], dtype=np.float32)
        probs_np *= mask
        total = probs_np.sum()
        if total > 0:
            probs_np /= total

    # ---- top-p (nucleus) filtering ----
    if top_p is not None and 0.0 < top_p < 1.0:
        idxs = np.argsort(probs_np)[::-1]
        cumulative = 0.0
        keep = []
        for i in idxs:
            cumulative += probs_np[i]
            keep.append(i)
            if cumulative >= top_p:
                break
        keep_set = set(keep)
        mask = np.array([1.0 if i in keep_set else 0.0 for i in range(len(probs_np))], dtype=np.float32)
        probs_np *= mask
        total = probs_np.sum()
        if total > 0:
            probs_np /= total

    # ---- sample from adjusted distribution ----
    import random
    r = random.random()
    c = 0.0
    for i, p in enumerate(probs_np):
        c += p
        if r <= c:
            return int(i)

    # Fallback: argmax
    return int(np.argmax(probs_np))

def generate_greedy_nocache(model: TinyGPLM, prompt: str, max_new_tokens: int = 128) -> str:
    ids = TOK.encode(prompt)[: model.cfg.max_seq]
    out_ids = ids[:]
    for _ in range(max_new_tokens):
        x_ids = mx.array([out_ids[:model.cfg.max_seq]], dtype=mx.int32)  # shape (1, T)
        logits = model.logits(x_ids)  # (1, T, vocab)
        last_logits = logits[:, -1, :]  # (1, vocab)

        next_id = _greedy_with_repetition_penalty(
            last_logits,
            generated_ids=out_ids,
            repetition_penalty=1.8,   # strong push away from repeats
            penalty_window=32,        # only care about recent tokens
        )

        out_ids.append(next_id)
        if next_id == TOK.eos_id:
            break
    return TOK.decode(out_ids)

def generate_topk(model: TinyGPLM, prompt: str, max_new_tokens: int = 128,
                  temperature: float = 0.8, top_k: int = 40) -> str:
    ids = TOK.encode(prompt)[: model.cfg.max_seq]
    caches = None
    for tok in ids[:-1]:
        tok_id = mx.array([[tok]], dtype=mx.int32)
        _, caches = model.step(tok_id, caches)
    cur = mx.array([[ids[-1]]], dtype=mx.int32)
    out_ids = ids[:]
    for _ in range(max_new_tokens):
        logits, caches = model.step(cur, caches)
        next_id = _sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=None,
            generated_ids=out_ids,
            repetition_penalty=1.5,   # milder than greedy
            penalty_window=64,
        )
        out_ids.append(int(next_id))
        if next_id == TOK.eos_id:
            break
        cur = mx.array([[next_id]], dtype=mx.int32)
    return TOK.decode(out_ids)

def generate_topp(model: TinyGPLM, prompt: str, max_new_tokens: int = 128,
                  temperature: float = 0.8, top_p: float = 0.9) -> str:
    ids = TOK.encode(prompt)[: model.cfg.max_seq]
    caches = None
    for tok in ids[:-1]:
        tok_id = mx.array([[tok]], dtype=mx.int32)
        _, caches = model.step(tok_id, caches)
    cur = mx.array([[ids[-1]]], dtype=mx.int32)
    out_ids = ids[:]
    for _ in range(max_new_tokens):
        logits, caches = model.step(cur, caches)
        next_id = _sample_next_token(
            logits,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            generated_ids=out_ids,
            repetition_penalty=1.5,
            penalty_window=64,
        )
        out_ids.append(int(next_id))
        if next_id == TOK.eos_id:
            break
        cur = mx.array([[next_id]], dtype=mx.int32)
    return TOK.decode(out_ids)

def run_eval_samples(model: TinyGPLM, step: int, prompt: str, max_new_tokens: int):
    """
    Run multiple evaluation prompts with several decoding strategies
    so you can see different behaviors as training progresses.
    """
    prompts = [
        ("default", prompt),
        ("story", "Once upon a time, "),
        ("facts", "The capital of France is "),
        ("code", "Write a Python function that "),
        ("reasoning", "If Alice has 3 apples and Bob gives her 2 more, "),
    ]

    for label, p in prompts:
        print(f"[{step}] === EVAL SAMPLES ({label}) ===")
        print(f"Prompt: {repr(p)}")

        # 1) Greedy with repetition penalty
        greedy_nocache = generate_greedy_nocache(model, p, max_new_tokens)
        print(f"[{step}] [{label}] greedy (no cache):\n{greedy_nocache}\n---")

        # 2) Top-k, fairly conservative
        topk_20_t07 = generate_topk(
            model,
            p,
            max_new_tokens,
            temperature=0.7,
            top_k=20,
        )
        print(f"[{step}] [{label}] top-k (k=20, T=0.7):\n{topk_20_t07}\n---")

        # 3) Top-k, more exploratory
        topk_80_t10 = generate_topk(
            model,
            p,
            max_new_tokens,
            temperature=1.0,
            top_k=80,
        )
        print(f"[{step}] [{label}] top-k (k=80, T=1.0):\n{topk_80_t10}\n---")

        # 4) Top-p (nucleus), conservative
        topp_09_t07 = generate_topp(
            model,
            p,
            max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        print(f"[{step}] [{label}] top-p (p=0.9, T=0.7):\n{topp_09_t07}\n---")

        # 5) Top-p, more exploratory
        topp_095_t10 = generate_topp(
            model,
            p,
            max_new_tokens,
            temperature=1.0,
            top_p=0.95,
        )
        print(f"[{step}] [{label}] top-p (p=0.95, T=1.0):\n{topp_095_t10}\n---")

# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser("Distributed MLX LM training (SPM BPE tokenizer only)")
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--trust-remote-code", action="store_true", default=True)

    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)

    p.add_argument("--backend", type=str, default="ring")
    p.add_argument("--expected-world", type=int, default=None)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--per-rank-logs", action="store_true")
    p.add_argument("--save-dir", type=str, default="model/checkpoints_spm")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-prompt", type=str, default="Hello, my name is ")
    p.add_argument("--eval-tokens", type=int, default=200)
    p.add_argument("--resume", type=str, default=None, help="Path to a .safetensors checkpoint to resume from")

    p.add_argument("--spm-model", type=str, required=True,
                   help="Path to tokenizer/fineweb_spm/spm.model")

    p.add_argument(
        "--config",
        type=str,
        default="configs/config_mlx_slm_beta_v2_100m.json",
        help="Path to TinyGPLM JSON model config.",
    )

    p.add_argument(
        "--lr-decay-start-step",
        type=int,
        default=None,
        help="Step at which to start cosine LR decay (default: 0.5 * max_steps).",
    )

    args = p.parse_args()

    set_tokenizer(args.spm_model)

    ds_cfg = None if args.dataset_config in (None, "", "None", "none") else args.dataset_config
    train_hf_distributed(
        dataset_name=args.dataset,
        dataset_config=ds_cfg,
        split=args.split,
        text_field=args.text_field,
        trust_remote_code=args.trust_remote_code,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        wd=args.wd,
        backend=args.backend,
        expected_world=args.expected_world,
        log_every=args.log_every,
        per_rank_logs=args.per_rank_logs,
        save_dir=args.save_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_prompt=args.eval_prompt,
        eval_tokens=args.eval_tokens,
        resume_ckpt=args.resume,
        config_path=args.config,
        lr_decay_start_step=args.lr_decay_start_step,
    )

if __name__ == "__main__":
    main()