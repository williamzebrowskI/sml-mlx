#!/usr/bin/env python3
"""Distributed MLX SFT trainer inspired by nanochat's chat_sft.py.

This is a 4-Mac Thunderbolt-ring, MLX-only supervised fine-tuning path.
It keeps the current repo's data-parallel strategy:
  - full model replica on every rank
  - each rank computes local forward/backward on different samples
  - gradients are all-reduced across ranks
  - rank 0 can be the only optimizer/checkpoint writer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.model import TransformerConfig, TransformerLM, count_parameters
from train.train import (
    _all_sum,
    _tree_add,
    _tree_scale,
    _allreduce_tree,
    _broadcast_model_from_rank0,
    _build_lr_schedule,
    _cast_model_floats,
    _clip_grads,
    _infer_resume_step,
    _load_checkpoint,
    _load_config_defaults,
    _resolve_dtype,
    _save_checkpoint,
)


def _norm_role(role: str) -> Optional[str]:
    r = (role or "").strip().lower()
    if r in {"user", "human", "prompt"}:
        return "user"
    if r in {"assistant", "gpt", "bot", "model"}:
        return "assistant"
    if r == "system":
        return "system"
    return None


def _extract_messages(ex: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
    raw = None
    if isinstance(ex.get("messages"), list):
        raw = ex["messages"]
    elif isinstance(ex.get("conversations"), list):
        raw = ex["conversations"]
    if raw is None:
        return None

    msgs: List[Tuple[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or msg.get("speaker")
        content = msg.get("content") or msg.get("value") or msg.get("text") or msg.get("message")
        if not role or not content:
            continue
        norm = _norm_role(str(role))
        if norm is None:
            continue
        text = str(content).strip()
        if text:
            msgs.append((norm, text))
    return msgs or None


def _extract_pair_from_fields(ex: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    inst = ex.get("instruction")
    out = ex.get("output")
    if isinstance(inst, str) and isinstance(out, str) and inst.strip() and out.strip():
        inp = ex.get("input")
        user = inst.strip()
        if isinstance(inp, str) and inp.strip():
            user = f"{user}\n{inp.strip()}"
        return user, out.strip()

    for user_key, assistant_key in [
        ("prompt", "completion"),
        ("prompt", "response"),
        ("question", "answer"),
        ("query", "response"),
        ("input", "output"),
    ]:
        user = ex.get(user_key)
        assistant = ex.get(assistant_key)
        if isinstance(user, str) and isinstance(assistant, str):
            user = user.strip()
            assistant = assistant.strip()
            if user and assistant:
                return user, assistant
    return None


def _choose_user_assistant_pair(
    messages: List[Tuple[str, str]],
    *,
    strategy: str,
) -> Optional[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    last_user: Optional[str] = None
    for role, content in messages:
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user is not None:
            pairs.append((last_user, content))
            last_user = None
    if not pairs:
        return None
    if strategy == "first":
        return pairs[0]
    if strategy == "last":
        return pairs[-1]
    return random.choice(pairs)


def _extract_pair(ex: Dict[str, Any], *, pair_strategy: str) -> Optional[Tuple[str, str]]:
    messages = _extract_messages(ex)
    if messages:
        pair = _choose_user_assistant_pair(messages, strategy=pair_strategy)
        if pair is not None:
            return pair
    return _extract_pair_from_fields(ex)


@dataclass(frozen=True)
class RecipeSource:
    name: str
    split: str
    config: Optional[str] = None
    weight: int = 1


def _recipe_sources(recipe: str) -> List[RecipeSource]:
    r = recipe.strip().lower()
    if r == "tulu_v2":
        return [RecipeSource(name="allenai/tulu-v2-sft-mixture", split="train")]
    if r == "ultrachat":
        return [RecipeSource(name="HuggingFaceH4/ultrachat_200k", split="train_sft")]
    if r in {"mix", "nanochat_like"}:
        return [
            RecipeSource(name="allenai/tulu-v2-sft-mixture", split="train", weight=4),
            RecipeSource(name="HuggingFaceH4/ultrachat_200k", split="train_sft", weight=1),
        ]
    raise ValueError(f"Unknown recipe: {recipe}")


class _PairCursor:
    def __init__(
        self,
        src: RecipeSource,
        *,
        world_size: int,
        rank: int,
        seed: int,
        pair_strategy: str,
        shuffle_buffer: int,
        trust_remote_code: bool,
    ):
        self.src = src
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.pair_strategy = pair_strategy
        self.shuffle_buffer = shuffle_buffer
        self.trust_remote_code = trust_remote_code
        self.epoch = 0
        self.example_index = 0
        self.ds = None
        self.it = None
        self._reset_dataset()

    def _build_dataset(self):
        ds = load_dataset(
            self.src.name,
            self.src.config,
            split=self.src.split,
            streaming=True,
            trust_remote_code=self.trust_remote_code,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + self.epoch)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self.epoch)
        return ds

    def _reset_dataset(self):
        self.ds = self._build_dataset()
        self.it = iter(self.ds)
        self.example_index = 0

    def next_pair(self) -> Tuple[str, str]:
        while True:
            try:
                ex = next(self.it)
            except StopIteration:
                self.epoch += 1
                self._reset_dataset()
                continue
            idx = self.example_index
            self.example_index += 1
            if self.world_size > 1 and (idx % self.world_size) != self.rank:
                continue
            if not isinstance(ex, dict):
                continue
            pair = _extract_pair(ex, pair_strategy=self.pair_strategy)
            if pair is not None:
                return pair


class ConversationSFTBatcher:
    def __init__(
        self,
        *,
        recipe: str,
        spm_model: str,
        world_size: int,
        rank: int,
        seed: int,
        pair_strategy: str,
        shuffle_buffer: int,
        trust_remote_code: bool,
        ignore_index: int,
        max_user_chars: int,
        max_assistant_chars: int,
        min_assistant_tokens: int,
    ):
        self.sources = [
            _PairCursor(
                src=src,
                world_size=world_size,
                rank=rank,
                seed=seed + i * 1_000_003,
                pair_strategy=pair_strategy,
                shuffle_buffer=shuffle_buffer,
                trust_remote_code=trust_remote_code,
            )
            for i, src in enumerate(_recipe_sources(recipe))
        ]
        self.schedule: List[int] = []
        for i, src in enumerate(_recipe_sources(recipe)):
            self.schedule.extend([i] * int(src.weight))
        self.schedule_pos = 0
        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        if self.pad_id < 0:
            self.pad_id = self.eos_id if self.eos_id >= 0 else 0
        self.ignore_index = ignore_index
        self.max_user_chars = max_user_chars
        self.max_assistant_chars = max_assistant_chars
        self.min_assistant_tokens = min_assistant_tokens

    def _next_source(self) -> _PairCursor:
        idx = self.schedule[self.schedule_pos % len(self.schedule)]
        self.schedule_pos += 1
        return self.sources[idx]

    def _encode_plain(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def _build_xy(self, user_text: str, assistant_text: str, *, seq_len: int) -> Optional[Tuple[List[int], List[int]]]:
        user = user_text.strip()
        assistant = assistant_text.strip()
        if not user or not assistant:
            return None
        if self.max_user_chars > 0:
            user = user[: self.max_user_chars]
        if self.max_assistant_chars > 0:
            assistant = assistant[: self.max_assistant_chars]

        min_assistant_tokens = max(1, int(self.min_assistant_tokens))
        prefix_budget = max(1, int(seq_len) - min_assistant_tokens)

        prefix = f"User: {user}\nAssistant: "
        prefix_ids = self._encode_plain(prefix)
        if len(prefix_ids) > prefix_budget:
            scale = prefix_budget / max(1, len(prefix_ids))
            new_len = max(1, int(len(user) * scale * 0.95))
            user = user[:new_len]
            prefix = f"User: {user}\nAssistant: "
            prefix_ids = self._encode_plain(prefix)
        if len(prefix_ids) > prefix_budget:
            return None

        full_ids = self._encode_plain(prefix + assistant)
        ids = [self.bos_id] + full_ids + [self.eos_id]
        boundary = 1 + len(prefix_ids)
        ids = ids[: seq_len + 1]
        if boundary >= len(ids):
            return None

        x_ids = ids[:-1]
        y_ids = ids[1:]
        mask_upto = min(max(0, boundary - 1), len(y_ids))
        if mask_upto:
            y_ids[:mask_upto] = [self.ignore_index] * mask_upto
        if all(tok == self.ignore_index for tok in y_ids):
            return None
        if len(x_ids) < seq_len:
            pad = seq_len - len(x_ids)
            x_ids.extend([self.pad_id] * pad)
            y_ids.extend([self.ignore_index] * pad)
        return x_ids, y_ids

    def sample_batch(self, *, batch_size: int, seq_len: int) -> Tuple[mx.array, mx.array]:
        x_rows: List[List[int]] = []
        y_rows: List[List[int]] = []
        while len(x_rows) < batch_size:
            user, assistant = self._next_source().next_pair()
            xy = self._build_xy(user, assistant, seq_len=seq_len)
            if xy is None:
                continue
            x_ids, y_ids = xy
            x_rows.append(x_ids)
            y_rows.append(y_ids)
        return mx.array(x_rows, dtype=mx.int32), mx.array(y_rows, dtype=mx.int32)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre], description="Distributed MLX nanochat-style SFT")
    parser.add_argument("--base-ckpt", type=str, default="")
    parser.add_argument("--spm-model", type=str, default="/Users/williamzebrowski/sml-mlx/tokenizer/fineweb_spm/spm.model")
    parser.add_argument("--recipe", type=str, default="nanochat_like", choices=["nanochat_like", "mix", "tulu_v2", "ultrachat"])
    parser.add_argument("--pair-strategy", type=str, default="last", choices=["first", "last", "random"])
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attention-impl", type=str, default="fast", choices=["fast", "vanilla"])
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.set_defaults(optimizer_rank0_only=True)
    parser.add_argument("--optimizer-rank0-only", action="store_true", dest="optimizer_rank0_only")
    parser.add_argument("--no-optimizer-rank0-only", action="store_false", dest="optimizer_rank0_only")
    parser.add_argument("--backend", type=str, default="ring")
    parser.add_argument("--collective-stream", type=str, default="cpu", choices=["cpu", "default"])
    parser.add_argument("--expected-world", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_sft/checkpoints/sft_v1")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument("--max-user-chars", type=int, default=8000)
    parser.add_argument("--max-assistant-chars", type=int, default=12000)
    parser.add_argument("--min-assistant-tokens", type=int, default=64)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--trace-first-step", action="store_true")

    if pre_args.config:
        cfg_path = Path(pre_args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        parser.set_defaults(**_load_config_defaults(str(cfg_path)))
        parser.set_defaults(config=str(cfg_path))

    args = parser.parse_args(remaining)

    try:
        group = mx.distributed.init(backend=args.backend, strict=True)
    except TypeError:
        group = mx.distributed.init(backend=args.backend)

    rank = int(group.rank() if callable(getattr(group, "rank", None)) else group.rank)
    world = int(group.size() if callable(getattr(group, "size", None)) else group.size)
    print(f"[rank {rank}] host={socket.gethostname()} world={world}", flush=True)
    if args.expected_world is not None and world != args.expected_world:
        raise RuntimeError(f"Expected world={args.expected_world}, got {world}")
    if args.checkpoint_rank < 0 or args.checkpoint_rank >= world:
        raise RuntimeError(f"checkpoint_rank must be in [0, {world - 1}], got {args.checkpoint_rank}")

    cfg = TransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        attention_impl=args.attention_impl,
    )
    model = TransformerLM(cfg)
    model_dtype = _resolve_dtype(args.dtype)
    _cast_model_floats(model, model_dtype)
    mx.eval(model.parameters())

    if args.base_ckpt:
        loaded = False
        if rank == 0:
            loaded = _load_checkpoint(args.base_ckpt, model)
            print(f"[ckpt] base_ckpt={args.base_ckpt} loaded={loaded}", flush=True)
            if not loaded:
                raise FileNotFoundError(f"Could not load base checkpoint: {args.base_ckpt}")

    start_step = 0
    if args.resume:
        loaded = False
        if rank == 0:
            loaded = _load_checkpoint(args.resume, model)
            if loaded:
                start_step = _infer_resume_step(args.resume)
            print(f"[ckpt] resume={args.resume} loaded={loaded} start_step={start_step}", flush=True)
        if world > 1:
            start_step_arr = _all_sum(
                mx.array(float(start_step if rank == 0 else 0.0), dtype=mx.float32),
                stream_mode=args.collective_stream,
            )
            mx.eval(start_step_arr)
            start_step = int(start_step_arr.item())

    _broadcast_model_from_rank0(model, rank, world, stream_mode=args.collective_stream)

    num_params = count_parameters(model)
    dtype_bytes = 2 if model_dtype in (mx.float16, mx.bfloat16) else 4
    param_gib = (num_params * dtype_bytes) / (1024**3)
    if rank == 0:
        print(
            f"[model] params={num_params/1e6:.2f}M dtype={args.dtype} approx_param_mem={param_gib:.2f} GiB/rank",
            flush=True,
        )
        print(
            f"[sft] recipe={args.recipe} world={world} seq={args.max_seq_len} batch={args.batch_size} "
            f"accum={args.grad_accum} opt_mode={'rank0_only' if args.optimizer_rank0_only else 'all_ranks'} "
            f"checkpoint_rank={args.checkpoint_rank}",
            flush=True,
        )

    batcher = ConversationSFTBatcher(
        recipe=args.recipe,
        spm_model=args.spm_model,
        world_size=world,
        rank=rank,
        seed=args.seed,
        pair_strategy=args.pair_strategy,
        shuffle_buffer=args.shuffle_buffer,
        trust_remote_code=args.trust_remote_code,
        ignore_index=args.ignore_index,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
        min_assistant_tokens=args.min_assistant_tokens,
    )

    optimizer = optim.AdamW(args.lr, weight_decay=args.weight_decay)
    lr_for_step = _build_lr_schedule(
        base_lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    def loss_fn(x, y):
        return model(x, targets=y, ignore_index=args.ignore_index)["loss"]

    step_and_grad = nn.value_and_grad(model, loss_fn)
    ema_loss = None
    loop_start = time.perf_counter()

    for step in range(start_step, args.max_steps):
        t0 = time.perf_counter()
        total_loss_local = 0.0
        grads_acc = None

        for micro in range(args.grad_accum):
            if args.trace_first_step and step == start_step:
                print(f"[rank {rank}] trace step={step+1} micro={micro} stage=sample_batch", flush=True)
            x, y = batcher.sample_batch(batch_size=args.batch_size, seq_len=args.max_seq_len)
            ready = _all_sum(mx.array(1.0, dtype=mx.float32), stream_mode=args.collective_stream)
            mx.eval(ready)
            if args.trace_first_step and step == start_step:
                print(f"[rank {rank}] trace step={step+1} micro={micro} stage=fwd_bwd", flush=True)
            loss, grads = step_and_grad(x, y)
            mx.eval(loss)
            total_loss_local += float(loss.item())
            grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)

        grads_acc = _tree_scale(grads_acc, 1.0 / float(args.grad_accum))
        if world > 1:
            if args.trace_first_step and step == start_step:
                print(f"[rank {rank}] trace step={step+1} stage=allreduce_grads", flush=True)
            grads_acc = _allreduce_tree(grads_acc, world, stream_mode=args.collective_stream)

        grads_acc, grad_norm = _clip_grads(grads_acc, args.grad_clip)
        lr_t = lr_for_step(step)
        if args.optimizer_rank0_only and world > 1:
            if rank == 0:
                optimizer.learning_rate = lr_t
                optimizer.update(model, grads_acc)
                mx.eval(model.parameters(), optimizer.state)
            _broadcast_model_from_rank0(model, rank, world, stream_mode=args.collective_stream)
        else:
            optimizer.learning_rate = lr_t
            optimizer.update(model, grads_acc)
            mx.eval(model.parameters(), optimizer.state)

        step_loss = mx.array(total_loss_local / float(args.grad_accum), dtype=mx.float32)
        if world > 1:
            step_loss = _all_sum(step_loss, stream_mode=args.collective_stream) / world
        mx.eval(step_loss)
        step_loss_value = float(step_loss.item())
        if not math.isfinite(step_loss_value) or not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite at step {step+1}: loss={step_loss_value}, grad_norm={grad_norm}")

        ema_loss = step_loss_value if ema_loss is None else (0.98 * ema_loss + 0.02 * step_loss_value)
        dt = time.perf_counter() - t0
        toks_per_step = args.batch_size * args.grad_accum * args.max_seq_len * world
        toks_per_sec = toks_per_step / max(dt, 1e-9)

        if rank == 0 and ((step + 1) % args.log_every == 0 or step == 0):
            print(
                f"[step {step+1:6d}] loss={step_loss_value:.4f} ema={ema_loss:.4f} "
                f"lr={lr_t:.3e} grad_norm={grad_norm:.3f} tok/s={toks_per_sec:,.0f}",
                flush=True,
            )

        if args.save_every > 0 and ((step + 1) % args.save_every == 0) and rank == args.checkpoint_rank:
            ckpt_path = os.path.join(args.save_dir, f"step_{step+1:07d}.safetensors")
            _save_checkpoint(
                ckpt_path,
                model,
                {
                    "step": step + 1,
                    "args": vars(args),
                    "config": asdict(cfg),
                    "world": world,
                    "backend": args.backend,
                    "timestamp": time.time(),
                },
            )
            print(f"[ckpt] saved {ckpt_path}", flush=True)

    if rank == args.checkpoint_rank:
        final_path = os.path.join(args.save_dir, "final.safetensors")
        _save_checkpoint(
            final_path,
            model,
            {
                "step": args.max_steps,
                "args": vars(args),
                "config": asdict(cfg),
                "world": world,
                "backend": args.backend,
                "duration_sec": time.perf_counter() - loop_start,
                "timestamp": time.time(),
            },
        )
        print(f"[done] saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
