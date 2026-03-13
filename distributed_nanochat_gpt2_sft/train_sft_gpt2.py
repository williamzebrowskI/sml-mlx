#!/usr/bin/env python3
"""Distributed MLX GPT-2 SFT path modeled after nanochat's chat_sft.py."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distributed_nanogpt_streaming.model import GPT2Config, GPT2LM, count_parameters
from distributed_nanochat_gpt2_sft.tasks import build_cursor, parse_source_specs
from train.train import (
    _all_sum,
    _allreduce_tree,
    _build_lr_schedule,
    _cast_model_floats,
    _clip_grads,
    _flatten_for_safetensors,
    _load_config_defaults,
    _resolve_dtype,
    _tree_add,
    _tree_scale,
)


USER_START = "<|user_start|>\n"
USER_END = "\n<|user_end|>\n"
ASSISTANT_START = "<|assistant_start|>\n"
ASSISTANT_END = "\n<|assistant_end|>\n"
PYTHON_START = "<|python_start|>\n"
PYTHON_END = "\n<|python_end|>\n"
OUTPUT_START = "<|output_start|>\n"
OUTPUT_END = "\n<|output_end|>\n"


def _tree_assign(template: Any, flat: Dict[str, mx.array], prefix: str = "") -> Any:
    if isinstance(template, dict):
        return {k: _tree_assign(v, flat, f"{prefix}.{k}" if prefix else k) for k, v in template.items()}
    if isinstance(template, list):
        return [_tree_assign(v, flat, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(template)]
    if isinstance(template, tuple):
        return tuple(_tree_assign(v, flat, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(template))
    if isinstance(template, mx.array):
        key = prefix if prefix else "param"
        if key not in flat:
            return template
        value = flat[key].astype(template.dtype)
        if value.shape != template.shape:
            raise ValueError(f"Shape mismatch for {key}: file={value.shape} model={template.shape}")
        return value
    return template


def _broadcast_tree_from_rank0(tree: Any, rank: int, world: int, stream_mode: str) -> Any:
    if world == 1:
        return tree

    def visit(value):
        if not isinstance(value, mx.array):
            return value
        src = value if rank == 0 else mx.zeros_like(value)
        return _all_sum(src, stream_mode=stream_mode)

    if isinstance(tree, dict):
        return {k: _broadcast_tree_from_rank0(v, rank, world, stream_mode) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_broadcast_tree_from_rank0(v, rank, world, stream_mode) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_broadcast_tree_from_rank0(v, rank, world, stream_mode) for v in tree)
    return visit(tree)


def _sync_ranks(world: int, stream_mode: str) -> None:
    if world <= 1:
        return
    marker = _all_sum(mx.array(1.0, dtype=mx.float32), stream_mode=stream_mode)
    mx.eval(marker)


def _save_training_checkpoint(path: str, model: GPT2LM, optimizer: optim.AdamW, metadata: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensors: Dict[str, mx.array] = {}
    _flatten_for_safetensors(model.parameters(), prefix="model", out=tensors)
    _flatten_for_safetensors(optimizer.state, prefix="optimizer", out=tensors)
    mx.save_safetensors(path, tensors)
    with open(path + ".json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _load_checkpoint_into(
    path: str,
    *,
    model: GPT2LM,
    optimizer: Optional[optim.AdamW],
    load_optimizer: bool,
) -> Dict[str, Any]:
    flat = mx.load(path)
    model.update(_tree_assign(model.parameters(), flat, prefix="model"))
    if optimizer is not None and load_optimizer and any(k.startswith("optimizer.") for k in flat):
        optimizer.state = _tree_assign(optimizer.state, flat, prefix="optimizer")
        mx.eval(optimizer.state)
    mx.eval(model.parameters())
    meta_path = path + ".json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def _inherit_model_args(args, base_meta: Dict[str, Any]) -> None:
    model_args = base_meta.get("model_args", {})
    mappings = {
        "block_size": "block_size",
        "vocab_size": "vocab_size",
        "n_layer": "n_layer",
        "n_head": "n_head",
        "n_embd": "n_embd",
        "dropout": "dropout",
        "bias": "bias",
        "attention_impl": "attention_impl",
    }
    for arg_name, meta_name in mappings.items():
        if getattr(args, arg_name) is None and model_args.get(meta_name) is not None:
            setattr(args, arg_name, model_args[meta_name])


@dataclass
class CursorState:
    cursor: Any
    weight: int
    name: str


class GPT2ConversationBatcher:
    def __init__(
        self,
        *,
        sources,
        tokenizer_name: str,
        rank: int,
        world: int,
        seed: int,
        trust_remote_code: bool,
        shuffle_buffer: int,
        ignore_index: int,
        buffer_size: int,
    ):
        from transformers import AutoTokenizer

        self.rank = rank
        self.world = world
        self.seed = seed
        self.ignore_index = ignore_index
        self.buffer_size = max(8, buffer_size)
        self.rng = np.random.default_rng(seed + rank * 1009)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer.model_max_length = 1_000_000_000
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bos_id = int(self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id)
        self.pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.bos_id)
        self.sources: List[CursorState] = []
        for i, spec in enumerate(sources):
            cursor = build_cursor(
                spec,
                rank=rank,
                world=world,
                seed=seed + i * 1_000_003,
                trust_remote_code=trust_remote_code,
            )
            self.sources.append(CursorState(cursor=cursor, weight=int(spec.weight), name=spec.kind))
        if not self.sources:
            raise ValueError("At least one source is required.")
        self.buffer: List[Tuple[List[int], List[int]]] = []

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _render_conversation(self, conversation: Dict[str, Any], *, max_tokens: int) -> Tuple[List[int], List[int]]:
        ids: List[int] = []
        mask: List[int] = []

        def add(token_ids: List[int] | int, mask_value: int) -> None:
            values = [token_ids] if isinstance(token_ids, int) else token_ids
            ids.extend(values)
            mask.extend([mask_value] * len(values))

        messages = list(conversation["messages"])
        if messages and messages[0].get("role") == "system" and len(messages) > 1:
            merged = dict(messages[1])
            merged["content"] = str(messages[0]["content"]).strip() + "\n\n" + str(messages[1]["content"]).strip()
            messages = [merged] + messages[2:]

        add(self.bos_id, 0)
        for i, message in enumerate(messages):
            role = message["role"]
            expected = "user" if i % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError(f"Conversation role mismatch at {i}: got {role}, expected {expected}")
            content = message["content"]
            if role == "user":
                add(self._encode(USER_START), 0)
                add(self._encode(str(content)), 0)
                add(self._encode(USER_END), 0)
                continue

            add(self._encode(ASSISTANT_START), 0)
            if isinstance(content, str):
                add(self._encode(content), 1)
            elif isinstance(content, list):
                for part in content:
                    text = str(part["text"])
                    part_type = part["type"]
                    if part_type == "text":
                        add(self._encode(text), 1)
                    elif part_type == "python":
                        add(self._encode(PYTHON_START), 1)
                        add(self._encode(text), 1)
                        add(self._encode(PYTHON_END), 1)
                    elif part_type == "python_output":
                        add(self._encode(OUTPUT_START), 0)
                        add(self._encode(text), 0)
                        add(self._encode(OUTPUT_END), 0)
                    else:
                        raise ValueError(f"Unsupported assistant part type: {part_type}")
            else:
                raise ValueError(f"Unsupported assistant content type: {type(content)}")
            add(self._encode(ASSISTANT_END), 1)

        return ids[:max_tokens], mask[:max_tokens]

    def _pick_source(self) -> CursorState:
        total = sum(max(1, src.weight) for src in self.sources)
        threshold = int(self.rng.integers(total))
        running = 0
        for src in self.sources:
            running += max(1, src.weight)
            if threshold < running:
                return src
        return self.sources[-1]

    def _next_conversation(self, *, max_tokens: int) -> Tuple[List[int], List[int]]:
        while True:
            src = self._pick_source()
            conversation = src.cursor.next_conversation()
            ids, mask = self._render_conversation(conversation, max_tokens=max_tokens)
            if ids and len(ids) == len(mask) and any(mask):
                return ids, mask

    def _refill(self, *, seq_len: int) -> None:
        row_capacity = seq_len + 1
        while len(self.buffer) < self.buffer_size:
            self.buffer.append(self._next_conversation(max_tokens=row_capacity))

    def sample_batch(self, *, batch_size: int, seq_len: int) -> Tuple[mx.array, mx.array]:
        row_capacity = seq_len + 1
        rows: List[List[int]] = []
        mask_rows: List[List[int]] = []
        row_lengths: List[int] = []

        for _ in range(batch_size):
            row: List[int] = []
            row_mask: List[int] = []
            content_len = 0
            while len(row) < row_capacity:
                self._refill(seq_len=seq_len)
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for idx, (ids, _) in enumerate(self.buffer):
                    conv_len = len(ids)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = idx
                        best_len = conv_len
                if best_idx >= 0:
                    ids, masks = self.buffer.pop(best_idx)
                    row.extend(ids)
                    row_mask.extend(masks)
                    content_len = len(row)
                    continue
                pad = remaining
                row.extend([self.pad_id] * pad)
                row_mask.extend([0] * pad)
                break
            row = row[:row_capacity]
            row_mask = row_mask[:row_capacity]
            row_lengths.append(content_len if content_len > 0 else row_capacity)
            rows.append(row)
            mask_rows.append(row_mask)

        x = np.asarray([r[:-1] for r in rows], dtype=np.int32)
        y = np.asarray([r[1:] for r in rows], dtype=np.int32)
        target_mask = np.asarray([m[1:] for m in mask_rows], dtype=np.int32)
        y[target_mask == 0] = self.ignore_index
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                y[i, max(0, content_len - 1) :] = self.ignore_index
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def _estimate_loss(
    *,
    model: GPT2LM,
    batcher: GPT2ConversationBatcher,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    ignore_index: int,
    rank: int,
    world: int,
    collective_stream: str,
) -> float:
    losses_out = []
    was_training = model.training
    model.eval()
    for _ in range(eval_iters):
        x, y = batcher.sample_batch(batch_size=batch_size, seq_len=block_size)
        logits = model.logits(x)
        per_token = losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
        mask = (y != ignore_index).astype(per_token.dtype)
        loss = (per_token * mask).sum() / (mask.sum() + 1e-6)
        if world > 1:
            loss = _all_sum(loss, stream_mode=collective_stream) / world
        mx.eval(loss)
        losses_out.append(float(loss.item()))
    if was_training:
        model.train()
    return float(sum(losses_out) / max(1, len(losses_out)))


def _generate_sample_text(
    *,
    model: GPT2LM,
    tokenizer_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = 1_000_000_000
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [int(tokenizer.eos_token_id)]
    out_ids = list(int(x) for x in prompt_ids)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            ctx = out_ids[-model.config.block_size :]
            x = mx.array(np.asarray([ctx], dtype=np.int32), dtype=mx.int32)
            logits = model.logits(x)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            probs = np.asarray(logits[0])
            if temperature <= 0.0 or top_k == 1:
                next_id = int(np.argmax(probs))
            else:
                scaled = probs / max(temperature, 1e-5)
                if top_k > 0 and top_k < scaled.shape[0]:
                    keep = np.argpartition(scaled, -top_k)[-top_k:]
                    masked = np.full_like(scaled, -np.inf)
                    masked[keep] = scaled[keep]
                    scaled = masked
                scaled = scaled - np.max(scaled)
                p = np.exp(scaled)
                p = p / np.clip(p.sum(), 1e-12, None)
                next_id = int(np.random.choice(p.shape[0], p=p))
            out_ids.append(next_id)
    finally:
        if was_training:
            model.train()
    return tokenizer.decode(out_ids, clean_up_tokenization_spaces=False)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre], description="Distributed MLX nanochat-style GPT-2 SFT")
    parser.add_argument("--base-ckpt", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--load-optimizer", action="store_true", default=True)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--train-sources", default="")
    parser.add_argument("--val-sources", default="")
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", default=None)
    parser.add_argument("--attention-impl", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--total-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/checkpoints/sft_v1")
    parser.add_argument("--sample-prompt", type=str, default="The most important thing to understand is")
    parser.add_argument("--sample-max-new-tokens", type=int, default=200)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--backend", type=str, default="ring")
    parser.add_argument("--collective-stream", type=str, default="cpu", choices=["cpu", "default"])
    parser.add_argument("--expected-world", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument("--optimizer-rank0-only", action="store_true", default=True)

    if pre_args.config:
        cfg_path = Path(pre_args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        parser.set_defaults(**_load_config_defaults(str(cfg_path)))
        parser.set_defaults(config=str(cfg_path))

    args = parser.parse_args(remaining)

    if args.base_ckpt:
        meta_path = args.base_ckpt + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                base_meta = json.load(f)
            _inherit_model_args(args, base_meta)

    defaults = {
        "block_size": 1024,
        "vocab_size": 50304,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.0,
        "bias": False,
        "attention_impl": "fast",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if (
        args.expected_world == 1
        and not os.environ.get("MLX_RANK")
        and not os.environ.get("MLX_HOSTFILE")
        and not os.environ.get("MLX_JACCL_COORDINATOR")
    ):
        class _LocalGroup:
            rank = 0
            size = 1

        group = _LocalGroup()
    else:
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
        raise RuntimeError(f"checkpoint_rank must be in [0, {world - 1}]")

    cfg = GPT2Config(
        block_size=int(args.block_size),
        vocab_size=int(args.vocab_size),
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_embd=int(args.n_embd),
        dropout=float(args.dropout),
        bias=bool(args.bias),
        attention_impl=str(args.attention_impl),
    )
    model = GPT2LM(cfg)
    model_dtype = _resolve_dtype(args.dtype)
    _cast_model_floats(model, model_dtype)
    mx.eval(model.parameters())

    optimizer = optim.AdamW(
        learning_rate=float(args.learning_rate),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )

    start_step = 0
    if args.resume:
        if rank == 0:
            meta = _load_checkpoint_into(args.resume, model=model, optimizer=optimizer, load_optimizer=True)
            start_step = int(meta.get("step", meta.get("iter_num", 0)))
            print(f"[resume] path={args.resume} start_step={start_step}", flush=True)
    elif args.base_ckpt:
        if rank == 0:
            _ = _load_checkpoint_into(
                args.base_ckpt,
                model=model,
                optimizer=optimizer,
                load_optimizer=bool(args.load_optimizer),
            )
            print(
                f"[base] ckpt={args.base_ckpt} optimizer_warmstart={'yes' if args.load_optimizer else 'no'}",
                flush=True,
            )

    model.update(_broadcast_tree_from_rank0(model.parameters(), rank, world, args.collective_stream))
    optimizer.state = _broadcast_tree_from_rank0(optimizer.state, rank, world, args.collective_stream)
    mx.eval(model.parameters(), optimizer.state)

    if args.resume and world > 1:
        step_arr = _all_sum(mx.array(float(start_step if rank == 0 else 0.0), dtype=mx.float32), stream_mode=args.collective_stream)
        mx.eval(step_arr)
        start_step = int(step_arr.item())

    train_specs = parse_source_specs(args.train_sources, default_shuffle_buffer=args.shuffle_buffer)
    val_specs = parse_source_specs(args.val_sources, default_shuffle_buffer=args.shuffle_buffer)
    train_batcher = GPT2ConversationBatcher(
        sources=train_specs,
        tokenizer_name=args.tokenizer_name,
        rank=rank,
        world=world,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        shuffle_buffer=args.shuffle_buffer,
        ignore_index=args.ignore_index,
        buffer_size=args.buffer_size,
    )
    val_batcher = None
    if val_specs:
        val_batcher = GPT2ConversationBatcher(
            sources=val_specs,
            tokenizer_name=args.tokenizer_name,
            rank=rank,
            world=world,
            seed=args.seed + 10_000_000,
            trust_remote_code=args.trust_remote_code,
            shuffle_buffer=args.shuffle_buffer,
            ignore_index=args.ignore_index,
            buffer_size=args.buffer_size,
        )
    _sync_ranks(world, args.collective_stream)

    world_tokens_per_micro = args.device_batch_size * args.block_size * world
    if args.total_batch_size % world_tokens_per_micro != 0:
        raise ValueError(
            f"total_batch_size={args.total_batch_size} must be divisible by "
            f"device_batch_size*block_size*world={world_tokens_per_micro}"
        )
    grad_accum = args.total_batch_size // world_tokens_per_micro
    lr_for_step = _build_lr_schedule(
        base_lr=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_iters,
    )

    if rank == 0:
        num_params = count_parameters(model)
        param_gib = (num_params * (2 if model_dtype in (mx.float16, mx.bfloat16) else 4)) / (1024**3)
        print(
            f"[model] params={num_params/1e6:.2f}M dtype={args.dtype} approx_param_mem={param_gib:.2f} GiB/rank",
            flush=True,
        )
        print(
            f"[sft] world={world} block={args.block_size} device_batch={args.device_batch_size} "
            f"total_batch_tokens={args.total_batch_size:,} grad_accum={grad_accum} "
            f"optimizer_warmstart={'yes' if args.load_optimizer else 'no'}",
            flush=True,
        )
        print(
            f"[data] train_sources={len(train_specs)} val_sources={len(val_specs)} tokenizer={args.tokenizer_name}",
            flush=True,
        )

    def loss_fn(x, y):
        logits = model.logits(x)
        per_token = losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
        mask = (y != args.ignore_index).astype(per_token.dtype)
        return (per_token * mask).sum() / (mask.sum() + 1e-6)

    step_and_grad = nn.value_and_grad(model, loss_fn)
    ema_loss = None
    loop_start = time.perf_counter()

    for step in range(start_step, args.max_iters):
        total_loss_local = 0.0
        grads_acc = None
        t0 = time.perf_counter()

        for _ in range(grad_accum):
            x, y = train_batcher.sample_batch(batch_size=args.device_batch_size, seq_len=args.block_size)
            _sync_ranks(world, args.collective_stream)
            loss, grads = step_and_grad(x, y)
            mx.eval(loss)
            total_loss_local += float(loss.item())
            grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)

        grads_acc = _tree_scale(grads_acc, 1.0 / float(grad_accum))
        if world > 1:
            grads_acc = _allreduce_tree(grads_acc, world, stream_mode=args.collective_stream)

        grads_acc, grad_norm = _clip_grads(grads_acc, args.grad_clip)
        lr_t = lr_for_step(step)
        if args.optimizer_rank0_only and world > 1:
            if rank == 0:
                optimizer.learning_rate = lr_t
                optimizer.update(model, grads_acc)
                mx.eval(model.parameters(), optimizer.state)
            model.update(_broadcast_tree_from_rank0(model.parameters(), rank, world, args.collective_stream))
        else:
            optimizer.learning_rate = lr_t
            optimizer.update(model, grads_acc)
            mx.eval(model.parameters(), optimizer.state)

        step_loss = mx.array(total_loss_local / float(grad_accum), dtype=mx.float32)
        if world > 1:
            step_loss = _all_sum(step_loss, stream_mode=args.collective_stream) / world
        mx.eval(step_loss)
        step_loss_value = float(step_loss.item())
        ema_loss = step_loss_value if ema_loss is None else (0.98 * ema_loss + 0.02 * step_loss_value)
        dt = time.perf_counter() - t0
        toks_per_sec = args.total_batch_size / max(dt, 1e-9)

        if rank == 0 and ((step + 1) % args.log_interval == 0 or step == start_step):
            print(
                f"[step {step+1:6d}] loss={step_loss_value:.4f} ema={ema_loss:.4f} "
                f"lr={lr_t:.3e} grad_norm={grad_norm:.3f} tok/s={toks_per_sec:,.0f}",
                flush=True,
            )

        should_eval = args.eval_interval > 0 and (step + 1) % args.eval_interval == 0
        if should_eval:
            _sync_ranks(world, args.collective_stream)
            val_loss = None
            if val_batcher is not None:
                val_loss = _estimate_loss(
                    model=model,
                    batcher=val_batcher,
                    eval_iters=args.eval_iters,
                    batch_size=args.device_batch_size,
                    block_size=args.block_size,
                    ignore_index=args.ignore_index,
                    rank=rank,
                    world=world,
                    collective_stream=args.collective_stream,
                )
            _sync_ranks(world, args.collective_stream)
            if rank == 0 and val_loss is not None:
                print(f"[eval {step+1:6d}] val_loss={val_loss:.4f}", flush=True)

        should_save = args.save_every > 0 and (step + 1) % args.save_every == 0
        if should_save and rank == args.checkpoint_rank:
            ckpt_path = os.path.join(args.save_dir, f"step_{step+1:07d}.safetensors")
            _save_training_checkpoint(
                ckpt_path,
                model,
                optimizer,
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
            if args.sample_prompt:
                sample_text = _generate_sample_text(
                    model=model,
                    tokenizer_name=args.tokenizer_name,
                    prompt=args.sample_prompt,
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temperature,
                    top_k=args.sample_top_k,
                )
                print(f"[sample {step+1:6d}] prompt={args.sample_prompt!r}", flush=True)
                print(sample_text, flush=True)
        if should_save:
            _sync_ranks(world, args.collective_stream)

    if rank == args.checkpoint_rank:
        final_path = os.path.join(args.save_dir, "final.safetensors")
        _save_training_checkpoint(
            final_path,
            model,
            optimizer,
            {
                "step": args.max_iters,
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
