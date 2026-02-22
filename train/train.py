#!/usr/bin/env python3
"""Distributed MLX pretraining runner for a small transformer LM.

Recommended launch (4 hosts, JACCL):

  /Users/williamzebrowski/sml-mlx/.venv/bin/mlx.launch \
    --hostfile /Users/williamzebrowski/sml-mlx/hosts_tb_jaccl.json \
    --backend jaccl -- \
    /Users/williamzebrowski/sml-mlx/.venv/bin/python3 \
    /Users/williamzebrowski/sml-mlx/train/train.py \
    --config /Users/williamzebrowski/sml-mlx/train/config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from safetensors.numpy import load_file as safetensors_load

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

try:
    from .model import TransformerConfig, TransformerLM, count_parameters
    from .data import (
        HFStreamingBatcher,
        StreamingDatasetAdapter,
        data_state_path,
        load_data_state,
        parse_source_configs,
        save_data_state,
    )
except ImportError:
    from model import TransformerConfig, TransformerLM, count_parameters
    from data import (
        HFStreamingBatcher,
        StreamingDatasetAdapter,
        data_state_path,
        load_data_state,
        parse_source_configs,
        save_data_state,
    )


def _tree_leaves(tree: Any):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _tree_leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _tree_leaves(v)
    else:
        yield tree


def _tree_add(a: Any, b: Any):
    if isinstance(a, dict):
        return {k: _tree_add(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [_tree_add(x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple):
        return tuple(_tree_add(x, y) for x, y in zip(a, b))
    if isinstance(a, mx.array):
        return a + b
    return a


def _tree_scale(tree: Any, scale: float):
    if isinstance(tree, dict):
        return {k: _tree_scale(v, scale) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_scale(v, scale) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_scale(v, scale) for v in tree)
    if isinstance(tree, mx.array):
        return tree * scale
    return tree


def _flatten_for_safetensors(tree: Any, prefix: str = "", out: Optional[dict] = None):
    if out is None:
        out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_for_safetensors(v, key, out)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            key = f"{prefix}.{i}" if prefix else str(i)
            _flatten_for_safetensors(v, key, out)
    elif isinstance(tree, tuple):
        for i, v in enumerate(tree):
            key = f"{prefix}.{i}" if prefix else str(i)
            _flatten_for_safetensors(v, key, out)
    elif isinstance(tree, mx.array):
        key = prefix if prefix else "param"
        out[key] = tree
    return out


def _all_sum(x: mx.array) -> mx.array:
    try:
        return mx.distributed.all_sum(x)
    except TypeError:
        return mx.distributed.all_sum(x, stream=mx.cpu)


def _allreduce_tree(tree: Any, world: int):
    if world == 1:
        return tree

    def reduce_leaf(v):
        if isinstance(v, mx.array):
            return _all_sum(v) / world
        return v

    return tree_map(reduce_leaf, tree)


def _grad_norm(tree: Any) -> float:
    sq = mx.array(0.0, dtype=mx.float32)
    for leaf in _tree_leaves(tree):
        if isinstance(leaf, mx.array):
            x = leaf.astype(mx.float32)
            sq = sq + (x * x).sum()
    norm = mx.sqrt(sq + 1e-12)
    mx.eval(norm)
    return float(norm.item())


def _clip_grads(tree: Any, max_norm: float):
    if max_norm <= 0:
        return tree, 0.0
    norm = _grad_norm(tree)
    if norm <= max_norm:
        return tree, norm
    scale = max_norm / (norm + 1e-6)
    return _tree_scale(tree, scale), norm


def _cast_model_floats(model: nn.Module, dtype):
    float_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    casted = tree_map(
        lambda x: x.astype(dtype)
        if isinstance(x, mx.array) and x.dtype in float_dtypes
        else x,
        model.parameters(),
    )
    model.update(casted)


def _save_checkpoint(path: str, model: nn.Module, metadata: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensors = _flatten_for_safetensors(model.parameters())
    mx.save_safetensors(path, tensors)
    with open(path + ".json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _load_checkpoint(path: str, model: nn.Module) -> bool:
    if not os.path.exists(path):
        return False

    flat = safetensors_load(path)

    def assign(template: Any, prefix: str = ""):
        if isinstance(template, dict):
            return {k: assign(v, f"{prefix}.{k}" if prefix else k) for k, v in template.items()}
        if isinstance(template, list):
            return [assign(v, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(template)]
        if isinstance(template, tuple):
            return tuple(assign(v, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(template))
        if isinstance(template, mx.array):
            key = prefix if prefix else "param"
            if key not in flat:
                return template
            value = mx.array(flat[key]).astype(template.dtype)
            if value.shape != template.shape:
                raise ValueError(
                    f"Checkpoint shape mismatch for {key}: file={value.shape}, model={template.shape}"
                )
            return value
        return template

    model.update(assign(model.parameters()))
    mx.eval(model.parameters())
    return True


def _infer_resume_step(resume_path: str) -> int:
    meta_path = resume_path + ".json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return int(meta.get("step", 0))
        except Exception:
            pass

    stem = Path(resume_path).stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def _broadcast_model_from_rank0(model: nn.Module, rank: int, world: int):
    if world == 1:
        return

    def bcast(v):
        if not isinstance(v, mx.array):
            return v
        src = v if rank == 0 else mx.zeros_like(v)
        return _all_sum(src)

    params = tree_map(bcast, model.parameters())
    model.update(params)
    mx.eval(model.parameters())


class TokenDataset:
    def __init__(self, path: str, token_dtype: str):
        self.path = path
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
        else:
            arr = np.memmap(path, dtype=np.dtype(token_dtype), mode="r")
        self.tokens = arr.reshape(-1)
        self.n_tokens = int(self.tokens.shape[0])
        if self.n_tokens < 2:
            raise ValueError(f"Dataset at {path} has too few tokens: {self.n_tokens}")

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        seed: int,
        step: int,
        rank: int,
        stream: int = 0,
    ):
        max_start = self.n_tokens - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset at {self.path} has {self.n_tokens} tokens, need > seq_len+1 ({seq_len + 1})"
            )

        step_seed = seed + (step * 1_000_003) + (rank * 100_003) + (stream * 9_973)
        rng = np.random.default_rng(step_seed)
        starts = rng.integers(0, max_start, size=batch_size)

        x = np.empty((batch_size, seq_len), dtype=np.int32)
        y = np.empty((batch_size, seq_len), dtype=np.int32)
        for i, s in enumerate(starts.tolist()):
            chunk = self.tokens[s : s + seq_len + 1]
            x[i] = np.asarray(chunk[:-1], dtype=np.int32)
            y[i] = np.asarray(chunk[1:], dtype=np.int32)

        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

def _resolve_dtype(name: str):
    table = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _build_lr_schedule(
    base_lr: float,
    min_lr_ratio: float,
    warmup_steps: int,
    max_steps: int,
):
    min_lr = base_lr * min_lr_ratio

    def lr_for_step(step: int) -> float:
        if step < warmup_steps:
            return base_lr * float(step + 1) / float(max(1, warmup_steps))
        if step >= max_steps:
            return min_lr
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    return lr_for_step


def _evaluate(
    model: TransformerLM,
    dataset: TokenDataset,
    eval_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    step: int,
    rank: int,
    world: int,
    ignore_index: int,
) -> float:
    losses = []
    for i in range(eval_steps):
        x, y = dataset.sample_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed,
            step=step + i,
            rank=rank,
            stream=99,
        )
        loss = model(x, targets=y, ignore_index=ignore_index)["loss"]
        if world > 1:
            loss = _all_sum(loss) / world
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def _load_config_defaults(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must be an object (key/value pairs).")
    normalized = {}
    for k, v in raw.items():
        normalized[k.replace("-", "_")] = v
    return normalized


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        parents=[pre],
        description="Small distributed MLX pretraining runner (36 GB worker-friendly defaults)"
    )
    parser.add_argument("--data-mode", type=str, default="tokens", choices=["tokens", "hf_stream"])
    parser.add_argument("--train-tokens", type=str, default="", help="Path to train tokens (.npy or flat binary)")
    parser.add_argument("--val-tokens", type=str, default="", help="Optional validation tokens (.npy or flat binary)")
    parser.add_argument("--token-dtype", type=str, default="uint16", choices=["uint16", "uint32", "int32"])
    parser.add_argument(
        "--spm-model",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/tokenizer/fineweb_spm/spm.model",
        help="SentencePiece model used for HF text streaming mode.",
    )
    parser.add_argument(
        "--train-sources",
        type=str,
        default="",
        help="HF source list (JSON string or path) for data-mode=hf_stream.",
    )
    parser.add_argument(
        "--val-sources",
        type=str,
        default="",
        help="Optional HF source list (JSON string or path) for eval in hf_stream mode.",
    )
    parser.set_defaults(add_bos=True, add_eos=True)
    parser.add_argument("--add-bos", action="store_true", dest="add_bos")
    parser.add_argument("--no-add-bos", action="store_false", dest="add_bos")
    parser.add_argument("--add-eos", action="store_true", dest="add_eos")
    parser.add_argument("--no-add-eos", action="store_false", dest="add_eos")
    parser.add_argument("--vocab-size", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attention-impl", type=str, default="fast", choices=["fast", "vanilla"])
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank micro-batch")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.set_defaults(optimizer_rank0_only=False)
    parser.add_argument(
        "--optimizer-rank0-only",
        action="store_true",
        dest="optimizer_rank0_only",
        help="Only rank 0 applies optimizer updates; other ranks receive weights via broadcast each step.",
    )
    parser.add_argument(
        "--no-optimizer-rank0-only",
        action="store_false",
        dest="optimizer_rank0_only",
    )

    parser.add_argument("--backend", type=str, default=os.getenv("MLX_BACKEND", "jaccl"))
    parser.add_argument("--expected-world", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=20)

    parser.add_argument("--save-dir", type=str, default="/Users/williamzebrowski/sml-mlx/train/checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", type=str, default="")

    if pre_args.config:
        cfg_path = Path(pre_args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg_defaults = _load_config_defaults(str(cfg_path))
        unknown = sorted(set(cfg_defaults.keys()) - set(vars(parser.parse_args([])).keys()))
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**cfg_defaults)
        parser.set_defaults(config=str(cfg_path))

    args = parser.parse_args(remaining)
    if args.vocab_size <= 0:
        raise ValueError("Missing/invalid --vocab-size (or vocab_size in --config).")
    if args.data_mode == "tokens" and not args.train_tokens:
        raise ValueError("Missing --train-tokens (or train_tokens in --config).")
    if args.data_mode == "hf_stream" and not args.train_sources:
        raise ValueError("Missing --train-sources (or train_sources in --config) for hf_stream mode.")

    try:
        group = mx.distributed.init(backend=args.backend, strict=True)
    except TypeError:
        group = mx.distributed.init(backend=args.backend)

    rank = int(group.rank() if callable(getattr(group, "rank", None)) else group.rank)
    world = int(group.size() if callable(getattr(group, "size", None)) else group.size)
    if args.expected_world is not None and world != args.expected_world:
        raise RuntimeError(f"Expected world={args.expected_world}, got {world}")

    model_dtype = _resolve_dtype(args.dtype)
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
    _cast_model_floats(model, model_dtype)
    mx.eval(model.parameters())

    num_params = count_parameters(model)
    dtype_bytes = 2 if model_dtype in (mx.float16, mx.bfloat16) else 4
    param_gib = (num_params * dtype_bytes) / (1024**3)
    if rank == 0:
        print(
            f"[model] params={num_params/1e6:.2f}M dtype={args.dtype} "
            f"approx_param_mem={param_gib:.2f} GiB/rank",
            flush=True,
        )
        print(
            f"[train] world={world} backend={args.backend} "
            f"seq={args.max_seq_len} batch={args.batch_size} accum={args.grad_accum} "
            f"opt_mode={'rank0_only' if args.optimizer_rank0_only else 'all_ranks'}",
            flush=True,
        )

    start_step = 0
    if args.resume:
        loaded = _load_checkpoint(args.resume, model)
        if loaded:
            start_step = _infer_resume_step(args.resume)
        if rank == 0:
            print(
                f"[ckpt] resume={args.resume} loaded={loaded} start_step={start_step}",
                flush=True,
            )

    _broadcast_model_from_rank0(model, rank, world)

    train_stream = None
    if args.data_mode == "tokens":
        train_data = TokenDataset(args.train_tokens, args.token_dtype)
        val_data = TokenDataset(args.val_tokens, args.token_dtype) if args.val_tokens else None
        if rank == 0:
            print(
                f"[data] mode=tokens train_tokens={train_data.n_tokens:,}"
                + (f" val_tokens={val_data.n_tokens:,}" if val_data else ""),
                flush=True,
            )
    else:
        train_sources = parse_source_configs(args.train_sources)
        val_sources = parse_source_configs(args.val_sources) if args.val_sources else []
        train_stream = HFStreamingBatcher(
            sources=train_sources,
            spm_model=args.spm_model,
            world_size=world,
            rank=rank,
            seed=args.seed,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
        )
        train_data = StreamingDatasetAdapter(train_stream)

        if val_sources:
            val_stream = HFStreamingBatcher(
                sources=val_sources,
                spm_model=args.spm_model,
                world_size=world,
                rank=rank,
                seed=args.seed + 99991,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
            )
            val_data = StreamingDatasetAdapter(val_stream)
        else:
            val_data = None

        if rank == 0:
            print(
                f"[data] mode=hf_stream train_sources={len(train_sources)} "
                f"val_sources={len(val_sources)} spm_model={args.spm_model}",
                flush=True,
            )
            if any(s.shuffle_buffer > 0 for s in train_sources):
                print(
                    "[warn] shuffle_buffer>0 reduces exact data-cursor resume fidelity. "
                    "Use shuffle_buffer=0 for exact-ish resume.",
                    flush=True,
                )

        if args.resume:
            ds_path = data_state_path(args.resume, rank)
            payload = load_data_state(ds_path)
            if payload is not None:
                state = payload.get("stream_state", payload)
                train_stream.load_state_dict(state)
                if "step" in payload:
                    start_step = max(start_step, int(payload["step"]))
                print(
                    f"[rank {rank}] loaded data state from {ds_path} (step={payload.get('step', 'n/a')})",
                    flush=True,
                )
            else:
                print(
                    f"[rank {rank}] data state not found for resume: {ds_path}",
                    flush=True,
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
    t_loop = time.perf_counter()

    for step in range(start_step, args.max_steps):
        t0 = time.perf_counter()
        total_loss_local = 0.0
        grads_acc = None

        for micro in range(args.grad_accum):
            x, y = train_data.sample_batch(
                batch_size=args.batch_size,
                seq_len=args.max_seq_len,
                seed=args.seed,
                step=step,
                rank=rank,
                stream=micro,
            )
            loss, grads = step_and_grad(x, y)
            mx.eval(loss)
            total_loss_local += float(loss.item())
            grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)

        grads_acc = _tree_scale(grads_acc, 1.0 / float(args.grad_accum))
        if world > 1:
            grads_acc = _allreduce_tree(grads_acc, world)

        grads_acc, grad_norm = _clip_grads(grads_acc, args.grad_clip)
        lr_t = lr_for_step(step)
        if args.optimizer_rank0_only and world > 1:
            if rank == 0:
                optimizer.learning_rate = lr_t
                optimizer.update(model, grads_acc)
                mx.eval(model.parameters(), optimizer.state)
            _broadcast_model_from_rank0(model, rank, world)
        else:
            optimizer.learning_rate = lr_t
            optimizer.update(model, grads_acc)
            mx.eval(model.parameters(), optimizer.state)

        step_loss = mx.array(total_loss_local / float(args.grad_accum), dtype=mx.float32)
        if world > 1:
            step_loss = _all_sum(step_loss) / world
        mx.eval(step_loss)
        step_loss_value = float(step_loss.item())
        if not math.isfinite(step_loss_value) or not math.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite at step {step+1}: loss={step_loss_value}, grad_norm={grad_norm}. "
                "Try --dtype bfloat16 and/or lower --lr."
            )

        ema_loss = step_loss_value if ema_loss is None else (0.98 * ema_loss + 0.02 * step_loss_value)
        dt = time.perf_counter() - t0
        toks_per_step = args.batch_size * args.grad_accum * args.max_seq_len * world
        toks_per_sec = toks_per_step / max(dt, 1e-9)

        if rank == 0 and ((step + 1) % args.log_every == 0 or step == 0):
            print(
                f"[step {step+1:6d}] loss={step_loss_value:.4f} ema={ema_loss:.4f} "
                f"lr={lr_t:.3e} grad_norm={grad_norm:.3f} "
                f"tok/s={toks_per_sec:,.0f}",
                flush=True,
            )

        if val_data and args.eval_every > 0 and ((step + 1) % args.eval_every == 0):
            val_loss = _evaluate(
                model=model,
                dataset=val_data,
                eval_steps=args.eval_steps,
                batch_size=args.batch_size,
                seq_len=args.max_seq_len,
                seed=args.seed + 777_777,
                step=step,
                rank=rank,
                world=world,
                ignore_index=args.ignore_index,
            )
            if rank == 0:
                val_ppl = math.exp(min(20.0, val_loss))
                print(
                    f"[eval {step+1:6d}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}",
                    flush=True,
                )

        if args.save_every > 0 and ((step + 1) % args.save_every == 0):
            ckpt_path = os.path.join(args.save_dir, f"step_{step+1:07d}.safetensors")
            metadata = {
                "step": step + 1,
                "args": vars(args),
                "config": asdict(cfg),
                "world": world,
                "backend": args.backend,
                "timestamp": time.time(),
            }
            if rank == 0:
                _save_checkpoint(ckpt_path, model, metadata)
                print(f"[ckpt] saved {ckpt_path}", flush=True)
            if args.data_mode == "hf_stream" and train_stream is not None:
                ds_payload = {"step": step + 1, "stream_state": train_stream.state_dict()}
                save_data_state(data_state_path(ckpt_path, rank), ds_payload)
                if rank == 0:
                    print(f"[ckpt] saved data-state for all ranks at step {step+1}", flush=True)

    final_path = os.path.join(args.save_dir, "final.safetensors")
    if rank == 0:
        metadata = {
            "step": args.max_steps,
            "args": vars(args),
            "config": asdict(cfg),
            "world": world,
            "backend": args.backend,
            "duration_sec": time.perf_counter() - t_loop,
            "timestamp": time.time(),
        }
        _save_checkpoint(final_path, model, metadata)
        print(f"[done] saved {final_path}", flush=True)
    if args.data_mode == "hf_stream" and train_stream is not None:
        save_data_state(
            data_state_path(final_path, rank),
            {"step": args.max_steps, "stream_state": train_stream.state_dict()},
        )


if __name__ == "__main__":
    main()
