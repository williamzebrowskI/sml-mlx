#!/usr/bin/env python3
"""Hybrid streaming nanoGPT-style GPT-2 pretraining in MLX."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distributed_nanogpt_streaming.model import GPT2Config, GPT2LM, count_parameters
from distributed_nanogpt_streaming.streaming_tokens import (
    HFTokenStreamingBatcher,
    StreamingTokenDatasetAdapter,
    data_state_path,
    load_data_state,
    parse_token_source_configs,
    save_data_state,
)
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

    def visit(v):
        if not isinstance(v, mx.array):
            return v
        src = v if rank == 0 else mx.zeros_like(v)
        return _all_sum(src, stream_mode=stream_mode)

    if isinstance(tree, dict):
        return {k: _broadcast_tree_from_rank0(v, rank, world, stream_mode) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_broadcast_tree_from_rank0(v, rank, world, stream_mode) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_broadcast_tree_from_rank0(v, rank, world, stream_mode) for v in tree)
    return visit(tree)


def _save_training_checkpoint(path: str, model: GPT2LM, optimizer: optim.AdamW, metadata: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensors: Dict[str, mx.array] = {}
    _flatten_for_safetensors(model.parameters(), prefix="model", out=tensors)
    _flatten_for_safetensors(optimizer.state, prefix="optimizer", out=tensors)
    mx.save_safetensors(path, tensors)
    with open(path + ".json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _load_training_checkpoint(path: str, model: GPT2LM, optimizer: optim.AdamW) -> Dict[str, Any]:
    flat = mx.load(path)
    model.update(_tree_assign(model.parameters(), flat, prefix="model"))
    optimizer.state = _tree_assign(optimizer.state, flat, prefix="optimizer")
    mx.eval(model.parameters(), optimizer.state)
    meta_path = path + ".json"
    metadata: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    return metadata


def _load_hf_gpt2(model: GPT2LM, model_type: str, *, dropout: float) -> None:
    from transformers import GPT2LMHeadModel

    del dropout
    hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd = hf.state_dict()
    current = {}
    _flatten_for_safetensors(model.parameters(), out=current)

    for key, tensor in sd.items():
        if key.endswith("attn.masked_bias") or key.endswith("attn.bias") or key == "lm_head.weight":
            continue
        if not key.startswith("transformer."):
            continue
        dest = key[len("transformer.") :]
        if dest not in current:
            continue
        arr = tensor.detach().cpu().numpy()
        if dest.endswith(("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")):
            arr = arr.T
        target = current[dest]
        value = mx.array(arr, dtype=target.dtype)
        if value.shape != target.shape:
            raise ValueError(f"Shape mismatch loading {key}: hf={value.shape} mlx={target.shape}")
        current[dest] = value

    model.update(_tree_assign(model.parameters(), current))
    mx.eval(model.parameters())


def _estimate_loss(
    *,
    model: GPT2LM,
    dataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    seed: int,
    iter_num: int,
    rank: int,
    world: int,
    collective_stream: str,
) -> float:
    losses = []
    was_training = model.training
    model.eval()
    for i in range(eval_iters):
        x, y = dataset.sample_batch(
            batch_size=batch_size,
            seq_len=block_size,
            seed=seed,
            step=iter_num + i,
            rank=rank,
            stream=91,
        )
        loss = model(x, targets=y)["loss"]
        if world > 1:
            loss = _all_sum(loss, stream_mode=collective_stream) / world
        mx.eval(loss)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(sum(losses) / max(1, len(losses)))


def _load_sample_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def _generate_sample_text(
    *,
    model: GPT2LM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("Prompt tokenized to empty ids and tokenizer has no eos_token_id fallback.")
        prompt_ids = [int(eos_id)]

    vocab_limit = int(getattr(tokenizer, "vocab_size", model.config.vocab_size))
    out_ids = list(int(x) for x in prompt_ids)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            ctx = out_ids[-model.config.block_size :]
            x = mx.array(np.asarray([ctx], dtype=np.int32), dtype=mx.int32)
            logits = model.logits(x)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            logits_np = np.asarray(logits[0])[:vocab_limit]

            if temperature <= 0.0 or top_k == 1:
                next_id = int(np.argmax(logits_np))
            else:
                scaled = logits_np / max(temperature, 1e-5)
                if top_k > 0 and top_k < scaled.shape[0]:
                    keep = np.argpartition(scaled, -top_k)[-top_k:]
                    masked = np.full_like(scaled, -np.inf)
                    masked[keep] = scaled[keep]
                    scaled = masked
                scaled = scaled - np.max(scaled)
                probs = np.exp(scaled)
                probs = probs / np.clip(probs.sum(), 1e-12, None)
                next_id = int(np.random.choice(probs.shape[0], p=probs))
            out_ids.append(next_id)
    finally:
        if was_training:
            model.train()

    return tokenizer.decode(out_ids, clean_up_tokenization_spaces=False)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre], description="Hybrid streaming nanoGPT-style GPT-2 pretraining")
    parser.add_argument("--train-sources", type=str, default="")
    parser.add_argument("--val-sources", type=str, default="")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--always-save-checkpoint", action="store_true", default=True)
    parser.add_argument("--init-from", type=str, default="scratch", choices=["scratch", "resume", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--attention-impl", type=str, default="fast", choices=["fast", "vanilla"])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-iters", type=int, default=600000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--decay-lr", action="store_true", default=True)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--lr-decay-iters", type=int, default=600000)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--backend", type=str, default="ring")
    parser.add_argument("--collective-stream", type=str, default="cpu", choices=["cpu", "default"])
    parser.add_argument("--expected-world", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="/Users/williamzebrowski/sml-mlx/distributed_nanogpt_streaming/checkpoints/gpt2_streaming_climbmix_v1")
    parser.set_defaults(save_stream_state=False)
    parser.add_argument("--save-stream-state", action="store_true", dest="save_stream_state")
    parser.add_argument("--no-save-stream-state", action="store_false", dest="save_stream_state")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--trace-first-step", action="store_true")
    parser.add_argument("--sample-prompt", type=str, default="")
    parser.add_argument("--sample-max-new-tokens", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--sample-tokenizer-name", type=str, default="gpt2")

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
    except RuntimeError as e:
        if args.expected_world == 1:
            try:
                group = mx.distributed.init(backend="any", strict=False)
            except TypeError:
                group = mx.distributed.init(backend="any")
        else:
            raise e

    rank = int(group.rank() if callable(getattr(group, "rank", None)) else group.rank)
    world = int(group.size() if callable(getattr(group, "size", None)) else group.size)
    if args.expected_world is not None and world != args.expected_world:
        raise RuntimeError(f"Expected world={args.expected_world}, got {world}")
    if args.gradient_accumulation_steps <= 0:
        raise RuntimeError("gradient_accumulation_steps must be > 0")
    if args.gradient_accumulation_steps % world != 0:
        raise RuntimeError(
            f"gradient_accumulation_steps={args.gradient_accumulation_steps} must be divisible by world={world}"
        )
    if args.checkpoint_rank < 0 or args.checkpoint_rank >= world:
        raise RuntimeError(f"checkpoint_rank must be in [0, {world - 1}]")

    seed_offset = rank
    mx.random.seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    train_sources = parse_token_source_configs(args.train_sources)
    if not train_sources:
        raise ValueError("Missing train_sources for streaming GPT-2 training.")
    val_sources = parse_token_source_configs(args.val_sources) if args.val_sources else []

    model_args = {
        "block_size": args.block_size,
        "vocab_size": args.vocab_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "bias": args.bias,
        "attention_impl": args.attention_impl,
    }

    if args.init_from == "resume" and not args.resume and not args.checkpoint_path:
        default_resume = os.path.join(args.save_dir, "ckpt.safetensors")
        args.resume = default_resume

    if args.init_from.startswith("gpt2"):
        preset = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[args.init_from]
        model_args.update(preset)
        model_args["vocab_size"] = 50257
        model_args["bias"] = True
        model_args["block_size"] = 1024

    cfg = GPT2Config(**model_args)
    model = GPT2LM(cfg)
    model_dtype = _resolve_dtype(args.dtype)
    _cast_model_floats(model, model_dtype)
    mx.eval(model.parameters())
    model.train()

    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        betas=[args.beta1, args.beta2],
        weight_decay=args.weight_decay,
        bias_correction=True,
    )

    train_batcher = HFTokenStreamingBatcher(
        sources=train_sources,
        world_size=world,
        rank=rank,
        seed=args.seed,
    )
    train_data = StreamingTokenDatasetAdapter(train_batcher)

    if val_sources:
        val_batcher = HFTokenStreamingBatcher(
            sources=val_sources,
            world_size=world,
            rank=rank,
            seed=args.seed + 99991,
        )
        val_data = StreamingTokenDatasetAdapter(val_batcher)
    else:
        val_batcher = None
        val_data = None

    iter_num = 0
    best_val_loss = 1e9

    if args.init_from == "resume":
        resume_path = args.resume or args.checkpoint_path
        if rank == 0:
            if not resume_path or not os.path.exists(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            meta_resume = _load_training_checkpoint(resume_path, model, optimizer)
            iter_num = int(meta_resume.get("iter_num", meta_resume.get("step", 0)))
            best_val_loss = float(meta_resume.get("best_val_loss", best_val_loss))
        if args.save_stream_state:
            ds_path = data_state_path(resume_path, rank)
            payload = load_data_state(ds_path)
            if payload is not None:
                train_batcher.load_state_dict(payload["train"])
                if val_batcher is not None and payload.get("val") is not None:
                    val_batcher.load_state_dict(payload["val"])
            else:
                print(f"[rank {rank}] stream state not found for resume: {ds_path}", flush=True)
    elif args.init_from.startswith("gpt2"):
        if rank == 0:
            _load_hf_gpt2(model, args.init_from, dropout=args.dropout)

    model.update(_broadcast_tree_from_rank0(model.parameters(), rank, world, args.collective_stream))
    mx.eval(model.parameters())
    if args.init_from == "resume":
        optimizer.state = _broadcast_tree_from_rank0(optimizer.state, rank, world, args.collective_stream)
        step_arr = _all_sum(mx.array(float(iter_num if rank == 0 else 0.0), dtype=mx.float32), stream_mode=args.collective_stream)
        best_arr = _all_sum(mx.array(float(best_val_loss if rank == 0 else 0.0), dtype=mx.float32), stream_mode=args.collective_stream)
        mx.eval(step_arr, best_arr, optimizer.state)
        iter_num = int(step_arr.item())
        best_val_loss = float(best_arr.item())

    local_accum = args.gradient_accumulation_steps // world
    tokens_per_iter = args.gradient_accumulation_steps * args.batch_size * model.config.block_size
    num_params = count_parameters(model)
    param_bytes = 2 if model_dtype in (mx.float16, mx.bfloat16) else 4
    sample_tokenizer: Optional[Any] = None
    sample_enabled = bool(args.sample_prompt and args.sample_max_new_tokens > 0)
    if sample_enabled and rank == args.checkpoint_rank:
        sample_tokenizer = _load_sample_tokenizer(args.sample_tokenizer_name)
    if rank == 0:
        print(f"[rank {rank}] host={socket.gethostname()} world={world}", flush=True)
        print(f"[tokens_per_iter] {tokens_per_iter:,}", flush=True)
        print(
            f"[model] params={num_params/1e6:.2f}M dtype={args.dtype} approx_param_mem={(num_params*param_bytes)/(1024**3):.2f} GiB/rank",
            flush=True,
        )
        print(
            f"[train] init_from={args.init_from} block={model.config.block_size} batch={args.batch_size} "
            f"grad_accum_global={args.gradient_accumulation_steps} grad_accum_local={local_accum} "
            f"save_stream_state={args.save_stream_state}",
            flush=True,
        )
        print(f"[data] train_sources={len(train_sources)} val_sources={len(val_sources)}", flush=True)

    lr_for_step = _build_lr_schedule(
        base_lr=args.learning_rate,
        min_lr_ratio=args.min_lr / max(args.learning_rate, 1e-12),
        warmup_steps=args.warmup_iters,
        max_steps=args.lr_decay_iters,
    )

    def loss_fn(x, y):
        return model(x, targets=y)["loss"]

    step_and_grad = nn.value_and_grad(model, loss_fn)
    running_loss = None
    loop_start = time.perf_counter()

    while iter_num < args.max_iters:
        should_eval = args.eval_interval > 0 and iter_num > 0 and (iter_num % args.eval_interval == 0)
        if should_eval:
            ckpt_path = os.path.join(args.save_dir, "ckpt.safetensors")
            train_loss = _estimate_loss(
                model=model,
                dataset=train_data,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=model.config.block_size,
                seed=args.seed,
                iter_num=iter_num,
                rank=rank,
                world=world,
                collective_stream=args.collective_stream,
            )
            val_loss = None
            if val_data is not None:
                val_loss = _estimate_loss(
                    model=model,
                    dataset=val_data,
                    eval_iters=args.eval_iters,
                    batch_size=args.batch_size,
                    block_size=model.config.block_size,
                    seed=args.seed,
                    iter_num=iter_num,
                    rank=rank,
                    world=world,
                    collective_stream=args.collective_stream,
                )
            if rank == 0:
                msg = f"[eval {iter_num:7d}] train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.4f}"
                print(msg, flush=True)
                save_ckpt = args.always_save_checkpoint or (val_loss is not None and val_loss < best_val_loss)
                if val_loss is not None:
                    best_val_loss = min(best_val_loss, val_loss)
                if save_ckpt:
                    _save_training_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        {
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "model_args": model_args,
                            "world": world,
                            "backend": args.backend,
                            "timestamp": time.time(),
                        },
                    )
                    print(f"[ckpt] saved {ckpt_path}", flush=True)
                    if sample_enabled and rank == args.checkpoint_rank:
                        sample_text = _generate_sample_text(
                            model=model,
                            tokenizer=sample_tokenizer,
                            prompt=args.sample_prompt,
                            max_new_tokens=args.sample_max_new_tokens,
                            temperature=args.sample_temperature,
                            top_k=args.sample_top_k,
                        )
                        print(f"[sample {iter_num:7d}] prompt={args.sample_prompt!r}", flush=True)
                        print(sample_text, flush=True)
            else:
                save_ckpt = False
            if save_ckpt and args.save_stream_state:
                payload = {
                    "train": train_batcher.state_dict(),
                    "val": val_batcher.state_dict() if val_batcher is not None else None,
                }
                save_data_state(data_state_path(ckpt_path, rank), payload)
            if world > 1:
                # Keep all ranks aligned when rank 0 spends extra time saving
                # checkpoints or generating a sample preview.
                eval_sync = _all_sum(mx.array(1.0, dtype=mx.float32), stream_mode=args.collective_stream)
                mx.eval(eval_sync)
            if args.eval_only:
                break
            model.train()

        t0 = time.perf_counter()
        total_loss_local = 0.0
        grads_acc = None

        for micro in range(local_accum):
            if args.trace_first_step and iter_num == 0:
                print(f"[rank {rank}] trace iter={iter_num} micro={micro} stage=sample_batch", flush=True)
            x, y = train_data.sample_batch(
                batch_size=args.batch_size,
                seq_len=model.config.block_size,
                seed=args.seed,
                step=iter_num,
                rank=rank,
                stream=micro,
            )
            if args.trace_first_step and iter_num == 0:
                print(f"[rank {rank}] trace iter={iter_num} micro={micro} stage=sample_done", flush=True)
            if world > 1:
                if args.trace_first_step and iter_num == 0:
                    print(f"[rank {rank}] trace iter={iter_num} micro={micro} stage=sample_sync", flush=True)
                ready = _all_sum(mx.array(1.0, dtype=mx.float32), stream_mode=args.collective_stream)
                mx.eval(ready)
                if args.trace_first_step and iter_num == 0:
                    print(f"[rank {rank}] trace iter={iter_num} micro={micro} stage=sample_sync_done", flush=True)
            if args.trace_first_step and iter_num == 0:
                print(f"[rank {rank}] trace iter={iter_num} micro={micro} stage=fwd_bwd", flush=True)
            loss, grads = step_and_grad(x, y)
            mx.eval(loss)
            total_loss_local += float(loss.item())
            grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)

        grads_acc = _tree_scale(grads_acc, 1.0 / float(local_accum))
        if world > 1:
            if args.trace_first_step and iter_num == 0:
                print(f"[rank {rank}] trace iter={iter_num} stage=allreduce_grads", flush=True)
            grads_acc = _allreduce_tree(grads_acc, world, stream_mode=args.collective_stream)

        grads_acc, grad_norm = _clip_grads(grads_acc, args.grad_clip)
        lr_t = lr_for_step(iter_num) if args.decay_lr else args.learning_rate
        optimizer.learning_rate = lr_t
        optimizer.update(model, grads_acc)
        mx.eval(model.parameters(), optimizer.state)

        step_loss = mx.array(total_loss_local / float(local_accum), dtype=mx.float32)
        if world > 1:
            step_loss = _all_sum(step_loss, stream_mode=args.collective_stream) / world
        mx.eval(step_loss)
        step_loss_value = float(step_loss.item())
        if not math.isfinite(step_loss_value) or not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite at iter {iter_num}: loss={step_loss_value} grad_norm={grad_norm}")

        running_loss = step_loss_value if running_loss is None else (0.95 * running_loss + 0.05 * step_loss_value)
        dt = time.perf_counter() - t0
        toks_per_sec = tokens_per_iter / max(dt, 1e-9)
        iter_num += 1

        if rank == 0 and (iter_num % args.log_interval == 0 or iter_num == 1):
            print(
                f"[iter {iter_num:7d}] loss={step_loss_value:.4f} ema={running_loss:.4f} "
                f"lr={lr_t:.3e} grad_norm={grad_norm:.3f} tok/s={toks_per_sec:,.0f}",
                flush=True,
            )

    if rank == args.checkpoint_rank:
        final_path = os.path.join(args.save_dir, "final.safetensors")
        _save_training_checkpoint(
            final_path,
            model,
            optimizer,
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "model_args": model_args,
                "world": world,
                "backend": args.backend,
                "duration_sec": time.perf_counter() - loop_start,
                "timestamp": time.time(),
            },
        )
        print(f"[done] saved {final_path}", flush=True)
    if args.save_stream_state:
        final_payload = {
            "train": train_batcher.state_dict(),
            "val": val_batcher.state_dict() if val_batcher is not None else None,
        }
        save_data_state(data_state_path(os.path.join(args.save_dir, "final.safetensors"), rank), final_payload)


if __name__ == "__main__":
    main()
