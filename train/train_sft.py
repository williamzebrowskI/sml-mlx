#!/usr/bin/env python3
"""Local chat-style SFT trainer on top of the surviving pretrain stack.

This mirrors the old nanochat-style training logic:
- prompt format: ``User: ...\nAssistant: ...``
- loss is masked on the prompt/prefix and only applied to assistant tokens
- supports HF streaming chat mixtures and local JSONL sources
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import sentencepiece as spm
from datasets import load_dataset

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

try:
    from .model import TransformerConfig, TransformerLM, count_parameters
except ImportError:
    from model import TransformerConfig, TransformerLM, count_parameters


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


def _save_checkpoint(path: str, model: nn.Module, metadata: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensors = _flatten_for_safetensors(model.parameters())
    mx.save_safetensors(path, tensors)
    with open(path + ".json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _load_checkpoint(path: str, model: nn.Module) -> bool:
    if not os.path.exists(path):
        return False

    flat = mx.load(path)

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
            value = flat[key].astype(template.dtype)
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
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return int(meta.get("step", 0))
    stem = Path(resume_path).stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def _cast_model_floats(model: nn.Module, dtype):
    float_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    casted = tree_map(
        lambda x: x.astype(dtype)
        if isinstance(x, mx.array) and x.dtype in float_dtypes
        else x,
        model.parameters(),
    )
    model.update(casted)


def _resolve_dtype(name: str):
    table = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _build_lr_schedule(base_lr: float, min_lr_ratio: float, warmup_steps: int, max_steps: int):
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


def _grad_norm_array(tree: Any) -> mx.array:
    sq = mx.array(0.0, dtype=mx.float32)
    for leaf in _tree_leaves(tree):
        if isinstance(leaf, mx.array):
            x = leaf.astype(mx.float32)
            sq = sq + (x * x).sum()
    return mx.sqrt(sq + 1e-12)


def _clip_grads(tree: Any, max_norm: float):
    if max_norm <= 0:
        return tree, 0.0
    norm_arr = _grad_norm_array(tree)
    mx.eval(norm_arr)
    norm = float(norm_arr.item())
    if norm <= max_norm:
        return tree, norm
    scale = max_norm / (norm + 1e-6)
    return _tree_scale(tree, scale), norm


def _clip_grads_for_compile(tree: Any, max_norm: float):
    if max_norm <= 0:
        return tree, mx.array(0.0, dtype=mx.float32)
    norm = _grad_norm_array(tree)
    limit = mx.array(max_norm, dtype=mx.float32)
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), limit / (norm + 1e-6))
    return _tree_scale(tree, scale), norm


def _build_local_compiled_train_step(
    *,
    model: TransformerLM,
    optimizer: optim.AdamW,
    grad_accum: int,
    grad_clip: float,
    ignore_index: int,
):
    def loss_fn(x, y):
        return model(x, targets=y, ignore_index=ignore_index)["loss"]

    step_and_grad = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=[model.state, optimizer.state], outputs=[model.state, optimizer.state])
    def compiled_train_step(xs, ys, lr):
        total_loss = mx.array(0.0, dtype=mx.float32)
        grads_acc = None
        for micro in range(grad_accum):
            loss, grads = step_and_grad(xs[micro], ys[micro])
            total_loss = total_loss + loss.astype(mx.float32)
            grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)
        grads_acc = _tree_scale(grads_acc, 1.0 / float(grad_accum))
        grads_acc, grad_norm = _clip_grads_for_compile(grads_acc, grad_clip)
        optimizer.learning_rate = lr
        optimizer.update(model, grads_acc)
        return total_loss / float(grad_accum), grad_norm

    return compiled_train_step


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _coalesce_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    if "config" in raw and isinstance(raw["config"], dict):
        raw = raw["config"]
    return raw


def _load_model_config(base_ckpt: str, model_config: Optional[str], seq_len_override: int | None) -> TransformerConfig:
    raw = None
    if model_config:
        raw = _coalesce_cfg(_load_json(model_config))
    else:
        meta_path = base_ckpt + ".json"
        if os.path.exists(meta_path):
            raw = _coalesce_cfg(_load_json(meta_path))
    if raw is None:
        raise FileNotFoundError(
            "Need a model config. Pass --model-config or provide a checkpoint with <ckpt>.json metadata."
        )

    cfg = TransformerConfig(
        vocab_size=int(raw["vocab_size"]),
        max_seq_len=int(raw.get("max_seq_len", raw.get("max_seq", 1024))),
        d_model=int(raw["d_model"]),
        n_heads=int(raw["n_heads"]),
        n_layers=int(raw["n_layers"]),
        mlp_ratio=float(raw.get("mlp_ratio", 4.0)),
        attention_impl=str(raw.get("attention_impl", "fast")),
    )
    if seq_len_override:
        cfg.max_seq_len = min(cfg.max_seq_len, int(seq_len_override))
    return cfg


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
        raw = ex.get("messages")
    elif isinstance(ex.get("conversations"), list):
        raw = ex.get("conversations")
    if raw is None:
        return None
    msgs: List[Tuple[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from") or m.get("speaker")
        content = m.get("content") or m.get("value") or m.get("text") or m.get("message")
        if not role or not content:
            continue
        rr = _norm_role(str(role))
        if rr is None:
            continue
        cc = str(content).strip()
        if not cc:
            continue
        msgs.append((rr, cc))
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
    for uk, ak in [
        ("prompt", "completion"),
        ("prompt", "response"),
        ("question", "answer"),
        ("query", "response"),
        ("input", "output"),
    ]:
        u = ex.get(uk)
        a = ex.get(ak)
        if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
            return u.strip(), a.strip()
    return None


def _choose_user_assistant_pair(messages: List[Tuple[str, str]], strategy: str) -> Optional[Tuple[str, str]]:
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


def _extract_pair(ex: Dict[str, Any], pair_strategy: str) -> Optional[Tuple[str, str]]:
    msgs = _extract_messages(ex)
    if msgs:
        pair = _choose_user_assistant_pair(msgs, strategy=pair_strategy)
        if pair is not None:
            return pair
    return _extract_pair_from_fields(ex)


def _pair_stream(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
) -> Iterator[Tuple[str, str]]:
    while True:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
        )
        if shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        for ex in ds:
            pair = _extract_pair(ex, pair_strategy=pair_strategy)
            if pair is not None:
                yield pair


def _mix_pair_iterator(
    *,
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
) -> Iterator[Tuple[str, str]]:
    tulu = _pair_stream(
        dataset_name="allenai/tulu-v2-sft-mixture",
        dataset_config=None,
        split="train",
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        trust_remote_code=trust_remote_code,
        pair_strategy=pair_strategy,
    )
    ultra = _pair_stream(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        dataset_config=None,
        split="train_sft",
        shuffle_buffer=shuffle_buffer,
        seed=seed + 1,
        trust_remote_code=trust_remote_code,
        pair_strategy=pair_strategy,
    )
    gens = [tulu, ultra]
    weights = [0.8, 0.2]
    while True:
        g = random.choices(gens, weights=weights, k=1)[0]
        yield next(g)


def _local_jsonl_pair_iterator(path: str, pair_strategy: str, seed: int) -> Iterator[Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    rows: List[Tuple[str, str]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            pair = _extract_pair(ex, pair_strategy=pair_strategy)
            if pair is not None:
                rows.append(pair)
    if not rows:
        raise RuntimeError(f"No valid user/assistant pairs found in {p}")
    rng = random.Random(seed)
    while True:
        rng.shuffle(rows)
        for pair in rows:
            yield pair


def _build_pair_iterator(
    *,
    recipe: str,
    train_jsonl: str,
    shuffle_buffer: int,
    seed: int,
    trust_remote_code: bool,
    pair_strategy: str,
) -> Iterator[Tuple[str, str]]:
    if train_jsonl:
        return _local_jsonl_pair_iterator(train_jsonl, pair_strategy=pair_strategy, seed=seed)
    recipe = recipe.strip().lower()
    if recipe == "tulu_v2":
        return _pair_stream(
            dataset_name="allenai/tulu-v2-sft-mixture",
            dataset_config=None,
            split="train",
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            trust_remote_code=trust_remote_code,
            pair_strategy=pair_strategy,
        )
    if recipe == "ultrachat":
        return _pair_stream(
            dataset_name="HuggingFaceH4/ultrachat_200k",
            dataset_config=None,
            split="train_sft",
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            trust_remote_code=trust_remote_code,
            pair_strategy=pair_strategy,
        )
    if recipe == "mix":
        return _mix_pair_iterator(
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            trust_remote_code=trust_remote_code,
            pair_strategy=pair_strategy,
        )
    raise ValueError(f"Unknown recipe: {recipe}")


def _build_eval_pair_iterator(*, val_jsonl: str, pair_strategy: str, seed: int):
    if not val_jsonl:
        return None
    return _local_jsonl_pair_iterator(val_jsonl, pair_strategy=pair_strategy, seed=seed)


def _build_xy_assistant_only(
    user_text: str,
    assistant_text: str,
    *,
    sp: spm.SentencePieceProcessor,
    seq_len: int,
    pad_id: int,
    ignore_index: int,
    max_user_chars: int = 0,
    max_assistant_chars: int = 0,
    min_assistant_tokens: int = 64,
) -> Optional[Tuple[List[int], List[int]]]:
    if not user_text or not assistant_text:
        return None
    user = user_text.strip()
    assistant = assistant_text.strip()
    if not user or not assistant:
        return None
    if max_user_chars and len(user) > max_user_chars:
        user = user[:max_user_chars]
    if max_assistant_chars and len(assistant) > max_assistant_chars:
        assistant = assistant[:max_assistant_chars]

    min_assistant_tokens = max(1, int(min_assistant_tokens))
    prefix_budget = max(1, int(seq_len) - min_assistant_tokens)

    def enc(txt: str) -> List[int]:
        return sp.encode(txt, out_type=int)

    prefix = f"User: {user}\nAssistant: "
    prefix_ids = enc(prefix)
    while len(prefix_ids) > prefix_budget and len(user) > 1:
        new_len = max(1, int(len(user) * 0.9))
        if new_len >= len(user):
            new_len = len(user) - 1
        user = user[:new_len]
        prefix = f"User: {user}\nAssistant: "
        prefix_ids = enc(prefix)
    if len(prefix_ids) > prefix_budget:
        return None

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    ids = []
    if bos_id >= 0:
        ids.append(bos_id)
    ids.extend(enc(prefix + assistant))
    if eos_id >= 0:
        ids.append(eos_id)
    ids = ids[: seq_len + 1]

    boundary = (1 if bos_id >= 0 else 0) + len(prefix_ids)
    if boundary >= len(ids):
        return None

    x_ids = ids[:-1]
    y_ids = ids[1:]
    mask_upto = min(max(0, boundary - (1 if bos_id >= 0 else 0)), len(y_ids))
    if mask_upto:
        y_ids[:mask_upto] = [ignore_index] * mask_upto
    if all(t == ignore_index for t in y_ids):
        return None

    if len(x_ids) < seq_len:
        pad = seq_len - len(x_ids)
        x_ids = x_ids + [pad_id] * pad
        y_ids = y_ids + [ignore_index] * pad
    return x_ids, y_ids


def _make_batch_iterator(
    pair_iter: Iterator[Tuple[str, str]],
    *,
    sp: spm.SentencePieceProcessor,
    seq_len: int,
    batch_size: int,
    pad_id: int,
    ignore_index: int,
    max_user_chars: int,
    max_assistant_chars: int,
    min_assistant_tokens: int,
):
    while True:
        X = np.full((batch_size, seq_len), pad_id, dtype=np.int32)
        Y = np.full((batch_size, seq_len), ignore_index, dtype=np.int32)
        filled = 0
        while filled < batch_size:
            user, assistant = next(pair_iter)
            xy = _build_xy_assistant_only(
                user,
                assistant,
                sp=sp,
                seq_len=seq_len,
                pad_id=pad_id,
                ignore_index=ignore_index,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                min_assistant_tokens=min_assistant_tokens,
            )
            if xy is None:
                continue
            x_ids, y_ids = xy
            X[filled] = np.asarray(x_ids, dtype=np.int32)
            Y[filled] = np.asarray(y_ids, dtype=np.int32)
            filled += 1
        yield mx.array(X, dtype=mx.int32), mx.array(Y, dtype=mx.int32)


def _estimate_loss(
    model: TransformerLM,
    batch_iter,
    eval_steps: int,
    ignore_index: int,
) -> float:
    vals = []
    for _ in range(eval_steps):
        x, y = next(batch_iter)
        loss = model(x, targets=y, ignore_index=ignore_index)["loss"]
        mx.eval(loss)
        vals.append(float(loss.item()))
    return float(sum(vals) / max(1, len(vals)))


def _sample_next_id(logits: mx.array, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    if temperature <= 0.0 or top_k <= 1:
        mx.eval(logits)
        return int(mx.argmax(logits, axis=-1).item())
    scaled = logits / max(temperature, 1e-6)
    if top_k > 0:
        vals, idx = mx.topk(scaled, k=min(top_k, scaled.shape[-1]), axis=-1)
        mx.eval(vals, idx)
        probs = mx.softmax(vals.astype(mx.float32), axis=-1)
        mx.eval(probs)
        probs_np = np.asarray(probs[0])
        idx_np = np.asarray(idx[0], dtype=np.int64)
        choice = rng.choice(len(idx_np), p=probs_np / probs_np.sum())
        return int(idx_np[choice])
    probs = mx.softmax(scaled.astype(mx.float32), axis=-1)
    mx.eval(probs)
    probs_np = np.asarray(probs[0])
    return int(rng.choice(len(probs_np), p=probs_np / probs_np.sum()))


def _generate_sample_text(
    model: TransformerLM,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    if prompt.startswith("User:"):
        full_prompt = prompt.rstrip()
        if not full_prompt.endswith("Assistant:"):
            full_prompt = full_prompt + "\nAssistant:"
    else:
        full_prompt = f"User: {prompt.strip()}\nAssistant:"

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    ids: List[int] = []
    if bos_id >= 0:
        ids.append(bos_id)
    ids.extend(sp.encode(full_prompt, out_type=int))

    caches = None
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        if len(ids) <= max_seq_len:
            x = mx.array([ids], dtype=mx.int32)
            logits, _ = model.logits(x, caches=None)
            next_logits = logits[:, -1, :]
        else:
            token = mx.array([[ids[-1]]], dtype=mx.int32)
            next_logits, caches = model.step(token, caches=caches)
        next_id = _sample_next_id(next_logits, temperature=temperature, top_k=top_k, rng=rng)
        ids.append(next_id)
        if eos_id >= 0 and next_id == eos_id:
            break
    core = [t for t in ids if t not in {tid for tid in [bos_id, eos_id] if tid >= 0}]
    return sp.decode(core)


def _load_config_defaults(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must be an object.")
    return {k.replace("-", "_"): v for k, v in raw.items()}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre], description="Chat-style SFT trainer on top of train/model.py")
    parser.add_argument("--base-ckpt", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--spm-model", type=str, default="/Users/williamzebrowski/sml-mlx/tokenizer/fineweb_spm/spm.model")
    parser.add_argument("--recipe", type=str, default="mix")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument("--pair-strategy", type=str, default="random")
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="/Users/williamzebrowski/sml-mlx/train/checkpoints/chat_sft_v1")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-user-chars", type=int, default=0)
    parser.add_argument("--max-assistant-chars", type=int, default=0)
    parser.add_argument("--min-assistant-tokens", type=int, default=64)
    parser.add_argument("--sample-prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--sample-max-new-tokens", type=int, default=80)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.set_defaults(compile_train_step=True)
    parser.add_argument("--compile-train-step", action="store_true", dest="compile_train_step")
    parser.add_argument("--no-compile-train-step", action="store_false", dest="compile_train_step")

    if pre_args.config:
        cfg_path = Path(pre_args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        parser.set_defaults(**_load_config_defaults(str(cfg_path)))

    args = parser.parse_args()

    base_ckpt = args.resume or args.base_ckpt
    if not base_ckpt:
        raise ValueError("Need --base-ckpt (or --resume).")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {base_ckpt}")

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pad_id = sp.pad_id()
    if pad_id is None or pad_id < 0:
        pad_id = 0

    cfg = _load_model_config(
        base_ckpt=base_ckpt,
        model_config=args.model_config or None,
        seq_len_override=int(args.seq_len) if args.seq_len else None,
    )
    model = TransformerLM(cfg)
    _cast_model_floats(model, _resolve_dtype(args.dtype))
    mx.eval(model.parameters())

    if not _load_checkpoint(base_ckpt, model):
        raise RuntimeError(f"Failed to load checkpoint: {base_ckpt}")
    mx.eval(model.parameters())

    optimizer = optim.AdamW(args.lr, weight_decay=args.weight_decay)
    train_iter = _build_pair_iterator(
        recipe=args.recipe,
        train_jsonl=args.train_jsonl,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        pair_strategy=args.pair_strategy,
    )
    val_pair_iter = _build_eval_pair_iterator(
        val_jsonl=args.val_jsonl,
        pair_strategy=args.pair_strategy,
        seed=args.seed + 999,
    )

    train_batch_iter = _make_batch_iterator(
        train_iter,
        sp=sp,
        seq_len=cfg.max_seq_len,
        batch_size=args.batch_size,
        pad_id=pad_id,
        ignore_index=args.ignore_index,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
        min_assistant_tokens=args.min_assistant_tokens,
    )
    val_batch_iter = (
        _make_batch_iterator(
            val_pair_iter,
            sp=sp,
            seq_len=cfg.max_seq_len,
            batch_size=args.batch_size,
            pad_id=pad_id,
            ignore_index=args.ignore_index,
            max_user_chars=args.max_user_chars,
            max_assistant_chars=args.max_assistant_chars,
            min_assistant_tokens=args.min_assistant_tokens,
        )
        if val_pair_iter is not None
        else None
    )

    lr_for_step = _build_lr_schedule(
        base_lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    def loss_fn(x, y):
        return model(x, targets=y, ignore_index=args.ignore_index)["loss"]

    step_and_grad = nn.value_and_grad(model, loss_fn)
    use_compiled_local_step = bool(args.compile_train_step)
    compiled_local_step = None
    if use_compiled_local_step:
        compiled_local_step = _build_local_compiled_train_step(
            model=model,
            optimizer=optimizer,
            grad_accum=int(args.grad_accum),
            grad_clip=float(args.grad_clip),
            ignore_index=int(args.ignore_index),
        )

    start_step = _infer_resume_step(args.resume) if args.resume else 0
    if start_step > 0:
        print(f"[resume] path={args.resume} start_step={start_step}", flush=True)

    num_params = count_parameters(model)
    dtype_bytes = 2 if _resolve_dtype(args.dtype) in (mx.float16, mx.bfloat16) else 4
    param_gib = (num_params * dtype_bytes) / (1024**3)
    print(
        f"[model] params={num_params/1e6:.2f}M dtype={args.dtype} approx_param_mem={param_gib:.2f} GiB",
        flush=True,
    )
    print(
        f"[sft] seq={cfg.max_seq_len} batch={args.batch_size} accum={args.grad_accum} "
        f"lr={args.lr:.2e} recipe={args.recipe if not args.train_jsonl else args.train_jsonl}",
        flush=True,
    )
    print(
        f"[compile] train_step={'enabled' if use_compiled_local_step else 'disabled'} (local single-host path)",
        flush=True,
    )

    ema_loss = None
    loop_start = time.perf_counter()

    for step in range(start_step, args.max_steps):
        t0 = time.perf_counter()
        lr_t = lr_for_step(step)
        if use_compiled_local_step:
            micro_x = []
            micro_y = []
            for _ in range(args.grad_accum):
                x, y = next(train_batch_iter)
                micro_x.append(x)
                micro_y.append(y)
            x_batch = mx.stack(micro_x, axis=0)
            y_batch = mx.stack(micro_y, axis=0)
            lr_arr = mx.array(lr_t, dtype=mx.float32)
            step_loss, grad_norm_arr = compiled_local_step(x_batch, y_batch, lr_arr)
            mx.eval(step_loss, grad_norm_arr, model.state, optimizer.state)
            step_loss_value = float(step_loss.item())
            grad_norm = float(grad_norm_arr.item())
        else:
            total_loss_local = 0.0
            grads_acc = None
            for _ in range(args.grad_accum):
                x, y = next(train_batch_iter)
                loss, grads = step_and_grad(x, y)
                mx.eval(loss)
                total_loss_local += float(loss.item())
                grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)
            grads_acc = _tree_scale(grads_acc, 1.0 / float(args.grad_accum))
            grads_acc, grad_norm = _clip_grads(grads_acc, args.grad_clip)
            optimizer.learning_rate = lr_t
            optimizer.update(model, grads_acc)
            mx.eval(model.parameters(), optimizer.state)
            step_loss_value = total_loss_local / float(args.grad_accum)

        if not math.isfinite(step_loss_value) or not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite at step {step+1}: loss={step_loss_value} grad_norm={grad_norm}")

        ema_loss = step_loss_value if ema_loss is None else (0.98 * ema_loss + 0.02 * step_loss_value)
        dt = time.perf_counter() - t0
        toks_per_sec = (args.batch_size * args.grad_accum * cfg.max_seq_len) / max(dt, 1e-9)

        if ((step + 1) % args.log_every == 0) or step == start_step:
            print(
                f"[step {step+1:6d}] loss={step_loss_value:.4f} ema={ema_loss:.4f} "
                f"lr={lr_t:.3e} grad_norm={grad_norm:.3f} tok/s={toks_per_sec:,.0f}",
                flush=True,
            )

        if val_batch_iter is not None and args.eval_every > 0 and ((step + 1) % args.eval_every == 0):
            val_loss = _estimate_loss(
                model=model,
                batch_iter=val_batch_iter,
                eval_steps=args.eval_steps,
                ignore_index=args.ignore_index,
            )
            val_ppl = math.exp(min(20.0, val_loss))
            print(f"[eval {step+1:6d}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}", flush=True)

        if args.save_every > 0 and ((step + 1) % args.save_every == 0):
            ckpt_path = os.path.join(args.save_dir, f"step_{step+1:07d}.safetensors")
            metadata = {
                "step": step + 1,
                "args": vars(args),
                "config": {
                    "vocab_size": cfg.vocab_size,
                    "max_seq_len": cfg.max_seq_len,
                    "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads,
                    "n_layers": cfg.n_layers,
                    "mlp_ratio": cfg.mlp_ratio,
                    "attention_impl": cfg.attention_impl,
                },
                "timestamp": time.time(),
            }
            _save_checkpoint(ckpt_path, model, metadata)
            print(f"[ckpt] saved {ckpt_path}", flush=True)
            if args.sample_max_new_tokens > 0:
                sample_text = _generate_sample_text(
                    model=model,
                    sp=sp,
                    prompt=args.sample_prompt,
                    max_seq_len=cfg.max_seq_len,
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temperature,
                    top_k=args.sample_top_k,
                    seed=args.seed + step + 1,
                )
                print(f"[sample {step+1:6d}] prompt={args.sample_prompt!r}", flush=True)
                print(sample_text, flush=True)

    final_path = os.path.join(args.save_dir, "final.safetensors")
    metadata = {
        "step": args.max_steps,
        "args": vars(args),
        "config": {
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "mlp_ratio": cfg.mlp_ratio,
            "attention_impl": cfg.attention_impl,
        },
        "duration_sec": time.perf_counter() - loop_start,
        "timestamp": time.time(),
    }
    _save_checkpoint(final_path, model, metadata)
    print(f"[done] saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
