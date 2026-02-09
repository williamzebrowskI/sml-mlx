#!/usr/bin/env python
"""
Fine-tune TinyGPLM on a mixture of Task SFT datasets (MMLU, GSM8K, ARC) using MLX.

Datasets:
  - cais/mmlu          (multiple-choice factual QA)
  - openai/gsm8k       (grade-school math word problems with reasoning)
  - allenai/ai2_arc    (ARC-Easy + ARC-Challenge science MCQ)

This script:
  - Loads a base checkpoint (e.g., your SmolSmolTalk SFT 100M model).
  - Streams and mixes the three datasets on-the-fly (no concatenation needed).
  - Formats all tasks into a unified "User: ...\\nAssistant: ..." prompt style.
  - Runs standard next-token prediction SFT with optional LR scheduler.

Example usage (single Mac, 100M config):

    PYTHONPATH=. python -m model.fine_tune_tasks \\
      --ckpt model/checkpoints_smolsmoltalk_sft_100m/smolsmoltalk_sft_020000.safetensors \\
      --spm-model tokenizer/fineweb_spm/spm.model \\
      --config configs/config_mlx_slm_beta_v2_100m.json \\
      --max-steps 50000 \\
      --seq-len 3072 \\
      --batch-size 1 \\
      --lr 5e-5 \\
      --wd 0.01 \\
      --log-every 50 \\
      --save-dir model/checkpoints_tasks_sft_100m \\
      --save-every 5000 \\
      --lr-decay-start 20000 \\
      --min-lr-factor 0.1
"""

import argparse
import os
import time
import importlib
import json
import math
import random
from typing import Dict, Iterator, Optional, List

from datasets import load_dataset

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Import model module; keep handle so globals like TOK update after set_tokenizer.
try:
    mdl = importlib.import_module("model.model")
except ImportError:
    mdl = importlib.import_module("model")

TinyGPLM = mdl.TinyGPLM
TinyGPConfig = mdl.TinyGPConfig
set_tokenizer = mdl.set_tokenizer
load_safetensors_model = mdl.load_safetensors_model
save_safetensors_model = mdl.save_safetensors_model


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def format_mmlu_example(ex: Dict) -> Optional[str]:
    """
    MMLU (cais/mmlu) expected fields:
      - subject: str
      - question: str
      - choices: List[str]
      - answer: int or str (index or letter)

    We format as a MCQ:

    User: [MMLU][subject=<subject>] <question>
    Choices:
    A. ...
    B. ...
    C. ...
    D. ...
    Assistant: The correct answer is <Letter>.
    """
    question = ex.get("question", None)
    choices = ex.get("choices") or ex.get("options")  # some variants use 'options'
    answer = ex.get("answer", None)
    subject = ex.get("subject", "unknown")

    if not question or not choices or answer is None:
        return None

    # Normalize answer to a letter A/B/C/D...
    if isinstance(answer, int):
        idx = answer
    else:
        # answer might be 'A','B','C','D' etc.
        if isinstance(answer, str) and answer.strip().upper() in "ABCD":
            idx = "ABCD".index(answer.strip().upper())
        else:
            # If we can't parse, bail out on this example
            return None

    if idx < 0 or idx >= len(choices):
        return None

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    lines.append(f"User: [MMLU][subject={subject}] {question}")
    lines.append("Choices:")
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}. {ch}")
    correct_letter = letters[idx]
    lines.append(f"Assistant: The correct answer is {correct_letter}.")
    return "\n".join(lines)


def format_gsm8k_example(ex: Dict) -> Optional[str]:
    """
    GSM8K (openai/gsm8k) expected fields:
      - question: str
      - answer: str (includes reasoning and final answer)

    We format as:

    User: [GSM8K] <question>
    Assistant: <answer>
    """
    q = ex.get("question", None)
    a = ex.get("answer", None)
    if not q or not a:
        return None
    return f"User: [GSM8K] {q}\nAssistant: {a}"


def format_arc_example(ex: Dict, difficulty_tag: str) -> Optional[str]:
    """
    ARC (allenai/ai2_arc) expected fields:
      - question: str
      - answerKey: str (e.g. 'A','B','C','D')
      - choices: { 'text': [...], 'label': [...] }

    We format as:

    User: [ARC-<difficulty>] <question>
    Choices:
    A. ...
    B. ...
    C. ...
    D. ...
    Assistant: The correct answer is <answerKey>.
    """
    question = ex.get("question", None)
    answer_key = ex.get("answerKey", None)
    choices = ex.get("choices", None)

    if not question or not answer_key or not choices:
        return None

    texts = choices.get("text", None)
    labels = choices.get("label", None)
    if not texts or not labels or len(texts) != len(labels):
        return None

    # We want to order options by label A,B,C,D,... if possible
    paired = list(zip(labels, texts))
    try:
        paired.sort(key=lambda x: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(x[0]))
    except ValueError:
        # Some weird label, just keep original order
        pass

    lines = []
    lines.append(f"User: [ARC-{difficulty_tag}] {question}")
    lines.append("Choices:")
    for lab, txt in paired:
        lines.append(f"{lab}. {txt}")
    lines.append(f"Assistant: The correct answer is {answer_key}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Streaming iterators for each dataset (infinite loops)
# ---------------------------------------------------------------------

def mmlu_stream(split: str = "test") -> Iterator[Dict[str, str]]:
    """
    Infinite generator over MMLU examples.

    NOTE: We use split='test' because MMLU is typically evaluated on 'test',
    but for SFT it's fine to train on this too for a toy model / research
    purposes. Adjust if you want 'train' or another split.
    """
    while True:
        ds = load_dataset("cais/mmlu", "all", split=split, streaming=True)
        for ex in ds:
            text = format_mmlu_example(ex)
            if text:
                yield {"text": text}


def gsm8k_stream(split: str = "train") -> Iterator[Dict[str, str]]:
    """
    Infinite generator over GSM8K examples.

    openai/gsm8k main config; typically use split='train' for SFT.
    """
    while True:
        ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
        for ex in ds:
            text = format_gsm8k_example(ex)
            if text:
                yield {"text": text}


def arc_stream(split: str = "train") -> Iterator[Dict[str, str]]:
    """
    Infinite generator over ARC-Easy + ARC-Challenge examples.

    Each yields either [ARC-Easy] or [ARC-Challenge] formatted text.
    """
    # Easy
    while True:
        ds_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, streaming=True)
        ds_chal = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, streaming=True)

        for ex in ds_easy:
            text = format_arc_example(ex, difficulty_tag="Easy")
            if text:
                yield {"text": text}

        for ex in ds_chal:
            text = format_arc_example(ex, difficulty_tag="Challenge")
            if text:
                yield {"text": text}


# ---------------------------------------------------------------------
# Mixed task iterator (no concatenation, streaming mixture)
# ---------------------------------------------------------------------

def task_text_iterator(split: str = "train") -> Iterator[Dict[str, str]]:
    """
    Stream a mixture of MMLU, GSM8K, and ARC examples.

    We do this by maintaining 3 infinite generators and, at each step,
    randomly picking one to draw from.

    No need to concatenate anything on disk; everything is streamed.
    """
    mmlu_gen = mmlu_stream(split="test")      # typical MMLU split
    gsm_gen = gsm8k_stream(split="train")
    arc_gen = arc_stream(split="train")

    gens = [mmlu_gen, gsm_gen, arc_gen]
    while True:
        g = random.choice(gens)
        yield next(g)


# ---------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------

def make_batch_iterator(sample_iter: Iterator[Dict[str, str]], seq_len: int, batch_size: int):
    """
    Turn a streaming iterator of {"text": "..."} into (X, Y) batches of token IDs.

    X, Y shapes: (batch_size, seq_len), dtype int32.
    """
    X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    filled = 0

    for ex in sample_iter:
        text = ex.get("text", "")
        if not text:
            continue

        ids = mdl.TOK.encode(text)
        if len(ids) < 2:
            continue

        ids = ids[: seq_len + 1]
        x_ids = ids[:-1]
        y_ids = ids[1:]

        if len(x_ids) < seq_len:
            pad = seq_len - len(x_ids)
            x_ids = x_ids + [mdl.PAD_ID_RUNTIME] * pad
            y_ids = y_ids + [mdl.PAD_ID_RUNTIME] * pad

        X[filled] = mx.array(x_ids, dtype=mx.int32)
        Y[filled] = mx.array(y_ids, dtype=mx.int32)
        filled += 1

        if filled == batch_size:
            yield X, Y
            X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            filled = 0


# ---------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------

def finetune_tasks(
    ckpt_path: str,
    spm_model: str,
    max_steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    wd: float,
    log_every: int,
    save_dir: str,
    save_every: int,
    config_path: Optional[str] = None,
    lr_decay_start: Optional[int] = None,
    min_lr_factor: float = 0.1,
):
    # 1. Initialize tokenizer & vocab
    set_tokenizer(spm_model)
    vocab_size = mdl.TOK.vocab_size

    # 2. Build model config & instance (100M config via JSON, or fallback)
    cfg_kwargs = {
        "vocab_size": vocab_size,
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "max_seq": seq_len,
        "max_grad_norm": 1.0,
    }
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        for k in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq", "max_grad_norm"):
            if k in cfg_json:
                cfg_kwargs[k] = cfg_json[k]
        if seq_len:
            cfg_kwargs["max_seq"] = min(seq_len, cfg_kwargs.get("max_seq", seq_len))

    cfg = TinyGPConfig(**cfg_kwargs)
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    print(f"[ft-tasks] model initialized (vocab={vocab_size}, d_model={cfg.d_model}, layers={cfg.n_layers})")

    # 3. Load base checkpoint (e.g., SmolSmolTalk SFT 100M)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ok = load_safetensors_model(ckpt_path, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}")
    mx.eval(model.parameters())
    print(f"[ft-tasks] loaded base checkpoint from {ckpt_path}")

    # 4. Optimizer + Scheduler
    opt = optim.AdamW(lr, weight_decay=wd)
    base_lr = lr
    decay_start = lr_decay_start if lr_decay_start is not None else max_steps  # if None, no decay

    def get_lr(step: int) -> float:
        if step < decay_start or decay_start >= max_steps:
            return base_lr
        t = (step - decay_start) / max(1, max_steps - decay_start)
        t = min(max(t, 0.0), 1.0)
        # Cosine decay from 1.0 -> 0.0
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
        # Map [0,1] to [min_lr_factor, 1.0]
        factor = min_lr_factor + (1.0 - min_lr_factor) * cos_factor
        return base_lr * factor

    # 5. Loss + grad fn (no mx.compile for safety)
    def loss_fn(x, y):
        return model(x, y)["loss"]

    step_fn = nn.value_and_grad(model, loss_fn)

    # 6. Data pipeline (mixed task stream)
    sample_iter = task_text_iterator(split="train")
    batch_iter = make_batch_iterator(sample_iter, seq_len=seq_len, batch_size=batch_size)

    update_step = 0
    last_log_time = time.time()

    for X, Y in batch_iter:
        if update_step >= max_steps:
            break

        # Update LR according to schedule
        opt.learning_rate = get_lr(update_step)

        loss, grads = step_fn(X, Y)
        mx.eval(loss, grads)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        update_step += 1

        # Logging
        if update_step % log_every == 0:
            now = time.time()
            dt = now - last_log_time
            last_log_time = now
            toks = batch_size * seq_len
            toks_per_s = toks / max(1e-9, dt)
            ppl = float(mx.exp(loss).item()) if float(loss.item()) < 20 else float("inf")
            lr_cur = getattr(opt, "learning_rate", None)
            lr_str = f" lr={lr_cur:.2e}" if lr_cur is not None else ""
            print(
                f"[{update_step}] loss={loss.item():.4f} ppl={ppl:.2f}{lr_str} "
                f"tok/s≈{toks_per_s:.0f}"
            )

        # Periodic checkpoint
        if update_step > 0 and save_every > 0 and update_step % save_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"tasks_sft_{update_step:06d}.safetensors")
            ok = save_safetensors_model(out_path, model)
            if ok:
                print(f"[{update_step}] saved TASK SFT checkpoint to {out_path}")
            else:
                print(f"[{update_step}] ⚠️ failed to save checkpoint to {out_path}")

    # Final checkpoint
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "tasks_sft_final.safetensors")
    ok = save_safetensors_model(final_path, model)
    if ok:
        print(f"[ft-tasks] saved final TASK SFT checkpoint to {final_path}")
    else:
        print(f"[ft-tasks] ⚠️ failed to save final checkpoint to {final_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Fine-tune TinyGPLM on Task SFT mixture (MMLU + GSM8K + ARC) with MLX")
    p.add_argument("--ckpt", type=str, required=True, help="Base .safetensors checkpoint to fine-tune")
    p.add_argument("--spm-model", type=str, required=True, help="Path to SentencePiece model (spm.model)")
    p.add_argument("--config", type=str, default=None, help="Optional JSON config for the model (100M, etc.)")

    p.add_argument("--max-steps", type=int, default=50_000, help="Number of update steps for Task SFT")
    p.add_argument("--seq-len", type=int, default=3072, help="Sequence length for fine-tuning")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW")
    p.add_argument("--wd", type=float, default=0.01, help="Weight decay for AdamW")

    p.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    p.add_argument("--save-dir", type=str, default="model/checkpoints_tasks_sft", help="Where to store Task SFT checkpoints")
    p.add_argument("--save-every", type=int, default=5000, help="Save checkpoint every N steps (0 = only final)")

    p.add_argument(
        "--lr-decay-start",
        type=int,
        default=None,
        help="Step at which to start LR cosine decay (None = no decay)",
    )
    p.add_argument(
        "--min-lr-factor",
        type=float,
        default=0.1,
        help="Minimum LR as a fraction of base LR at the end of decay (e.g. 0.1 => min LR = 0.1 * base_lr)",
    )

    args = p.parse_args()

    finetune_tasks(
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        log_every=args.log_every,
        save_dir=args.save_dir,
        save_every=args.save_every,
        config_path=args.config,
        lr_decay_start=args.lr_decay_start,
        min_lr_factor=args.min_lr_factor,
    )


if __name__ == "__main__":
    main()