#!/usr/bin/env python
"""
Fine-tune TinyGPLM on Smol-SmolTalk (short user/assistant conversations) using MLX.

Dataset: HuggingFaceTB/smol-smoltalk

This script is kept as a dedicated SmolSmolTalk finetuner.
The general instruction/chat finetuner lives in `model/fine_tune.py`.

Usage example (single Mac):

PYTHONPATH=. python -m model.fine_tune_smolsmoltalk \
  --ckpt model/checkpoints_spm/ckpt_1000000.safetensors \
  --spm-model tokenizer/fineweb_spm/spm.model \
  --max-steps 50000 \
  --seq-len 2048 \
  --batch-size 1 \
  --lr 5e-5 \
  --save-dir model/checkpoints_smolsmoltalk_sft \
  --lr-decay-start 20000 \
  --min-lr-factor 0.1
"""

import argparse
import os
import time
import math
import importlib

from datasets import load_dataset

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Import from your existing model module.
# We keep a handle to the module so globals like TOK update after set_tokenizer.
try:
    mdl = importlib.import_module("model.model")
except ImportError:
    mdl = importlib.import_module("model")

TinyGPLM = mdl.TinyGPLM
TinyGPConfig = mdl.TinyGPConfig
set_tokenizer = mdl.set_tokenizer
load_safetensors_model = mdl.load_safetensors_model
save_safetensors_model = mdl.save_safetensors_model


# ---------- Smol-SmolTalk streaming iterator ----------

def smol_smoltalk_text_iterator(split: str = "train"):
    """
    Stream Smol-SmolTalk conversations and yield {"text": "..."} entries.

    Dataset: HuggingFaceTB/smol-smoltalk

    Each row has:
      - messages: list of { "content": str, "role": str }
      - source: str

    We flatten each conversation into a User/Assistant chat transcript like:

      User: ...
      Assistant: ...
      User: ...
      Assistant: ...

    and train the LM on that text with next-token prediction.
    """
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split, streaming=True)

    for ex in ds:
        messages = ex.get("messages", None)
        if not messages or not isinstance(messages, list):
            continue

        lines = []
        for turn in messages:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if not content:
                continue

            role_l = role.lower()
            if role_l == "user":
                prefix = "User"
            elif role_l == "assistant":
                prefix = "Assistant"
            else:
                # Unknown role; treat as user by default
                prefix = "User"

            lines.append(f"{prefix}: {content}")

        if not lines:
            continue

        text = "\n".join(lines)
        yield {"text": text}


# ---------- Batching ----------

def make_batch_iterator(sample_iter, seq_len: int, batch_size: int):
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

        # Standard LM training: next-token prediction
        ids = ids[: seq_len + 1]
        x_ids = ids[:-1]
        y_ids = ids[1:]

        # Pad to seq_len if needed
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


# ---------- Fine-tune loop ----------

def finetune_smol_smoltalk(
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
    split: str = "train",
    lr_decay_start: int = 0,
    min_lr_factor: float = 0.1,
    resume_ckpt: str | None = None,
    resume_step: int | None = None,
    skip_batches: int = 0,
):
    # 1. Initialize tokenizer & vocab
    set_tokenizer(spm_model)
    vocab_size = mdl.TOK.vocab_size

    # 2. Build model config & instance (100M-ish config)
    cfg = TinyGPConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        max_seq=seq_len,
        max_grad_norm=1.0,
    )
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    print(f"[ft] model initialized (vocab={vocab_size}, d_model={cfg.d_model}, layers={cfg.n_layers})")

    # 3. Load base or resume checkpoint
    ckpt_to_load = resume_ckpt if resume_ckpt else ckpt_path
    if not os.path.exists(ckpt_to_load):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_to_load}")
    ok = load_safetensors_model(ckpt_to_load, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_to_load}")
    mx.eval(model.parameters())
    print(f"[ft] loaded checkpoint from {ckpt_to_load}")

    # 4. Optimizer + LR scheduler state
    opt = optim.AdamW(lr, weight_decay=wd)
    base_lr = lr

    # 5. Loss + grad fn (no compile for simplicity/robustness)
    def loss_fn(x, y):
        return model(x, y)["loss"]

    step_fn = nn.value_and_grad(model, loss_fn)

    # 6. Data pipeline
    sample_iter = smol_smoltalk_text_iterator(split=split)
    batch_iter = make_batch_iterator(sample_iter, seq_len=seq_len, batch_size=batch_size)

    update_step = 0
    if resume_step is not None:
        update_step = max(0, int(resume_step))
    elif resume_ckpt:
        stem = os.path.splitext(os.path.basename(resume_ckpt))[0]
        parts = stem.split("_")
        if parts and parts[-1].isdigit():
            update_step = int(parts[-1])
    if update_step > 0:
        print(f"[ft] resuming at step {update_step}")

    last_log_time = time.time()

    # Optional fast-forward in the stream (best-effort; streaming dataset is non-deterministic)
    if skip_batches > 0:
        skipped = 0
        for _ in range(skip_batches):
            try:
                next(batch_iter)
                skipped += 1
            except StopIteration:
                break
        print(f"[ft] skipped {skipped} batches to approximate dataset position")

    for X, Y in batch_iter:
        if update_step >= max_steps:
            break

        loss, grads = step_fn(X, Y)
        mx.eval(loss, grads)

        # ---- Cosine LR decay for SFT (optional) ----
        if lr_decay_start > 0 and update_step >= lr_decay_start:
            # progress in [0, 1]
            t = (update_step - lr_decay_start) / max(1, max_steps - lr_decay_start)
            t = min(1.0, max(0.0, t))
            # cosine from 1 → min_lr_factor
            factor = min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + math.cos(math.pi * t))
            opt.learning_rate = base_lr * factor
        else:
            opt.learning_rate = base_lr
        # -------------------------------------------

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
            out_path = os.path.join(save_dir, f"smolsmoltalk_sft_{update_step:06d}.safetensors")
            ok = save_safetensors_model(out_path, model)
            if ok:
                print(f"[{update_step}] saved fine-tuned checkpoint to {out_path}")
            else:
                print(f"[{update_step}] ⚠️ failed to save checkpoint to {out_path}")

    # Final checkpoint
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "smolsmoltalk_sft_final.safetensors")
    ok = save_safetensors_model(final_path, model)
    if ok:
        print(f"[ft] saved final fine-tuned checkpoint to {final_path}")
    else:
        print(f"[ft] ⚠️ failed to save final checkpoint to {final_path}")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser("Fine-tune TinyGPLM on Smol-SmolTalk with MLX")
    p.add_argument("--ckpt", type=str, required=True, help="Base .safetensors checkpoint to fine-tune")
    p.add_argument("--spm-model", type=str, required=True, help="Path to SentencePiece model (spm.model)")

    p.add_argument("--max-steps", type=int, default=50_000, help="Number of update steps for SFT")
    p.add_argument("--seq-len", type=int, default=3072, help="Sequence length for fine-tuning")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW")
    p.add_argument("--wd", type=float, default=0.01, help="Weight decay for AdamW")

    p.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    p.add_argument("--save-dir", type=str, default="model/checkpoints_smolsmoltalk_sft", help="Where to store SFT checkpoints")
    p.add_argument("--save-every", type=int, default=5000, help="Save checkpoint every N steps (0 = only final)")
    p.add_argument("--split", type=str, default="train", help="Which Smol-SmolTalk split to use (train/test)")

    p.add_argument(
        "--lr-decay-start",
        type=int,
        default=0,
        help="Step at which to start cosine LR decay (0 = no decay).",
    )
    p.add_argument(
        "--min-lr-factor",
        type=float,
        default=0.1,
        help="Final LR as a fraction of initial LR (e.g. 0.1 -> decay to 10% of base LR).",
    )
    p.add_argument("--resume", type=str, default=None, help="Optional path to resume from an SFT checkpoint")
    p.add_argument("--resume-step", type=int, default=None, help="Optional step number to resume from")
    p.add_argument("--skip-batches", type=int, default=0, help="How many batches to skip from the stream on resume")

    args = p.parse_args()

    finetune_smol_smoltalk(
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
        split=args.split,
        lr_decay_start=args.lr_decay_start,
        min_lr_factor=args.min_lr_factor,
        resume_ckpt=args.resume,
        resume_step=args.resume_step,
        skip_batches=args.skip_batches,
    )


if __name__ == "__main__":
    main()
