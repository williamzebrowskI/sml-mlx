#!/usr/bin/env python
"""
Fine-tune TinyGPLM on GooAQ Q/A pairs using MLX, in User/Assistant chat format.

Intended as a SECOND SFT STAGE after Smol-SmolTalk, e.g.:

    Base pretrain   : FineWeb -> ckpt_1000000.safetensors
    Chat SFT        : Smol-SmolTalk -> smolsmoltalk_sft_005000.safetensors
    GooAQ QA SFT    : this script, starting from smolsmoltalk_sft_005000

Usage example (single Mac):

PYTHONPATH=. python -m model.finetune_gooaq \
  --ckpt model/checkpoints_smolsmoltalk_sft/smolsmoltalk_sft_005000.safetensors \
  --spm-model tokenizer/fineweb_spm/spm.model \
  --max-steps 5000 \
  --seq-len 3072 \
  --batch-size 1 \
  --lr 1e-5 \
  --save-dir model/checkpoints_gooaq_sft
"""

import argparse
import os
import time
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


# ---------- GooAQ streaming iterator ----------

def _is_bad_answer(a: str) -> bool:
    """
    Very simple filter to skip obviously 'array-like' answers, which are
    literal string representations of Python lists in some GooAQ rows.
    """
    if not a:
        return True
    s = a.strip()
    # Skip answers that look like "['foo', 'bar']"
    if s.startswith("[") and s.endswith("]"):
        return True
    return False


def gooaq_text_iterator(split: str = "train"):
    """
    Stream GooAQ Q/A pairs from Hugging Face and yield {"text": "..."} entries.

    Dataset: allenai/gooaq

    Fields (from the dataset card):
      - question: str
      - answer: str (may be None or an array-like string)

    We convert each example into a User/Assistant transcript:

      User: {question}
      Assistant: {answer}

    and train the LM on that text with next-token prediction.
    """
    ds = load_dataset(
        "allenai/gooaq",
        split=split,
        streaming=True,
        trust_remote_code=True,  # avoid interactive prompt
    )

    for ex in ds:
        q = ex.get("question", None)
        a = ex.get("answer", None)

        if not q or not a:
            continue

        # Basic filter for bad / array-like answers
        if _is_bad_answer(a):
            continue

        text = f"User: {q}\nAssistant: {a}"
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

def finetune_gooaq(
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
):
    # 1. Initialize tokenizer & vocab
    set_tokenizer(spm_model)
    vocab_size = mdl.TOK.vocab_size

    # 2. Build model config & instance
    cfg = TinyGPConfig(
        vocab_size=vocab_size,
        d_model=384,
        n_heads=6,
        n_layers=12,
        max_seq=seq_len,
        max_grad_norm=1.0,
    )
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    print(f"[ft-gooaq] model initialized (vocab={vocab_size}, d_model={cfg.d_model}, layers={cfg.n_layers})")

    # 3. Load base checkpoint (ideally a Smol-SmolTalk SFT checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ok = load_safetensors_model(ckpt_path, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}")
    mx.eval(model.parameters())
    print(f"[ft-gooaq] loaded base checkpoint from {ckpt_path}")

    # 4. Optimizer
    opt = optim.AdamW(lr, weight_decay=wd)

    # 5. Loss + grad fn (no compile for simplicity/robustness)
    def loss_fn(x, y):
        return model(x, y)["loss"]

    step_fn = nn.value_and_grad(model, loss_fn)

    # 6. Data pipeline
    sample_iter = gooaq_text_iterator(split=split)
    batch_iter = make_batch_iterator(sample_iter, seq_len=seq_len, batch_size=batch_size)

    update_step = 0
    last_log_time = time.time()

    for X, Y in batch_iter:
        if update_step >= max_steps:
            break

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
            out_path = os.path.join(save_dir, f"gooaq_sft_{update_step:06d}.safetensors")
            ok = save_safetensors_model(out_path, model)
            if ok:
                print(f"[{update_step}] saved fine-tuned checkpoint to {out_path}")
            else:
                print(f"[{update_step}] ⚠️ failed to save checkpoint to {out_path}")

    # Final checkpoint
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "gooaq_sft_final.safetensors")
    ok = save_safetensors_model(final_path, model)
    if ok:
        print(f"[ft-gooaq] saved final fine-tuned checkpoint to {final_path}")
    else:
        print(f"[ft-gooaq] ⚠️ failed to save final checkpoint to {final_path}")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser("Fine-tune TinyGPLM on GooAQ (User/Assistant QA) with MLX")
    p.add_argument("--ckpt", type=str, required=True, help="Base .safetensors checkpoint to fine-tune (e.g. Smol-SmolTalk SFT)")
    p.add_argument("--spm-model", type=str, required=True, help="Path to SentencePiece model (spm.model)")

    p.add_argument("--max-steps", type=int, default=5000, help="Number of update steps for GooAQ SFT")
    p.add_argument("--seq-len", type=int, default=3072, help="Sequence length for fine-tuning")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate for AdamW")
    p.add_argument("--wd", type=float, default=0.01, help="Weight decay for AdamW")

    p.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    p.add_argument("--save-dir", type=str, default="model/checkpoints_gooaq_sft", help="Where to store GooAQ SFT checkpoints")
    p.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps (0 = only final)")
    p.add_argument("--split", type=str, default="train", help="Which GooAQ split to use (train/validation/test)")

    args = p.parse_args()

    finetune_gooaq(
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
    )


if __name__ == "__main__":
    main()