# """
# Train a GPT-style LM with MLX only (no NumPy), with resume support and debug logs:

# • Loads a JSONL dataset of {"text": ...}
# • Tokenizes with your SentencePiece model
# • Packs into (context_size+1)-length samples: inputs → targets
# • Streams shuffled minibatches via MLX (mx.random)
# • Uses AdamW, cosine LR schedule, tqdm, and periodic checkpoints
# • Can resume from a safetensors checkpoint mid-run
# """

# import argparse
# import json
# import math
# import time
# import pathlib

# import datasets
# import mlx.core         as mx
# import mlx.nn           as nn
# import mlx.nn.losses    as losses
# import mlx.optimizers  as optim
# from tqdm              import tqdm

# from model.model       import Transformer, TransformerConfig
# from tokenizer         import sp_tokenizer


# def to_samples(ids: mx.array, context_size: int) -> mx.array:
#     """
#     Reshape a 1D array of token-ids into shape (N, context_size+1)
#     where each row is [x_0 ... x_{L-1}, x_1 ... x_L].
#     """
#     window = context_size + 1
#     n = ids.shape[0] // window
#     trimmed = ids[: n * window]
#     return trimmed.reshape(n, window)


# def iterate_batches(
#     samples: mx.array,
#     batch_size: int,
#     *,
#     seed: int = 0
# ):
#     """
#     Yield (batch_size, context+1) slices forever, shuffling each epoch.
#     """
#     N = samples.shape[0]
#     key = mx.random.key(seed)
#     perm = mx.random.permutation(N, key=key)
#     idx = 0
#     while True:
#         if idx + batch_size > N:
#             key, = mx.random.split(key, 1)
#             perm = mx.random.permutation(N, key=key)
#             idx = 0
#         batch_idx = perm[idx : idx + batch_size]
#         yield samples[batch_idx]
#         idx += batch_size


# def lr_schedule(step, *, base_lr, warmup, total_steps):
#     if step < warmup:
#         return base_lr * step / warmup
#     pct = (step - warmup) / (total_steps - warmup)
#     # cosine decays down to 0.1×
#     return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pct)))


# def get_args():
#     p = argparse.ArgumentParser("MLX GPT-LM trainer")
#     p.add_argument("--config",           required=True, help="Path to model/config.json")
#     p.add_argument("--dataset",          required=True, help="JSONL file, one {'text':...} per line")
#     p.add_argument("--tokenizer",        required=True, help="SentencePiece .model file")
#     p.add_argument("--out",              required=True, help="Directory to save checkpoints")
#     p.add_argument("--context_size",     type=int,   default=512, help="Sequence length")
#     p.add_argument("--batch_size",       type=int,   default=4,   help="Per-step batch size")
#     p.add_argument("--total_steps",      type=int,   default=8000, help="Total training iterations")
#     p.add_argument("--lr",               type=float, default=2e-4, help="Base learning rate")
#     p.add_argument("--warmup",           type=int,   default=5000, help="Linear warmup iters")
#     p.add_argument("--steps_report",     type=int,   default=100,  help="Log every N steps")
#     p.add_argument("--steps_checkpoint", type=int,   default=5000, help="Checkpoint every N steps")
#     p.add_argument("--resume",           type=str,   default=None, help="Path to safetensors checkpoint to resume from")
#     return p.parse_args()


# def main():
#     args = get_args()
#     print(f"[DEBUG] Parsed args: {args}")

#     # 1) load SentencePiece
#     print(f"[DEBUG] Loading tokenizer from {args.tokenizer}")
#     tok = sp_tokenizer.load(args.tokenizer)

#     # 2) load & tokenize dataset
#     print(f"[DEBUG] Loading JSONL dataset from {args.dataset}")
#     ds = datasets.load_dataset("json", data_files={"train": args.dataset}, split="train")
#     print(f"[DEBUG] Tokenizing...")
#     ds = ds.map(lambda ex: {"ids": tok.encode(ex["text"])}, remove_columns=["text"] )
#     print(f"[DEBUG] Tokenization done: {len(ds)} examples")

#     # 3) flatten all IDs → 1D array (single allocation)
#     t0 = time.perf_counter()

#     # (a) build a plain Python list of ints (no GPU allocations yet)
#     flat_ids = []
#     for ids in ds["ids"]:
#         flat_ids.extend(ids)            # fast list append

#     # (b) one bulk copy to MX
#     ids_arr = mx.array(flat_ids, dtype=mx.int32)

#     t1 = time.perf_counter()
#     print(f"[DEBUG] Flatten took {(t1 - t0):.2f}s; total tokens = {ids_arr.shape[0]}")

#     # 4) pack into samples of length context_size+1
#     print(f"[DEBUG] Packing into context+1 samples with context_size={args.context_size}")
#     samples = to_samples(ids_arr, args.context_size)
#     print(f"[DEBUG] Produced {samples.shape[0]} samples of length {args.context_size+1}")

#     # 5) build our MLX batch iterator
#     train_iter = iterate_batches(samples, args.batch_size, seed=42)
#     print(f"[DEBUG] Built batch iterator (batch_size={args.batch_size})")

#     # 6) load model config & initialize
#     print(f"[DEBUG] Instantiating model from config {args.config}")
#     cfg   = TransformerConfig(**json.load(open(args.config)))
#     cfg.context_size = args.context_size  
#     model = Transformer(cfg)
#     print(f"[DEBUG] Model instantiated.")

#     # 7) optimizer
#     optim_obj = optim.AdamW(learning_rate=args.lr, weight_decay=0.1)
#     print(f"[DEBUG] Optimizer (AdamW) created with lr={args.lr}")

#     # resume logic
#     start_step = 0
#     if args.resume:
#         print(f"[DEBUG] Resuming from checkpoint {args.resume}")
#         model.load_weights(args.resume)
#         # infer start step from filename: ckpt_XXXXX.safetensors
#         ck_name = pathlib.Path(args.resume).stem
#         try:
#             start_step = int(ck_name.split("_")[-1])
#         except:
#             print(f"[WARN] Could not parse step from {ck_name}, starting from step 0")
#         print(f"[DEBUG] Loaded weights at step {start_step}")

#     # 8) prepare loss+grad function
#     def loss_fn(mdl, batch):
#         x = batch[:, :-1]  # (B, L)
#         y = batch[:, 1:]   # (B, L)
#         logits = mdl(x)    # (B, L, V)
#         return losses.cross_entropy(
#             logits.reshape(-1, cfg.vocab_size),
#             y.reshape(-1)
#         ).mean()

#     loss_and_grad = nn.value_and_grad(model, loss_fn)
#     print(f"[DEBUG] loss_and_grad function ready")

#     # ensure output dir exists
#     out_dir = pathlib.Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print(f"[DEBUG] Checkpoints will be saved to {out_dir}")

#     # 9) training loop
#     losses_buf = []
#     t_start = time.perf_counter()
#     for step_idx in range(start_step + 1, args.total_steps + 1):
#         optim_obj.learning_rate = lr_schedule(
#             step_idx, base_lr=args.lr, warmup=args.warmup, total_steps=args.total_steps
#         )

#         batch = next(train_iter)
#         loss, grads = loss_and_grad(model, batch)
#         optim_obj.update(model, grads)
#         losses_buf.append(float(loss))

#         # report
#         if step_idx % args.steps_report == 0:
#             t_now = time.perf_counter()
#             avg = sum(losses_buf) / len(losses_buf)
#             print(
#                 f"[{step_idx:5d}/{args.total_steps}] "
#                 f"loss={avg:.3f}  it/s={(args.steps_report/(t_now-t_start)):.2f}"
#             )
#             losses_buf.clear()
#             t_start = t_now

#         # checkpoint
#         if step_idx % args.steps_checkpoint == 0:
#             ck = out_dir / f"ckpt_{step_idx:06d}.safetensors"
#             model.save_weights(str(ck))
#             print(f"[DEBUG] Saved checkpoint {ck}")

#     # final
#     final_ck = out_dir / "ckpt_final.safetensors"
#     model.save_weights(str(final_ck))
#     print(f"✅ Done — saved final checkpoint {final_ck}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
MLX GPT-style trainer
─────────────────────
* JSONL + SentencePiece loader  (cached)
* 95 : 5  train / val split
* AdamW + cosine LR schedule
* optional gradient-clipping (--grad_clip)
* checkpoint / resume
* reports both validation loss and perplexity
* saves a ‘best’ model after warmup only
"""

from __future__ import annotations

import argparse, json, math, pathlib, time
from typing import Any

import datasets
import mlx.core as mx
import mlx.nn   as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim

from tokenizer           import sp_tokenizer
from model.model         import Transformer, TransformerConfig


# --------------------------------------------------------------------------- #
#  data helpers                                                               #
# --------------------------------------------------------------------------- #
def encode_to_ids(ex: dict[str, str], *, tok) -> dict[str, list[int]]:
    """Named function so 🤗 datasets can reuse its cache."""
    return {"ids": tok.encode(ex["text"]) }


def to_samples(ids: mx.array, ctx: int) -> mx.array:
    win, n_rows = ctx + 1, ids.shape[0] // (ctx + 1)
    return ids[: n_rows * win].reshape(n_rows, win)


def batches(samples: mx.array, bs: int, *, seed=0):
    """Infinite stream of shuffled mini-batches."""
    N, key = samples.shape[0], mx.random.key(seed)
    perm, idx = mx.random.permutation(N, key=key), 0
    while True:
        if idx + bs > N:
            key, = mx.random.split(key, 1)
            perm, idx = mx.random.permutation(N, key=key), 0
        yield samples[perm[idx : idx + bs]]
        idx += bs


# --------------------------------------------------------------------------- #
#  LR schedule & grad clipping                                                #
# --------------------------------------------------------------------------- #
def lr_cosine(step, *, base, warmup, total):
    if step < warmup:
        return base * step / warmup
    t = (step - warmup) / (total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def _tree_iter(obj: Any):
    """Recursively yield mx.array leaves."""
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj: yield from _tree_iter(x)
    elif isinstance(obj, dict):
        for v in obj.values(): yield from _tree_iter(v)


def clip_by_global_norm(grads: Any, max_norm: float):
    """Scale all gradients so that their global L2-norm ≤ max_norm."""
    flats = list(_tree_iter(grads))
    norm  = math.sqrt(sum(float((g ** 2).sum()) for g in flats))
    if norm <= max_norm:
        return grads
    scale = max_norm / (norm + 1e-6)
    def _scale(obj):
        if isinstance(obj, mx.array):
            return obj * scale
        elif isinstance(obj, list):
            return [_scale(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(_scale(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: _scale(v) for k, v in obj.items()}
        return obj
    return _scale(grads)


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--context_size", type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--total_steps",  type=int, default=160_000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup",       type=int,   default=5_000)
    p.add_argument("--grad_clip",    type=float, default=None,
                   help="global-norm gradient clipping (e.g. 1.0)")
    p.add_argument("--steps_report", type=int,   default=100)
    p.add_argument("--steps_checkpoint", type=int, default=1_000)
    p.add_argument("--resume")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    args = cli()
    print("[DEBUG]", args)

    # 1) tokenizer
    tok = sp_tokenizer.load(args.tokenizer)

    # 2) dataset  (🤗 caches tokenised Arrow after first run)
    ds = datasets.load_dataset("json",
                               data_files={"train": args.dataset},
                               split="train")
    ds = ds.map(encode_to_ids, fn_kwargs={"tok": tok}, remove_columns=["text"])
    print(f"[DEBUG] tokenised {len(ds)} paragraphs")

    ids_arr = mx.array([id_ for ids in ds["ids"] for id_ in ids], dtype=mx.int32)
    print(f"[DEBUG] total tokens = {ids_arr.shape[0]}")

    samples = to_samples(ids_arr, args.context_size)
    split   = int(samples.shape[0] * 0.95)
    tr_s, val_s = samples[:split], samples[split:]
    print(f"[DEBUG] built iterator – {tr_s.shape[0]:6d} train  /  {val_s.shape[0]:6d} val")

    train_it = batches(tr_s, args.batch_size, seed=42)

    # 3) model
    cfg = TransformerConfig(**json.load(open(args.config)))
    cfg.context_size = args.context_size
    model = Transformer(cfg)

    # 4) optimiser
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.1)

    # 5) resume
    start = 0
    if args.resume:
        model.load_weights(args.resume)
        start = int(pathlib.Path(args.resume).stem.split("_")[-1])
        print(f"[DEBUG] resumed from {start}")

    # 6) prepare best-model tracking
    best_val = float('inf')
    min_best_step = args.warmup

    # 7) loss & grad
    def xent(mdl, b):
        x, y = b[:, :-1], b[:, 1:]
        return losses.cross_entropy(
            mdl(x).reshape(-1, cfg.vocab_size),
            y.reshape(-1)
        ).mean()

    loss_grad = nn.value_and_grad(model, xent)

    out_dir = pathlib.Path(args.out);
    out_dir.mkdir(exist_ok=True, parents=True)
    buf, t0 = [], time.perf_counter()

    # 8) train loop
    for step in range(start + 1, args.total_steps + 1):
        opt.learning_rate = lr_cosine(step, base=args.lr,
                                      warmup=args.warmup,
                                      total=args.total_steps)

        batch = next(train_it)
        loss, g = loss_grad(model, batch)

        if args.grad_clip:
            g = clip_by_global_norm(g, args.grad_clip)

        opt.update(model, g)
        buf.append(float(loss))

        if step % args.steps_report == 0:
            t1 = time.perf_counter()
            # compute val loss & ppl over a small sample
            N_val = min(256, val_s.shape[0])
            key   = mx.random.key(step)
            idxs  = mx.random.randint(0, val_s.shape[0], (N_val,), dtype=mx.int32, key=key)
            vl = 0.0
            for i in idxs.tolist():
                vl += float(xent(model, val_s[i][None, :]))
            avg_val = vl / N_val
            ppl     = math.exp(avg_val)

            print(f"[{step:6d}/{args.total_steps}] "
                  f"loss={sum(buf)/len(buf):.3f}  it/s={args.steps_report/(t1-t0):.2f}"  \
                  f"   ↳ val_loss={avg_val:.3f}  val_ppl={ppl:.1f}")

            # save best model after warmup
            if step >= min_best_step and avg_val < best_val:
                best_val = avg_val
                best_ck = out_dir / "ckpt_best.safetensors"
                model.save_weights(str(best_ck))
                print(f"[DEBUG] saved best model → {best_ck.name} (val_loss={best_val:.3f})")

            buf.clear(); t0 = t1

        if step % args.steps_checkpoint == 0:
            ck = out_dir / f"ckpt_{step:06d}.safetensors"
            model.save_weights(str(ck))
            print(f"[DEBUG] saved {ck.name}")

    model.save_weights(str(out_dir / "ckpt_final.safetensors"))
    print("✅ training complete")

if __name__ == "__main__":
    main()