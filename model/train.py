# #!/usr/bin/env python3
# """
# MLX GPT‐style trainer (simplified, GPU/MPS‐transparent)
# ────────────────────────────────────────────────────────
# * JSONL → Arrow caching of tokenized dataset
# * Packs into (context_size+1)‐length samples
# * Streams shuffled minibatches via MLX (mx.random)
# * AdamW + cosine LR schedule with warmup + plateau‐aware decay
# * Gradient accumulation (--grad_accum)
# * Optional global‐norm clipping (--grad_clip)
# * Checkpoint + resume support via .save_weights / .load_weights
# * Periodic validation and perplexity reporting
# """

# from __future__ import annotations
# import argparse
# import json
# import math
# import pathlib
# import time
# from typing import Any

# import datasets
# import mlx.core as mx
# import mlx.nn as nn
# import mlx.nn.losses as losses
# import mlx.optimizers as optim
# from mlx.utils import tree_map  # canonical tree_map

# from tokenizer import sp_tokenizer
# from model.model import Transformer, TransformerConfig


# # ──────────────────────────────────────────────────────────────────────────────
# # helpers – data
# # ──────────────────────────────────────────────────────────────────────────────
# def encode_to_ids(ex: dict[str, str], *, tok):
#     """Tokenize a single JSONL record {"text": …} → {"ids": [token ids] }."""
#     return {"ids": tok.encode(ex["text"])}


# def to_samples(ids: mx.array, ctx: int) -> mx.array:
#     """
#     Reshape a 1D array of token‐ids into shape (N, ctx+1) where each row is
#     [x0 … x_{ctx-1}, x1 … x_ctx].  This forms inputs → targets pairs.
#     """
#     window = ctx + 1
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
#     Yield (batch_size, context+1) mini‐batches forever, shuffling each epoch.
#     """
#     N = samples.shape[0]
#     key = mx.random.key(seed)
#     perm = mx.random.permutation(N, key=key)
#     idx = 0
#     while True:
#         if idx + batch_size > N:
#             # start a new epoch‐shuffle
#             key, = mx.random.split(key, 1)
#             perm = mx.random.permutation(N, key=key)
#             idx = 0
#         batch_idx = perm[idx : idx + batch_size]
#         yield samples[batch_idx]
#         idx += batch_size


# # ──────────────────────────────────────────────────────────────────────────────
# # helpers – LR schedule & global‐norm clipping
# # ──────────────────────────────────────────────────────────────────────────────
# def lr_schedule(step: int, *, base_lr: float, warmup: int, total_steps: int) -> float:
#     """
#     Linear warmup to base_lr over `warmup` steps, then cosine decay to 0.
#     """
#     if step < warmup:
#         return base_lr * step / warmup
#     pct = (step - warmup) / (total_steps - warmup)
#     return base_lr * 0.5 * (1.0 + math.cos(math.pi * pct))


# def _tree_iter(obj: Any):
#     """
#     Recursively yield all mx.array leaves in a pytree.
#     """
#     if isinstance(obj, mx.array):
#         yield obj
#     elif isinstance(obj, (list, tuple)):
#         for x in obj:
#             yield from _tree_iter(x)
#     elif isinstance(obj, dict):
#         for v in obj.values():
#             yield from _tree_iter(v)


# def clip_by_global_norm(tree: Any, max_norm: float):
#     """
#     Scale all gradients so their global L2 norm ≤ max_norm.
#     """
#     flats = list(_tree_iter(tree))
#     norm = math.sqrt(sum(float((g ** 2).sum()) for g in flats))
#     if norm <= max_norm:
#         return tree
#     scale = max_norm / (norm + 1e-6)
#     def _scale(x):
#         if isinstance(x, mx.array):
#             return x * scale
#         elif isinstance(x, list):
#             return [_scale(y) for y in x]
#         elif isinstance(x, tuple):
#             return tuple(_scale(y) for y in x)
#         elif isinstance(x, dict):
#             return {k: _scale(v) for k, v in x.items()}
#         return x
#     return _scale(tree)


# # ──────────────────────────────────────────────────────────────────────────────
# # CLI
# # ──────────────────────────────────────────────────────────────────────────────
# def get_args():
#     p = argparse.ArgumentParser("MLX GPT‐LM trainer")
#     p.add_argument("--config",      required=True,
#                    help="Path to TransformerConfig JSON (e.g. model/config.json)")
#     p.add_argument("--dataset",     required=True,
#                    help="Path to JSONL file (e.g. data/raw/openwebtext_2B.jsonl)")
#     p.add_argument("--tokenizer",   required=True,
#                    help="SentencePiece .model file (e.g. tokenizer/spm.model)")
#     p.add_argument("--out",         required=True,
#                    help="Directory to save checkpoints")
#     p.add_argument("--context_size", type=int,   default=512,
#                    help="Transformer context length (tokens). Overrides config.json setting.")
#     p.add_argument("--batch_size",   type=int,   default=4,
#                    help="Per‐step micro‐batch size")
#     p.add_argument("--grad_accum",   type=int,   default=1,
#                    help="Number of micro‐batches to accumulate before weight update")
#     p.add_argument("--total_steps",  type=int,   default=200_000,
#                    help="Total training iterations")
#     p.add_argument("--lr",           type=float, default=1e-4,
#                    help="Base learning rate")
#     p.add_argument("--warmup",       type=int,   default=10_000,
#                    help="Linear warm‐up steps")
#     p.add_argument("--grad_clip",    type=float, default=None,
#                    help="Global‐norm gradient clipping (e.g. 0.5)")
#     p.add_argument("--lr_patience",  type=int,   default=50,
#                    help="Number of report intervals w/o val improvement before LR decay")
#     p.add_argument("--lr_factor",    type=float, default=0.5,
#                    help="Multiply base LR by this factor on plateau")
#     p.add_argument("--lr_min",       type=float, default=1e-6,
#                    help="Floor for base LR")
#     p.add_argument("--steps_report",     type=int, default=100,
#                    help="Report every N steps")
#     p.add_argument("--steps_checkpoint", type=int, default=1_000,
#                    help="Checkpoint every N steps")
#     p.add_argument("--resume",
#                    help="Path to safetensors checkpoint to resume from (e.g. ckpt_120000.safetensors)")
#     return p.parse_args()


# # ──────────────────────────────────────────────────────────────────────────────
# # main
# # ──────────────────────────────────────────────────────────────────────────────
# def main():
#     args = get_args()
#     print(f"[DEBUG] CLI args: {args}")

#     # ── 1) Load SentencePiece tokenizer ──────────────────────────────────────
#     tok = sp_tokenizer.load(args.tokenizer)

#     # ── 2) Load or build Arrow cache for JSONL dataset ──────────────────────
#     arrow_path = pathlib.Path(args.dataset).with_suffix(".arrow")
#     if arrow_path.exists():
#         ds = datasets.load_from_disk(str(arrow_path))
#         print(f"[DEBUG] Loaded tokenized dataset from {arrow_path}")
#     else:
#         # Stream‐tokenize and save Arrow for caching
#         ds = datasets.load_dataset("json",
#                                    data_files={"train": args.dataset},
#                                    split="train")
#         ds = ds.map(encode_to_ids, fn_kwargs={"tok": tok}, remove_columns=["text"])
#         print(f"[DEBUG] Tokenized {len(ds):,} paragraphs → {arrow_path}")
#         ds.save_to_disk(str(arrow_path))

#     # ── 3) Flatten all token IDs into one mx.array ──────────────────────────
#     ids_list = []
#     for row in ds["ids"]:
#         ids_list.extend(row)
#     ids_arr = mx.array(ids_list, dtype=mx.int32)
#     print(f"[DEBUG] Total tokens = {ids_arr.shape[0]:,}")

#     # ── 4) Pack into samples of length context_size+1 ───────────────────────
#     #     Each row of 'samples' has shape (context_size+1,)
#     context = args.context_size
#     samples = to_samples(ids_arr, context)
#     split_idx = int(samples.shape[0] * 0.95)
#     tr_s, val_s = samples[:split_idx], samples[split_idx:]
#     print(f"[DEBUG] Built iterator – {tr_s.shape[0]:,} train / {val_s.shape[0]:,} val")

#     train_it = iterate_batches(tr_s, args.batch_size, seed=42)

#     # ── 5) Instantiate Transformer model ────────────────────────────────────
#     cfg = TransformerConfig(**json.load(open(args.config)))
#     # Override context_size in the config to match our CLI
#     cfg.context_size = context
#     model = Transformer(cfg)
#     print(f"[DEBUG] Model instantiated with hidden_size={cfg.hidden_size} and {cfg.n_layer} layers")

#     # ── 6) Resume if requested ──────────────────────────────────────────────
#     start_step = 0
#     if args.resume:
#         model.load_weights(args.resume)
#         ck_name = pathlib.Path(args.resume).stem
#         try:
#             start_step = int(ck_name.split("_")[-1])
#         except:
#             print(f"[WARN] Could not parse step from {ck_name}, starting from 0")
#         print(f"[DEBUG] Resumed from checkpoint at step {start_step}")

#     # ── 7) Prepare optimizer and gradient accumulation state ─────────────────
#     base_lr      = args.lr
#     opt          = optim.AdamW(learning_rate=base_lr, weight_decay=0.1)
#     accum_target = max(1, args.grad_accum)
#     g_accum      = None
#     accum_count  = 0

#     # ── 8) Bookkeeping for validation & LR scheduling ───────────────────────
#     best_val = float("inf")
#     no_imp   = 0
#     buf, t0  = [], time.perf_counter()

#     # ── 9) Define loss + gradient function ─────────────────────────────────
#     def xent(mdl, batch: mx.array) -> mx.array:
#         x = batch[:, :-1]  # inputs, shape (B, L)
#         y = batch[:, 1:]   # targets, shape (B, L)
#         logits = mdl(x)    # (B, L, vocab_size)
#         return losses.cross_entropy(
#             logits.reshape(-1, cfg.vocab_size),
#             y.reshape(-1)
#         ).mean()

#     value_and_grad = nn.value_and_grad(model, xent)

#     # ── 10) Ensure output directory exists ─────────────────────────────────
#     out_dir = pathlib.Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print(f"[DEBUG] Checkpoints will be saved to {out_dir}")

#     # ── 11) Training loop ─────────────────────────────────────────────────
#     for step in range(start_step + 1, args.total_steps + 1):
#         # (a) Compute scheduled LR
#         opt.learning_rate = lr_schedule(
#             step, base_lr=base_lr, warmup=args.warmup, total_steps=args.total_steps
#         )

#         # (b) Fetch next micro‐batch, compute loss & grads
#         batch = next(train_it)
#         loss, grads = value_and_grad(model, batch)

#         # (c) Accumulate gradients
#         if g_accum is None:
#             g_accum = grads
#         else:
#             g_accum = tree_map(mx.add, g_accum, grads)
#         accum_count += 1

#         # (d) If seen enough micro‐batches, update weights
#         if accum_count == accum_target:
#             if args.grad_clip is not None:
#                 g_accum = clip_by_global_norm(g_accum, args.grad_clip)
#             # average over micro‐batches
#             g_accum = tree_map(lambda g: g / accum_target, g_accum)
#             opt.update(model, g_accum)
#             g_accum, accum_count = None, 0

#         buf.append(float(loss))

#         # ── (e) Reporting & validation ────────────────────────────────
#         if step % args.steps_report == 0:
#             t1 = time.perf_counter()
#             # Sample up to 256 validation examples at random
#             N_val = min(256, val_s.shape[0])
#             key = mx.random.key(step)
#             idxs = mx.random.randint(0, val_s.shape[0], (N_val,), dtype=mx.int32, key=key)
#             vl = sum(float(xent(model, val_s[i][None, :])) for i in idxs.tolist()) / N_val
#             ppl = math.exp(vl)
#             tr_loss = sum(buf) / len(buf)
#             elapsed = t1 - t0
#             print(f"[{step:6d}/{args.total_steps}] "
#                   f"loss={tr_loss:.3f}  it/s={args.steps_report/elapsed:.2f}  "
#                   f"↳ val_loss={vl:.3f}  val_ppl={ppl:.1f}")
#             buf.clear()
#             t0 = t1

#             # LR plateau logic
#             if step >= args.warmup:
#                 if vl < best_val:
#                     best_val = vl
#                     no_imp = 0
#                     best_ck = out_dir / "ckpt_best.safetensors"
#                     model.save_weights(str(best_ck))
#                     print(f"[DEBUG] Saved best → {best_ck.name} (val_loss={best_val:.3f})")
#                 else:
#                     no_imp += 1

#                 if no_imp >= args.lr_patience:
#                     old_lr = base_lr
#                     base_lr = max(base_lr * args.lr_factor, args.lr_min)
#                     print(f"[DEBUG] plateau@{step}: base_lr {old_lr:.2e} → {base_lr:.2e}")
#                     no_imp = 0

#         # ── (f) Periodic checkpoint ────────────────────────────────────
#         if step % args.steps_checkpoint == 0:
#             ck = out_dir / f"ckpt_{step:06d}.safetensors"
#             model.save_weights(str(ck))
#             print(f"[DEBUG] Saved checkpoint → {ck.name}")

#     # ── 12) Final save ────────────────────────────────────────────────
#     final_ck = out_dir / "ckpt_final.safetensors"
#     model.save_weights(str(final_ck))
#     print("✅ Training complete; final weights →", final_ck)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
MLX GPT‐style trainer (optimized for Apple Silicon)
────────────────────────────────────────────────────────
* JSONL → Arrow caching of tokenized dataset
* Packs into (context_size+1)‐length samples
* Streams shuffled minibatches via MLX (mx.random)
* AdamW + cosine LR schedule with warmup + plateau‐aware decay
* Gradient accumulation (--grad_accum)
* Optional global‐norm clipping (--grad_clip)
* Checkpoint + resume support via .save_weights / .load_weights
* Periodic validation and perplexity reporting
* Optimized for Apple Silicon GPU utilization
"""

from __future__ import annotations
import argparse
import json
import math
import pathlib
import time
from typing import Any, Iterator, List

import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map  # canonical tree_map

from tokenizer import sp_tokenizer
from model.model import Transformer, TransformerConfig


# ──────────────────────────────────────────────────────────────────────────────
# helpers – data
# ──────────────────────────────────────────────────────────────────────────────
def encode_to_ids(ex: dict[str, str], *, tok):
    """Tokenize a single JSONL record {"text": …} → {"ids": [token ids] }."""
    return {"ids": tok.encode(ex["text"])}


def to_samples(ids: mx.array, ctx: int) -> mx.array:
    """
    Reshape a 1D array of token‐ids into shape (N, ctx+1) where each row is
    [x0 … x_{ctx-1}, x1 … x_ctx].  This forms inputs → targets pairs.
    """
    window = ctx + 1
    n = ids.shape[0] // window
    trimmed = ids[: n * window]
    return trimmed.reshape(n, window)


def iterate_batches(
    samples: mx.array,
    batch_size: int,
    *,
    seed: int = 0
):
    """
    Yield (batch_size, context+1) mini‐batches forever, shuffling each epoch.
    """
    N = samples.shape[0]
    key = mx.random.key(seed)
    perm = mx.random.permutation(N, key=key)
    idx = 0
    while True:
        if idx + batch_size > N:
            # start a new epoch‐shuffle
            key, = mx.random.split(key, 1)
            perm = mx.random.permutation(N, key=key)
            idx = 0
        batch_idx = perm[idx : idx + batch_size]
        yield samples[batch_idx]
        idx += batch_size


# ──────────────────────────────────────────────────────────────────────────────
# helpers – LR schedule & global‐norm clipping
# ──────────────────────────────────────────────────────────────────────────────
def lr_schedule(step: int, *, base_lr: float, warmup: int, total_steps: int) -> float:
    """
    Linear warmup to base_lr over `warmup` steps, then cosine decay to 0.
    """
    if step < warmup:
        return base_lr * step / warmup
    pct = (step - warmup) / (total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * pct))


def _tree_iter(obj: Any):
    """
    Recursively yield all mx.array leaves in a pytree.
    """
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _tree_iter(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tree_iter(v)


def clip_by_global_norm(tree: Any, max_norm: float):
    """
    Scale all gradients so their global L2 norm ≤ max_norm.
    """
    flats = list(_tree_iter(tree))
    norm = math.sqrt(sum(float((g ** 2).sum()) for g in flats))
    if norm <= max_norm:
        return tree
    scale = max_norm / (norm + 1e-6)
    def _scale(x):
        if isinstance(x, mx.array):
            return x * scale
        elif isinstance(x, list):
            return [_scale(y) for y in x]
        elif isinstance(x, tuple):
            return tuple(_scale(y) for y in x)
        elif isinstance(x, dict):
            return {k: _scale(v) for k, v in x.items()}
        return x
    return _scale(tree)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser("MLX GPT‐LM trainer")
    p.add_argument("--config",      required=True,
                   help="Path to TransformerConfig JSON (e.g. model/config.json)")
    p.add_argument("--dataset",     required=True,
                   help="Path to JSONL file (e.g. data/raw/openwebtext_2B.jsonl)")
    p.add_argument("--tokenizer",   required=True,
                   help="SentencePiece .model file (e.g. tokenizer/spm.model)")
    p.add_argument("--out",         required=True,
                   help="Directory to save checkpoints")
    p.add_argument("--context_size", type=int,   default=512,
                   help="Transformer context length (tokens). Overrides config.json setting.")
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="Per‐step micro‐batch size")
    p.add_argument("--grad_accum",   type=int,   default=1,
                   help="Number of micro‐batches to accumulate before weight update")
    p.add_argument("--total_steps",  type=int,   default=200_000,
                   help="Total training iterations")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Base learning rate")
    p.add_argument("--warmup",       type=int,   default=10_000,
                   help="Linear warm‐up steps")
    p.add_argument("--grad_clip",    type=float, default=None,
                   help="Global‐norm gradient clipping (e.g. 0.5)")
    p.add_argument("--lr_patience",  type=int,   default=50,
                   help="Number of report intervals w/o val improvement before LR decay")
    p.add_argument("--lr_factor",    type=float, default=0.5,
                   help="Multiply base LR by this factor on plateau")
    p.add_argument("--lr_min",       type=float, default=1e-6,
                   help="Floor for base LR")
    p.add_argument("--steps_report",     type=int, default=100,
                   help="Report every N steps")
    p.add_argument("--steps_checkpoint", type=int, default=1_000,
                   help="Checkpoint every N steps")
    p.add_argument("--resume",
                   help="Path to safetensors checkpoint to resume from (e.g. ckpt_120000.safetensors)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    
    # Print device information for verification
    print(f"[DEBUG] MLX default device: {mx.default_device()}")
    try:
        active_mem = mx.get_active_memory() / 1024**3
        peak_mem = mx.get_peak_memory() / 1024**3
        print(f"[DEBUG] GPU Memory - Active: {active_mem:.1f} GB, Peak: {peak_mem:.1f} GB")
        # Set memory pool for better GPU utilization (90% of peak)
        mx.set_memory_limit(int(0.9 * mx.get_peak_memory()))
    except Exception:
        print("[DEBUG] Memory info not available")
    print(f"[DEBUG] CLI args: {args}")

    # ── 1) Load SentencePiece tokenizer ──────────────────────────────────────
    tok = sp_tokenizer.load(args.tokenizer)

    # ── 2) Load or build Arrow cache for JSONL dataset ──────────────────────
    arrow_path = pathlib.Path(args.dataset).with_suffix(".arrow")
    if arrow_path.exists():
        ds = datasets.load_from_disk(str(arrow_path))
        print(f"[DEBUG] Loaded tokenized dataset from {arrow_path}")
    else:
        # Stream‐tokenize and save Arrow for caching
        ds = datasets.load_dataset("json",
                                   data_files={"train": args.dataset},
                                   split="train")
        ds = ds.map(encode_to_ids, fn_kwargs={"tok": tok}, remove_columns=["text"])
        print(f"[DEBUG] Tokenized {len(ds):,} paragraphs → {arrow_path}")
        ds.save_to_disk(str(arrow_path))

    # ── 3) Stream token IDs → single mx.array (memory-safe) ─────────────────
    def stream_token_arrays(dataset, chunk_tokens: int = 8192 * 1024) -> Iterator[mx.array]:
        """
        Yield mx.arrays of ≤ chunk_tokens int32s so we never hold >~32 MB.
        """
        buf: List[int] = []
        for ids in dataset["ids"]:
            buf.extend(ids)
            while len(buf) >= chunk_tokens:
                chunk = np.array(buf[:chunk_tokens], dtype=np.int32)
                yield mx.array(chunk)
                del buf[:chunk_tokens]
        if buf:
            yield mx.array(np.array(buf, dtype=np.int32))

    arrays = list(stream_token_arrays(ds))
    ids_arr = mx.concatenate(arrays, axis=0)
    print(f"[DEBUG] Total tokens = {ids_arr.shape[0]:,}")

    # ── 4) Pack into samples of length context_size+1 ───────────────────────
    context = args.context_size
    samples = to_samples(ids_arr, context)
    split_idx = int(samples.shape[0] * 0.95)
    tr_s, val_s = samples[:split_idx], samples[split_idx:]
    print(f"[DEBUG] Built iterator – {tr_s.shape[0]:,} train / {val_s.shape[0]:,} val")

    train_it = iterate_batches(tr_s, args.batch_size, seed=42)

    # ── 5) Instantiate Transformer model ────────────────────────────────────
    cfg = TransformerConfig(**json.load(open(args.config)))
    cfg.context_size = context
    model = Transformer(cfg)
    
    # Count parameters
    def count_params(params):
        return sum(param.size for param in _tree_iter(params))
    
    total_params = count_params(model.parameters())
    print(f"[DEBUG] Model instantiated with {total_params:,} parameters")
    print(f"[DEBUG] Config: {cfg.n_layer} layers, {cfg.hidden_size} hidden, {cfg.n_head} heads")

    # ── 6) Resume if requested ──────────────────────────────────────────────
    start_step = 0
    if args.resume:
        model.load_weights(args.resume)
        ck_name = pathlib.Path(args.resume).stem
        try:
            start_step = int(ck_name.split("_")[-1])
        except ValueError:
            print(f"[WARN] Could not parse step from {ck_name}, starting from 0")
        print(f"[DEBUG] Resumed from checkpoint at step {start_step}")

    # ── 7) Prepare optimizer and gradient accumulation state ────────────────
    base_lr      = args.lr
    opt          = optim.AdamW(learning_rate=base_lr, weight_decay=0.1)
    accum_target = max(1, args.grad_accum)
    g_accum      = None
    accum_count  = 0

    # ── 8) Bookkeeping for validation & LR scheduling ───────────────────────
    best_val = float("inf")
    no_imp   = 0
    buf, t0  = [], time.perf_counter()

    # ── 9) Define loss + gradient function ─────────────────────────────────
    def xent(mdl, batch: mx.array) -> mx.array:
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = mdl(x)
        return losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1)
        ).mean()

    value_and_grad = nn.value_and_grad(model, xent)

    # ── 10) Ensure output directory exists ─────────────────────────────────
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Checkpoints will be saved to {out_dir}")

    # Pre-compile the model with a dummy batch for better performance
    print("[DEBUG] Pre-compiling model...")
    dummy_batch = next(train_it)
    _ = value_and_grad(model, dummy_batch)
    mx.eval(_)
    print("[DEBUG] Model pre-compilation complete")

    # ── 11) Training loop ─────────────────────────────────────────────────
    for step in range(start_step + 1, args.total_steps + 1):
        # (a) Compute scheduled LR
        opt.learning_rate = lr_schedule(
            step, base_lr=base_lr, warmup=args.warmup, total_steps=args.total_steps
        )

        # (b) Fetch next micro‐batch, compute loss & grads
        batch = next(train_it)
        loss, grads = value_and_grad(model, batch)
        mx.eval(loss, grads)

        # (c) Accumulate gradients
        if g_accum is None:
            g_accum = grads
        else:
            g_accum = tree_map(mx.add, g_accum, grads)
        accum_count += 1

        # (d) If seen enough micro‐batches, update weights
        if accum_count == accum_target:
            if args.grad_clip is not None:
                g_accum = clip_by_global_norm(g_accum, args.grad_clip)
            g_accum = tree_map(lambda g: g / accum_target, g_accum)
            opt.update(model, g_accum)
            mx.eval(model.parameters())
            g_accum, accum_count = None, 0

        buf.append(float(loss))

        # (e) Reporting & validation
        if step % args.steps_report == 0:
            t1 = time.perf_counter()
            N_val = min(128, val_s.shape[0])
            key = mx.random.key(step)
            idxs = mx.random.randint(0, val_s.shape[0], (N_val,), dtype=mx.int32, key=key)
            val_batch = val_s[idxs]
            val_losses = []
            batch_size_val = 8
            for i in range(0, N_val, batch_size_val):
                vb = val_batch[i:i + batch_size_val]
                vl_batch = xent(model, vb)
                mx.eval(vl_batch)
                val_losses.append(float(vl_batch))
            vl = sum(val_losses) / len(val_losses)
            ppl = math.exp(vl)
            tr_loss = sum(buf) / len(buf)
            elapsed = t1 - t0
            try:
                active_mem = mx.get_active_memory() / 1024**3
                peak_mem = mx.get_peak_memory() / 1024**3
            except Exception:
                active_mem = peak_mem = 0
            print(f"[{step:6d}/{args.total_steps}] "
                  f"loss={tr_loss:.3f}  it/s={args.steps_report/elapsed:.2f}  "
                  f"lr={opt.learning_rate:.2e}  "
                  f"mem={active_mem:.1f}/{peak_mem:.1f}GB  "
                  f"↳ val_loss={vl:.3f}  val_ppl={ppl:.1f}")
            buf.clear()
            t0 = t1

            # LR plateau logic
            if step >= args.warmup:
                if vl < best_val:
                    best_val = vl
                    no_imp = 0
                    best_ck = out_dir / "ckpt_best.safetensors"
                    model.save_weights(str(best_ck))
                    print(f"[DEBUG] Saved best → {best_ck.name} (val_loss={best_val:.3f})")
                else:
                    no_imp += 1

                if no_imp >= args.lr_patience:
                    old_lr = base_lr
                    base_lr = max(base_lr * args.lr_factor, args.lr_min)
                    print(f"[DEBUG] plateau@{step}: base_lr {old_lr:.2e} → {base_lr:.2e}")
                    no_imp = 0

        # (f) Periodic checkpoint
        if step % args.steps_checkpoint == 0:
            ck = out_dir / f"ckpt_{step:06d}.safetensors"
            model.save_weights(str(ck))
            print(f"[DEBUG] Saved checkpoint → {ck.name}")

    # ── 12) Final save ────────────────────────────────────────────────────
    final_ck = out_dir / "ckpt_final.safetensors"
    model.save_weights(str(final_ck))
    print("✅ Training complete; final weights →", final_ck)


if __name__ == "__main__":
    main()