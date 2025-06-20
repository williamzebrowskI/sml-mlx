#!/usr/bin/env python3
# """
# MLX GPT‐style trainer (optimized for Apple Silicon)
# ────────────────────────────────────────────────────────
# * JSONL → Arrow caching of tokenized dataset
# * Packs into (context_size+1)‐length samples
# * Streams shuffled minibatches via MLX (mx.random)
# * AdamW + cosine LR schedule with warmup + plateau‐aware decay
# * Gradient accumulation (--grad_accum)
# * Optional global‐norm clipping (--grad_clip)
# * Checkpoint + resume support via .save_weights / .load_weights
# * Periodic validation and perplexity reporting
# * Optimized for Apple Silicon GPU utilization
# """

# from __future__ import annotations
# import argparse
# import json
# import math
# import pathlib
# import time
# from typing import Any, Iterator, List

# import datasets
# import mlx.core as mx
# import mlx.nn as nn
# import mlx.nn.losses as losses
# import mlx.optimizers as optim
# import numpy as np
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
    
#     # Print device information for verification
#     print(f"[DEBUG] MLX default device: {mx.default_device()}")
#     try:
#         active_mem = mx.get_active_memory() / 1024**3
#         peak_mem = mx.get_peak_memory() / 1024**3
#         print(f"[DEBUG] GPU Memory - Active: {active_mem:.1f} GB, Peak: {peak_mem:.1f} GB")
#         # Set memory pool for better GPU utilization (90% of peak)
#         mx.set_memory_limit(int(0.9 * mx.get_peak_memory()))
#     except Exception:
#         print("[DEBUG] Memory info not available")
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

#     # ── 3) Stream token IDs → single mx.array (memory-safe) ─────────────────
#     def stream_token_arrays(dataset, chunk_tokens: int = 8192 * 1024) -> Iterator[mx.array]:
#         """
#         Yield mx.arrays of ≤ chunk_tokens int32s so we never hold >~32 MB.
#         """
#         buf: List[int] = []
#         for ids in dataset["ids"]:
#             buf.extend(ids)
#             while len(buf) >= chunk_tokens:
#                 chunk = np.array(buf[:chunk_tokens], dtype=np.int32)
#                 yield mx.array(chunk)
#                 del buf[:chunk_tokens]
#         if buf:
#             yield mx.array(np.array(buf, dtype=np.int32))

#     arrays = list(stream_token_arrays(ds))
#     ids_arr = mx.concatenate(arrays, axis=0)
#     print(f"[DEBUG] Total tokens = {ids_arr.shape[0]:,}")

#     # ── 4) Pack into samples of length context_size+1 ───────────────────────
#     context = args.context_size
#     samples = to_samples(ids_arr, context)
#     split_idx = int(samples.shape[0] * 0.95)
#     tr_s, val_s = samples[:split_idx], samples[split_idx:]
#     print(f"[DEBUG] Built iterator – {tr_s.shape[0]:,} train / {val_s.shape[0]:,} val")

#     train_it = iterate_batches(tr_s, args.batch_size, seed=42)

#     # ── 5) Instantiate Transformer model ────────────────────────────────────
#     cfg = TransformerConfig(**json.load(open(args.config)))
#     cfg.context_size = context
#     model = Transformer(cfg)
    
#     # Count parameters
#     def count_params(params):
#         return sum(param.size for param in _tree_iter(params))
    
#     total_params = count_params(model.parameters())
#     print(f"[DEBUG] Model instantiated with {total_params:,} parameters")
#     print(f"[DEBUG] Config: {cfg.n_layer} layers, {cfg.hidden_size} hidden, {cfg.n_head} heads")

#     # ── 6) Resume if requested ──────────────────────────────────────────────
#     start_step = 0
#     if args.resume:
#         model.load_weights(args.resume)
#         ck_name = pathlib.Path(args.resume).stem
#         try:
#             start_step = int(ck_name.split("_")[-1])
#         except ValueError:
#             print(f"[WARN] Could not parse step from {ck_name}, starting from 0")
#         print(f"[DEBUG] Resumed from checkpoint at step {start_step}")

#     # ── 7) Prepare optimizer and gradient accumulation state ────────────────
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
#         x = batch[:, :-1]
#         y = batch[:, 1:]
#         logits = mdl(x)
#         return losses.cross_entropy(
#             logits.reshape(-1, cfg.vocab_size),
#             y.reshape(-1)
#         ).mean()

#     value_and_grad = nn.value_and_grad(model, xent)

#     # ── 10) Ensure output directory exists ─────────────────────────────────
#     out_dir = pathlib.Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print(f"[DEBUG] Checkpoints will be saved to {out_dir}")

#     # Pre-compile the model with a dummy batch for better performance
#     print("[DEBUG] Pre-compiling model...")
#     dummy_batch = next(train_it)
#     _ = value_and_grad(model, dummy_batch)
#     mx.eval(_)
#     print("[DEBUG] Model pre-compilation complete")

#     # ── 11) Training loop ─────────────────────────────────────────────────
#     for step in range(start_step + 1, args.total_steps + 1):
#         # (a) Compute scheduled LR
#         opt.learning_rate = lr_schedule(
#             step, base_lr=base_lr, warmup=args.warmup, total_steps=args.total_steps
#         )

#         # (b) Fetch next micro‐batch, compute loss & grads
#         batch = next(train_it)
#         loss, grads = value_and_grad(model, batch)
#         mx.eval(loss, grads)

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
#             g_accum = tree_map(lambda g: g / accum_target, g_accum)
#             opt.update(model, g_accum)
#             mx.eval(model.parameters())
#             g_accum, accum_count = None, 0

#         buf.append(float(loss))

#         # (e) Reporting & validation
#         if step % args.steps_report == 0:
#             t1 = time.perf_counter()
#             N_val = min(128, val_s.shape[0])
#             key = mx.random.key(step)
#             idxs = mx.random.randint(0, val_s.shape[0], (N_val,), dtype=mx.int32, key=key)
#             val_batch = val_s[idxs]
#             val_losses = []
#             batch_size_val = 8
#             for i in range(0, N_val, batch_size_val):
#                 vb = val_batch[i:i + batch_size_val]
#                 vl_batch = xent(model, vb)
#                 mx.eval(vl_batch)
#                 val_losses.append(float(vl_batch))
#             vl = sum(val_losses) / len(val_losses)
#             ppl = math.exp(vl)
#             tr_loss = sum(buf) / len(buf)
#             elapsed = t1 - t0
#             try:
#                 active_mem = mx.get_active_memory() / 1024**3
#                 peak_mem = mx.get_peak_memory() / 1024**3
#             except Exception:
#                 active_mem = peak_mem = 0
#             print(f"[{step:6d}/{args.total_steps}] "
#                   f"loss={tr_loss:.3f}  it/s={args.steps_report/elapsed:.2f}  "
#                   f"lr={opt.learning_rate:.2e}  "
#                   f"mem={active_mem:.1f}/{peak_mem:.1f}GB  "
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

#         # (f) Periodic checkpoint
#         if step % args.steps_checkpoint == 0:
#             ck = out_dir / f"ckpt_{step:06d}.safetensors"
#             model.save_weights(str(ck))
#             print(f"[DEBUG] Saved checkpoint → {ck.name}")

#     # ── 12) Final save ────────────────────────────────────────────────────
#     final_ck = out_dir / "ckpt_final.safetensors"
#     model.save_weights(str(final_ck))
#     print("✅ Training complete; final weights →", final_ck)


# if __name__ == "__main__":
#     main()

# train_openelm_mlx.py – OpenELM‑270 M pre‑training on MLX
# -----------------------------------------------------------------------------
# This script loads an **Arrow‑backed** OpenWebText‑2B corpus (or a JSONL file),
# streams mini‑batches without ever materialising the full token array in RAM,
# and trains the 270 M‑parameter OpenELM model you defined in `model/`.
# -----------------------------------------------------------------------------
# Usage example:
#   python train_openelm_mlx.py \
#       --yaml configs/openelm_270M_pretrain.yaml \
#       --tokenizer tokenizer.model \
#       --dataset data/raw/openwebtext_2B.arrow  \
#       --out runs/elm270m
# -----------------------------------------------------------------------------

# from __future__ import annotations
# import argparse, pathlib, time, math, yaml
# from typing import Iterator, Dict, List, Any

# import mlx.core as mx
# import mlx.nn   as nn
# import mlx.optimizers as optim
# import mlx.nn.losses as losses
# from mlx.utils import tree_map

# from datasets import load_dataset, load_from_disk
# import sentencepiece as spm

# import numpy as np
# from model.model  import OpenELM
# from model.config import preset_270m, SMLMConfig

# # ───────────────────────────────────────────────────────────────────────────────
# # helpers ─ data streaming
# # ───────────────────────────────────────────────────────────────────────────────

# def encode_sp(example: Dict[str, Any], *, sp: spm.SentencePieceProcessor, key: str) -> Dict[str, List[int]]:
#     ids = sp.encode(example[key], out_type=int, add_bos=True, add_eos=True)
#     return {"ids": ids}



# def sample_generator(dataset, ctx: int, bs: int):
#     """
#     Stream `(bs, ctx+1)` batches forever.
#     Uses NumPy to ensure the buffer is int32 before MLX wraps it.
#     """
#     window = ctx + 1
#     buf: list[int] = []

#     while True:
#         for example in dataset:
#             buf.extend(example["ids"])

#             # emit as many full windows as we can
#             while len(buf) >= window * bs:
#                 # 1) cast Python-int list → NumPy int32
#                 chunk = np.asarray(buf[: window * bs], dtype=np.int32)
#                 # 2) wrap with MLX (dtype is already int32)
#                 arr   = mx.array(chunk)

#                 del buf[: window * bs]
#                 yield arr.reshape(bs, window)

#         dataset = iter(dataset)      # restart Arrow iterator if exhausted

# # ───────────────────────────────────────────────────────────────────────────────
# # helpers ─ LR schedule & grad clip
# # ───────────────────────────────────────────────────────────────────────────────

# def cosine_lr(step: int, *, base: float, warmup: int, total: int, min_lr: float) -> float:
#     if step < warmup:
#         return base * step / warmup
#     t = (step - warmup) / (total - warmup)
#     return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


# def clip_global(tree, max_norm: float):
#     """
#     Scale all gradients so their global L2-norm ≤ max_norm.
#     Works even when the pytree contains strings / None.
#     """
#     flats: list[mx.array] = [
#         leaf for leaf in tree_map(lambda x: x, tree)     # flatten
#         if isinstance(leaf, mx.array)                    # keep tensors only
#     ]

#     total = math.sqrt(sum(float((g ** 2).sum()) for g in flats))
#     if total <= max_norm:
#         return tree

#     scale = max_norm / (total + 1e-6)
#     return tree_map(
#         lambda x: x * scale if isinstance(x, mx.array) else x,
#         tree
#     )

# # ───────────────────────────────────────────────────────────────────────────────
# # CLI
# # ───────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser("OpenELM‑270M MLX trainer (Arrow streaming)")
#     p.add_argument("--yaml",      required=True, help="CoreNet YAML recipe path")
#     p.add_argument("--tokenizer", required=True, help="SentencePiece model")
#     p.add_argument("--dataset",   required=True, help="Path to .arrow directory OR .jsonl file")
#     p.add_argument("--out",       required=True, help="Output checkpoint directory")
#     p.add_argument("--device",    choices=["cpu", "gpu"], default="gpu")
#     return p.parse_args()

# # ───────────────────────────────────────────────────────────────────────────────
# # main
# # ───────────────────────────────────────────────────────────────────────────────

# def main():
#     args = get_args()

#     # device
#     mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
#     print("[info] device:", mx.default_device())

#     # YAML hyper‑params
#     cfg_y = yaml.safe_load(open(args.yaml))
#     ctx_len = cfg_y["dataset"]["language_modeling"]["sequence_length"]
#     batch   = cfg_y["dataset"]["train_batch_size0"]
#     steps   = cfg_y["scheduler"]["max_iterations"]
#     warm    = cfg_y["scheduler"]["warmup_iterations"]
#     max_lr  = cfg_y["scheduler"]["cosine"]["max_lr"]
#     min_lr  = cfg_y["scheduler"]["cosine"]["min_lr"]
#     clip_v  = cfg_y["common"]["grad_clip"]
#     wd      = cfg_y["optim"]["weight_decay"]

#     # tokenizer
#     sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

#     pad_id = sp.piece_to_id("<pad>")
#     if pad_id == -1:
#         pad_id = -100 


#     # dataset loading
#     ds_path = pathlib.Path(args.dataset)
#     if ds_path.is_dir():
#         print(f"[info] loading Arrow dataset from {ds_path}")
#         ds = load_from_disk(str(ds_path))
#     else:
#         print(f"[info] loading JSONL from {ds_path}")
#         ds = load_dataset("json", data_files=str(ds_path), split="train")
#         ds = ds.map(encode_sp, fn_kwargs={"sp": sp, "key": "text"}, remove_columns=ds.column_names, num_proc=4)
#     # ensure NumPy output & iterate infinitely
#     ds = ds.with_format("numpy", columns=["ids"])
#     data_iter = sample_generator(ds, ctx_len, batch)

#     # model & optim
#     cfg: SMLMConfig = preset_270m()
#     model = OpenELM(cfg)
#     opt = optim.AdamW(learning_rate=max_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)

#     if hasattr(mx, "set_default_dtype"):
#         mx.set_default_dtype(mx.bfloat16)

#     def loss_fn(m: nn.Module, b: mx.array):
#         x, y = b[:, :-1], b[:, 1:]
#         logits = m(x)                             # (B, L-1, vocab)

#         # mask out positions that equal pad_id
#         if pad_id != -100:                        # –100 means “no explicit <pad>”
#             valid = (y != pad_id).astype(mx.float32)        # (B, L-1)
#             loss = losses.cross_entropy(
#                 logits.reshape(-1, sp.vocab_size()),
#                 y.reshape(-1),
#                 reduction="none",                 # per-token loss
#             ).reshape(*y.shape) * valid           # zero out pad positions
#             loss = loss.sum() / valid.sum()       # mean over real tokens
#         else:
#             loss = losses.cross_entropy(
#                 logits.reshape(-1, sp.vocab_size()),
#                 y.reshape(-1)
#             ).mean()

#         return loss.astype(mx.float32)

#     value_and_grad = nn.value_and_grad(model, loss_fn)
#     _ = value_and_grad(model, next(data_iter));  mx.eval(_)

#     ckpt_dir = pathlib.Path(args.out); ckpt_dir.mkdir(parents=True, exist_ok=True)

#     acc_l, acc_s, t0 = 0.0, 0, time.time()
#     for step in range(1, steps + 1):
#         opt.learning_rate = cosine_lr(step, base=max_lr, warmup=warm, total=steps, min_lr=min_lr)
#         batch = next(data_iter)
#         loss, grads = value_and_grad(model, batch);  mx.eval(loss, grads)
#         grads = clip_global(grads, clip_v)
#         opt.update(model, grads);  mx.eval(model.parameters())

#         acc_l += float(loss);  acc_s += 1
#         if step % 500 == 0:
#             dt = time.time() - t0
#             print(f"[step {step:7d}/{steps}] loss={acc_l/acc_s:.3f} lr={opt.learning_rate:.2e} {acc_s/dt:.2f} it/s")
#             acc_l = acc_s = 0; t0 = time.time()
#         if step % 5000 == 0:
#             ck = ckpt_dir / f"ckpt_{step:07d}.safetensors"; model.save_weights(str(ck))
#             print("[info] checkpoint →", ck.name)

#     model.save_weights(str(ckpt_dir / "ckpt_final.safetensors"))
#     print("✅ Training complete.")


# if __name__ == "__main__":
#     main()


# model/train.py
from __future__ import annotations
import argparse, pathlib, time, math, json
from typing import Iterator, Dict, Any

import mlx.core as mx
import mlx.nn   as nn
import mlx.optimizers as optim
import mlx.nn.losses as losses
from mlx.utils import tree_map

from datasets import load_dataset, load_from_disk
import sentencepiece as spm

import numpy as np
from model.model import OpenELM, SMLMConfig
from model.utils import generate_stream # your spot-check gen

# ───────────────────────────────────────────────────────────────────────────────
def encode_sp(example: Dict[str, Any], *, sp: spm.SentencePieceProcessor, key: str):
    ids = sp.encode(example[key], out_type=int,
                    add_bos=True, add_eos=True)
    return {"ids": ids}


def sample_generator(dataset, ctx: int, bs: int) -> Iterator[mx.array]:
    """
    Yield `(bs, ctx+1)` batches forever, using NumPy to enforce int32.
    """
    window = ctx + 1
    buf: list[int] = []

    while True:
        for ex in dataset:
            buf.extend(ex["ids"])
            while len(buf) >= window * bs:
                # 1) cast to NumPy int32
                chunk = np.asarray(buf[: window * bs], dtype=np.int32)
                # 2) wrap with MLX
                arr   = mx.array(chunk)
                # 3) pop from Python list
                del buf[: window * bs]
                yield arr.reshape(bs, window)
        dataset = iter(dataset)

def cosine_lr(step: int, *, base: float, warmup: int,
              total: int, min_lr: float) -> float:
    if step < warmup:
        return base * step / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))

def clip_global(tree, max_norm: float):
    flats = [leaf for leaf in tree_map(lambda x: x, tree)
             if isinstance(leaf, mx.array)]
    total = math.sqrt(sum(float((g**2).sum()) for g in flats))
    if total <= max_norm:
        return tree
    scale = max_norm / (total + 1e-6)
    return tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree)

# ───────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser("OpenELM MLX trainer w/ validation")
    p.add_argument("--config",   required=True,
                   help="Path to unified config.json")
    p.add_argument("--tokenizer",required=True,
                   help="SentencePiece .model")
    p.add_argument("--dataset",  required=True,
                   help=".arrow dir or JSONL train")
    p.add_argument("--out",      required=True,
                   help="checkpoint directory")
    p.add_argument("--device",   choices=["cpu","gpu"],
                   default="gpu")
    p.add_argument("--resume",
                   help="Path to .safetensors checkpoint")
    return p.parse_args()

# ───────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    mx.set_default_device(mx.cpu if args.device=="cpu" else mx.gpu)
    print("[info] device:", mx.default_device())

    # 1) load config.json
    cfg: SMLMConfig = SMLMConfig.from_json(args.config)

    # 2) tokenizer + pad handling
    sp    = spm.SentencePieceProcessor(model_file=args.tokenizer)
    pad_id = sp.piece_to_id("<pad>")
    if pad_id < 0: pad_id = -100

    # 3) train dataset
    ds_path = pathlib.Path(args.dataset)
    if ds_path.is_dir():
        ds = load_from_disk(str(ds_path))
    else:
        ds = load_dataset("json", data_files=str(ds_path),
                           split="train")
        ds = ds.map(encode_sp, fn_kwargs={"sp":sp,"key":"text"},
                    remove_columns=ds.column_names,
                    num_proc=4)
    ds = ds.with_format("numpy", columns=["ids"])
    train_it = sample_generator(ds,
                                cfg.context_size,
                                cfg.train_batch_size)

    # 5) model & optimizer
    model = OpenELM(cfg)
    opt   = optim.AdamW(
        learning_rate=cfg.max_lr,
        betas=(0.9,0.95), eps=1e-8,
        weight_decay=cfg.weight_decay
    )
    # set default dtype for weights/activations
    if cfg.torch_dtype=="bfloat16" and hasattr(mx, "set_default_dtype"):
        mx.set_default_dtype(mx.bfloat16)

    # 6) resume logic
    start_step = 0
    if args.resume:
        model.load_weights(args.resume)
        try:
            start_step = int(pathlib.Path(args.resume).stem.split("_")[-1])
        except:
            pass
        print("[info] resumed from", args.resume)

    # 7) loss + grad
    def loss_fn(m, batch):
        x, y = batch[:,:-1], batch[:,1:]
        logits = m(x)
        if pad_id >= 0:
            valid = (y!=pad_id).astype(mx.float32)
            ce = losses.cross_entropy(
              logits.reshape(-1, cfg.vocab_size),
              y.reshape(-1),
              reduction="none"
            ).reshape(*y.shape) * valid
            return (ce.sum()/valid.sum()).astype(mx.float32)
        return losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1)
        ).mean().astype(mx.float32)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    _ = value_and_grad(model, next(train_it)); mx.eval(_)

    # 8) training loop
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_l = acc_s = 0
    t0    = time.time()

    for step in range(start_step+1, cfg.max_iterations+1):
        opt.learning_rate = cosine_lr(
            step,
            base=cfg.max_lr,
            warmup=cfg.warmup_iterations,
            total=cfg.max_iterations,
            min_lr=cfg.min_lr
        )

        batch = next(train_it)
        loss, grads = value_and_grad(model, batch)
        mx.eval(loss, grads)
        grads = clip_global(grads, cfg.grad_clip)
        opt.update(model, grads)
        mx.eval(model.parameters())

        acc_l += float(loss)
        acc_s += 1

        # train‐loss report
        if step % 100 == 0:
            print(f"[{step}/{cfg.max_iterations}] "
                  f"train_loss={acc_l/acc_s:.3f}  lr={opt.learning_rate:.2e}")
            acc_l = acc_s = 0
            t0    = time.time()

        if step % 2500 == 0:
            ck = out_dir / f"ckpt_{step:07d}.safetensors"
            model.save_weights(str(ck))
            print("[info] checkpoint →", ck.name)

        #     # spot-check generation, streaming token-by-token
        #     prompt = "The meaning of life is"
        #     print("\n🦜‍🦉  Sample from step", step)
        #     print("User >", prompt)
        #     print("Model >", end=" ", flush=True)

        #     for token in generate_stream(prompt, max_new=20, top_k=40, temp=0.7):
        #         print(token, end="", flush=True)
        #     print("\n")

    # final save
    final_ck = out_dir / "ckpt_final.safetensors"
    model.save_weights(str(final_ck))
    print("✅ Training complete; final →", final_ck)

if __name__=="__main__":
    main()