#!/usr/bin/env python3
"""
MLX GPT-style trainer  (Apple-Silicon friendly, fp32)
─────────────────────────────────────────────────────
…  ←  unchanged doc-string  …
"""
from __future__ import annotations
import argparse, json, math, pathlib, time
from typing import Any

import datasets
import mlx.core as mx
import mlx.nn   as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim
from mlx.utils import tree_map

from tokenizer   import sp_tokenizer
from model.model import Transformer, TransformerConfig

# ───────────────────────── data helpers ─────────────────────────
def encode_to_ids(ex: dict[str, str], *, tok):               # unchanged
    return {"ids": tok.encode(ex["text"])}

def to_samples(ids: mx.array, ctx: int) -> mx.array:
    win, rows = ctx + 1, ids.shape[0] // (ctx + 1)
    return ids[: rows * win].reshape(rows, win)

def batches(samples: mx.array, bs: int, *, seed=0):          # unchanged
    N, key = samples.shape[0], mx.random.key(seed)
    perm, idx = mx.random.permutation(N, key=key), 0
    while True:
        if idx + bs > N:
            key, = mx.random.split(key, 1)
            perm, idx = mx.random.permutation(N, key=key), 0
        yield samples[perm[idx: idx + bs]]
        idx += bs

# ─────────────────── scheduler / clipping helpers ──────────────
def lr_schedule(step: int, *, base: float, warmup: int, total: int) -> float:
    if step < warmup:                     # linear warm-up
        return base * step / warmup
    t = (step - warmup) / (total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * t))  # cosine

def _tree_iter(obj: Any):
    if isinstance(obj, mx.array):    yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj: yield from _tree_iter(x)
    elif isinstance(obj, dict):
        for v in obj.values(): yield from _tree_iter(v)

def clip_by_global_norm(tree: Any, max_norm: float):
    flats = list(_tree_iter(tree))
    norm  = math.sqrt(sum(float((g**2).sum()) for g in flats))
    if norm <= max_norm: return tree
    scale = max_norm / (norm + 1e-6)
    def _s(x):
        if   isinstance(x, mx.array): return x * scale
        elif isinstance(x, list):     return [_s(y) for y in x]
        elif isinstance(x, tuple):    return tuple(_s(y) for y in x)
        elif isinstance(x, dict):     return {k:_s(v) for k,v in x.items()}
        return x
    return _s(tree)

# ─────────────────────────── CLI ───────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--context_size", type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--grad_accum",   type=int, default=1)

    p.add_argument("--total_steps",  type=int, default=200_000)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--warmup",       type=int,   default=10_000)
    p.add_argument("--grad_clip",    type=float, default=None)

    p.add_argument("--lr_patience",  type=int,   default=50)
    p.add_argument("--lr_factor",    type=float, default=0.5)
    p.add_argument("--lr_min",       type=float, default=1e-6)

    p.add_argument("--steps_report",     type=int, default=100)
    p.add_argument("--steps_checkpoint", type=int, default=1_000)
    p.add_argument("--resume")
    return p.parse_args()

# ─────────────────────────── main ──────────────────────────────
def main():
    args = cli();  print("[DEBUG]", args)

    tok = sp_tokenizer.load(args.tokenizer)

    arrow = pathlib.Path(args.dataset).with_suffix(".arrow")
    if arrow.exists():
        ds = datasets.load_from_disk(str(arrow))
        print(f"[DEBUG] loaded tokenised dataset from {arrow}")
    else:
        ds = datasets.load_dataset("json",
                                   data_files={"train": args.dataset},
                                   split="train")
        ds = ds.map(encode_to_ids, fn_kwargs={"tok":tok},
                    remove_columns=["text"])
        print(f"[DEBUG] tokenised {len(ds):,} paragraphs → {arrow}")
        ds.save_to_disk(str(arrow))

    ids = mx.array([tok_id for row in ds["ids"] for tok_id in row],
                   dtype=mx.int32)
    print(f"[DEBUG] total tokens = {ids.shape[0]:,}")

    samples = to_samples(ids, args.context_size)
    tr_s, val_s = samples[:int(0.95*len(samples))], samples[int(0.95*len(samples)):]
    print(f"[DEBUG] iterator – {tr_s.shape[0]:,} train / {val_s.shape[0]:,} val")

    train_it = batches(tr_s, args.batch_size, seed=42)

    cfg = TransformerConfig(**json.load(open(args.config)))
    cfg.context_size = args.context_size
    model = Transformer(cfg)

    start = 0
    if args.resume:
        model.load_weights(args.resume)
        start = int(pathlib.Path(args.resume).stem.split("_")[-1])
        print(f"[DEBUG] resumed from step {start}")

    # just-the-forward compiled (keeps model intact)
    fwd = mx.compile(model)

    base_lr = args.lr
    opt     = optim.AdamW(learning_rate=base_lr, weight_decay=0.1)
    accum_target = max(1, args.grad_accum)
    g_accum = None;  accum_cnt = 0

    best_val = float("inf");  no_imp = 0
    buf, t0  = [], time.perf_counter()
    out_dir  = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    def xent(_unused, b):
        x,y = b[:,:-1], b[:,1:]
        return losses.cross_entropy(fwd(x).reshape(-1,cfg.vocab_size),
                                    y.reshape(-1)).mean()
    v_and_g = nn.value_and_grad(model, xent)

    for step in range(start+1, args.total_steps+1):
        opt.learning_rate = lr_schedule(step, base=base_lr,
                                        warmup=args.warmup,
                                        total=args.total_steps)
        loss, grads = v_and_g(model, next(train_it))
        g_accum = grads if g_accum is None else tree_map(mx.add,g_accum,grads)
        accum_cnt += 1

        if accum_cnt == accum_target:
            if args.grad_clip: g_accum = clip_by_global_norm(g_accum, args.grad_clip)
            g_accum = tree_map(lambda g: g/accum_target, g_accum)
            opt.update(model, g_accum)
            g_accum, accum_cnt = None, 0
        buf.append(float(loss))

        if step % args.steps_report == 0:
            idx = mx.random.randint(0,val_s.shape[0],(256,),dtype=mx.int32,
                                    key=mx.random.key(step))
            vl = sum(float(xent(None, val_s[i][None,:])) for i in idx.tolist())/256
            print(f"[{step:6d}/{args.total_steps}] "
                  f"loss={sum(buf)/len(buf):.3f}  "
                  f"it/s={args.steps_report/(time.perf_counter()-t0):.2f}  "
                  f"↳ val_loss={vl:.3f}  val_ppl={math.exp(vl):.1f}")
            buf.clear(); t0=time.perf_counter()

            if step>=args.warmup:
                if vl<best_val:
                    best_val=vl; no_imp=0
                    model.save_weights(str(out_dir/"ckpt_best.safetensors"))
                    print(f"[DEBUG] new best {best_val:.3f}")
                else:
                    no_imp+=1
                if no_imp>=args.lr_patience:
                    old=base_lr; base_lr=max(base_lr*args.lr_factor,args.lr_min)
                    print(f"[DEBUG] plateau@{step}: {old:.2e}→{base_lr:.2e}")
                    no_imp=0
        if step % args.steps_checkpoint==0:
            ck=out_dir/f"ckpt_{step:06d}.safetensors"
            model.save_weights(str(ck)); print(f"[DEBUG] saved {ck.name}")

    model.save_weights(str(out_dir/"ckpt_final.safetensors"))
    print("✅ training complete")

if __name__=="__main__":
    main()