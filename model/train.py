# model/train.py
from __future__ import annotations
import argparse, pathlib, math, json, time
from typing import Iterator, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.nn.losses as losses
from mlx.utils import tree_map

from datasets import load_dataset, load_from_disk
import sentencepiece as spm
import numpy as np

from model.model import OpenELM, SMLMConfig


# ───────────────────────── barrier & broadcast helpers ─────────
def _barrier() -> None:
    mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32)))

def _broadcast_params(params, rank: int) -> None:
    for p in tree_map(lambda x: x, params):
        if not isinstance(p, mx.array):
            continue
        if rank != 0:
            p[...] = 0
        p[...] = mx.distributed.all_sum(p)


# ───────────────────────── misc helpers ────────────────────────
def encode_sp(example: Dict[str, Any], *, sp: spm.SentencePieceProcessor, key: str):
    ids = sp.encode(example[key], out_type=int, add_bos=True, add_eos=True)
    return {"ids": ids}

def sample_generator(dataset, ctx: int, bs: int) -> Iterator[mx.array]:
    window, buf = ctx + 1, []
    while True:
        for ex in dataset:
            buf.extend(ex["ids"])
            while len(buf) >= window * bs:
                chunk = np.asarray(buf[: window * bs], dtype=np.int32)
                del buf[: window * bs]
                yield mx.array(chunk).reshape(bs, window)
        dataset = iter(dataset)

def cosine_lr(step, *, base, warmup, total, min_lr):
    if step < warmup:
        return base * step / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))

def clip_global(tree, max_norm):
    flats = [l for l in tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
    total = math.sqrt(sum(float((g**2).sum()) for g in flats))
    if total <= max_norm:
        return tree
    scale = max_norm / (total + 1e-6)
    return tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree)

def get_args():
    p = argparse.ArgumentParser("OpenELM MLX trainer")
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--resume")
    return p.parse_args()


# ─────────────────────────── main ──────────────────────────────
def main():
    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    print(f"[Rank {rank}] launcher OK ({rank+1}/{size})", flush=True)

    args = get_args()
    mx.set_default_device(mx.gpu)

    # config & tokenizer
    cfg = SMLMConfig.from_json(args.config)
    sp  = spm.SentencePieceProcessor(model_file=args.tokenizer)
    pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") >= 0 else -100

    # ─── heterogeneous local batch sizes ───────────────────────
    if rank == 2:
        LOCAL_BS = 16
    elif rank == 3:
        LOCAL_BS = 8
    else:
        LOCAL_BS = cfg.train_batch_size
    SCALE = LOCAL_BS / cfg.train_batch_size

    # dataset
    ds_path   = pathlib.Path(args.dataset)
    arrow_dir = ds_path.with_suffix(".arrow")
    if ds_path.is_dir() and any(ds_path.glob("*.arrow")):
        ds = load_from_disk(str(ds_path))
    else:
        if not arrow_dir.exists() and rank == 0:
            raw = load_dataset("json", data_files=str(ds_path), split="train")
            tmp = raw.map(
                encode_sp,
                fn_kwargs={"sp": sp, "key": "text"},
                remove_columns=raw.column_names,
                num_proc=4,
            )
            tmp.with_format("numpy", columns=["ids"]).save_to_disk(str(arrow_dir))
        _barrier()
        ds = load_from_disk(str(arrow_dir))

    # shard and shuffle per rank
    ds = ds.shard(num_shards=size, index=rank, contiguous=True)
    ds = ds.shuffle(seed=42 + rank)

    train_it = sample_generator(ds, cfg.context_size, LOCAL_BS)

    # optional tiny validation set
    val_it  = None
    val_dir = ds_path.with_name("val.arrow")
    if val_dir.exists():
        val_ds = load_from_disk(str(val_dir))
        val_ds = val_ds.shard(num_shards=size, index=rank, contiguous=True)
        val_ds = val_ds.shuffle(seed=999 + rank)
        val_it = sample_generator(val_ds, cfg.context_size, LOCAL_BS)

    # model & optimizer
    model = OpenELM(cfg)
    opt   = optim.AdamW(cfg.max_lr, betas=(0.9, .98), eps=1e-8, weight_decay=cfg.weight_decay)
    if cfg.torch_dtype == "bfloat16" and hasattr(mx, "set_default_dtype"):
        mx.set_default_dtype(mx.bfloat16)

    # resume checkpoint
    start_step = 0
    if args.resume and rank == 0:
        model.load_weights(args.resume)
        try:
            start_step = int(pathlib.Path(args.resume).stem.split("_")[-1])
        except ValueError:
            pass
        print(f"[Rank 0] resumed from {args.resume}", flush=True)

    _barrier()
    _broadcast_params(model.parameters(), rank)
    print(f"[Rank {rank}] weights synced – entering loop (local_bs={LOCAL_BS})", flush=True)

    # mini warm-up restart length
    RESTART_WARM = 1000

    # loss / grad fn
    def loss_fn(m, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = m(x)
        valid  = (y != pad_id).astype(mx.float32) if pad_id >= 0 else 1.0
        ce = losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).reshape(*y.shape)
        return (ce * valid).sum() / valid.sum()

    value_and_grad = nn.value_and_grad(model, loss_fn)
    _ = value_and_grad(model, next(train_it)); mx.eval(_)

    # training loop
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    acc_l = acc_s = 0
    for step in range(start_step + 1, cfg.max_iterations + 1):

        # LR schedule
        if step < start_step + RESTART_WARM:
            opt.learning_rate = cfg.max_lr * (step - start_step) / RESTART_WARM
        else:
            opt.learning_rate = cosine_lr(
                step,
                base   = cfg.max_lr,
                warmup = cfg.warmup_iterations,
                total  = cfg.max_iterations,
                min_lr = cfg.min_lr,
            )

        loss, grads = value_and_grad(model, next(train_it)); mx.eval(loss, grads)
        if SCALE != 1.0:  # scale for heterogeneous BS
            grads = tree_map(lambda g: g * SCALE if isinstance(g, mx.array) else g, grads)
        grads = clip_global(grads, cfg.grad_clip)
        opt.update(model, grads); mx.eval(model.parameters())

        acc_l += float(loss); acc_s += 1
        if step % 10 == 0 and rank == 0:
            print(f"[{step}/{cfg.max_iterations}] loss={acc_l/acc_s:.3f} lr={opt.learning_rate:.2e}",
                  flush=True)
            acc_l = acc_s = 0

        # validation
        if val_it and step % 5000 == 0 and rank == 0:
            vloss = 0.0
            for _ in range(100):
                vloss += float(loss_fn(model, next(val_it)))
            print(f"[val] step {step}: {vloss/100:.3f}", flush=True)

        # checkpoint
        if step % 2500 == 0 and rank == 0:
            ck = out_dir / f"ckpt_{step:07d}.safetensors"
            model.save_weights(str(ck))
            print(f"[Rank 0] checkpoint → {ck.name}", flush=True)

    if rank == 0:
        model.save_weights(str(out_dir / "ckpt_final.safetensors"))
        print("✅ Training complete", flush=True)


if __name__ == "__main__":
    main()