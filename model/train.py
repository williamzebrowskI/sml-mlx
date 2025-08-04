

# # model/train.py
# from __future__ import annotations
# import argparse, pathlib, math, json, time, itertools
# from typing import Iterator, Dict, Any

# import mlx.core as mx
# import mlx.nn as nn
# import mlx.optimizers as optim
# import mlx.nn.losses as losses
# from mlx.utils import tree_map
# import wandb

# from datasets import load_dataset, DownloadConfig
# import sentencepiece as spm
# import numpy as np

# from model.model import OpenELM, SMLMConfig

# import socket, os
# print(f"[BOOT] host={socket.gethostname()} rank_env={os.getenv('MLX_RANK')}")


# # ───────────────────────── barrier & broadcast helpers ─────────
# def _barrier() -> None:
#     mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32)))


# def _broadcast_params(params, rank: int) -> None:
#     for p in tree_map(lambda x: x, params):
#         if not isinstance(p, mx.array):
#             continue
#         if rank != 0:
#             p[...] = 0
#         p[...] = mx.distributed.all_sum(p)


# # ───────────────────────── misc helpers ────────────────────────
# def encode_sp(example: Dict[str, Any], *, sp: spm.SentencePieceProcessor, key: str):
#     ids = sp.encode(example[key], out_type=int, add_bos=True, add_eos=True)
#     return {"ids": ids}


# def sample_generator(dataset: Iterator[Dict[str, Any]], ctx: int, bs: int) -> Iterator[mx.array]:
#     window, buf = ctx + 1, []
#     while True:
#         for ex in dataset:
#             buf.extend(ex["ids"])
#             while len(buf) >= window * bs:
#                 chunk = np.asarray(buf[: window * bs], dtype=np.int32)
#                 del buf[: window * bs]
#                 yield mx.array(chunk).reshape(bs, window)


# def cosine_lr(step, *, base, warmup, total, min_lr):
#     if step < warmup:
#         return base * step / warmup
#     t = (step - warmup) / (total - warmup)
#     return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


# def clip_global(tree, max_norm):
#     flats = [l for l in tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
#     total = math.sqrt(sum(float((g**2).sum()) for g in flats))
#     if total <= max_norm:
#         return tree
#     scale = max_norm / (total + 1e-6)
#     return tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree)


# def get_args():
#     p = argparse.ArgumentParser("OpenSLM MLX trainer")
#     p.add_argument("--config",         required=True)
#     p.add_argument("--tokenizer",      required=True)
#     p.add_argument("--dataset",        required=True,
#                    help="HF dataset id (streaming)")
#     p.add_argument("--dataset-config", default=None,
#                    help="HF dataset config name")
#     p.add_argument("--train-split",    default="train",
#                    help="split or slice, e.g. 'train', 'train[:1%]'")
#     p.add_argument("--out",            required=True)
#     p.add_argument("--device",         choices=["cpu", "gpu"], default="gpu")
#     p.add_argument("--resume")
#     return p.parse_args()


# download_config = DownloadConfig(max_retries=5)   # retryable downloads


# # ─────────────────────────── main ──────────────────────────────
# def main():
#     group = mx.distributed.init()
#     rank, size = group.rank(), group.size()
#     print(f"[Rank {rank}] launcher OK ({rank+1}/{size})", flush=True)

#     args = get_args()
#     mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

#     # config & tokenizer
#     cfg = SMLMConfig.from_json(args.config)
#     sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
#     pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") >= 0 else -100

#     # per-GPU micro-batch
#     LOCAL_BS    = 4
#     ACCUM_STEPS = 16

#     # streaming dataset load & preprocess
#     for attempt in range(1, 6):
#         try:
#             time.sleep(5)
#             print(f"[Rank {rank}] load_dataset try {attempt}/5 …", flush=True)
#             ds = load_dataset(
#                 args.dataset,
#                 args.dataset_config,
#                 split=args.train_split,
#                 streaming=True,
#                 download_config=download_config,
#                 trust_remote_code=True,
#             )
#             print(f"[Rank {rank}] ✔ load_dataset complete", flush=True)
#             break
#         except Exception as e:
#             print(f"[Rank {rank}] ⚠️ load_dataset failed: {e!r}", flush=True)
#             if attempt == 5:
#                 raise
#             time.sleep(5)

#     ds = ds.map(lambda ex: encode_sp(ex, sp=sp, key="text"))
#     ds = ds.shard(num_shards=size, index=rank, contiguous=True)
#     ds = ds.shuffle(seed=42 + rank)

#     # ─── offset handling (FIXED) ─────────────────────────────────
#     out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
#     offset_file = out_dir / "offset.txt"
#     offset = int(offset_file.read_text()) if offset_file.exists() else 0
#     print(f"[Rank {rank}] skipping first {offset:,} global tokens")

#     tokens_per_rank_mb = LOCAL_BS * (cfg.context_size + 1)
#     rank_offset_tokens = offset // size
#     skip_batches = rank_offset_tokens // tokens_per_rank_mb
#     train_it = itertools.islice(
#         sample_generator(ds, cfg.context_size, LOCAL_BS),
#         skip_batches,
#         None
#     )
#     train_it = sample_generator(ds, cfg.context_size, LOCAL_BS) 
#     # running token counter
#     # tokens_per_update = size * LOCAL_BS * (cfg.context_size + 1) * ACCUM_STEPS
#     # tokens_per_micro = size * LOCAL_BS * (cfg.context_size + 1)
#     # ───────────────────────────────────────────────────────────

#     model = OpenELM(cfg)
#     opt   = optim.AdamW(cfg.max_lr, betas=(0.9, .98), eps=1e-8,
#                         weight_decay=cfg.weight_decay)
#     if cfg.torch_dtype == "bfloat16" and hasattr(mx, "set_default_dtype"):
#         mx.set_default_dtype(mx.bfloat16)

#     start_step = 0
#     if args.resume and rank == 0:
#         model.load_weights(args.resume)
#         try:
#             start_step = int(pathlib.Path(args.resume).stem.split("_")[-1])
#         except ValueError:
#             pass
#         print(f"[Rank 0] resumed from {args.resume}", flush=True)

#     _barrier()
#     _broadcast_params(model.parameters(), rank)
#     print(f"[Rank {rank}] weights synced – entering loop (local_bs={LOCAL_BS})", flush=True)

#     if rank == 0:
#         wandb.init(
#           project="fineweb-pretrain",
#           config={**cfg.__dict__, "LOCAL_BS": LOCAL_BS, "ACCUM_STEPS": ACCUM_STEPS, "world_size": size},
#           name=f"pretrain-{start_step:06d}"
#         )

#     # loss fn, warm-up compile (unchanged) …
#     def loss_fn(m, batch):
#         x, y = batch[:, :-1], batch[:, 1:]
#         logits = m(x)
#         valid = (y != pad_id).astype(mx.float32) if pad_id >= 0 else 1.0
#         ce = losses.cross_entropy(
#             logits.reshape(-1, cfg.vocab_size),
#             y.reshape(-1),
#             reduction="none",
#         ).reshape(*y.shape)
#         return (ce * valid).sum() / valid.sum()

#     value_and_grad = nn.value_and_grad(model, loss_fn)
#     _ = value_and_grad(model, mx.array(np.zeros((1, 4), dtype=np.int32))); mx.eval(_)

#     acc_l = acc_s = 0
#     accum_grads = None
#     micro_step  = 0
#     RESTART_WARM = 10000

#     for global_step in range(start_step + 1, cfg.max_iterations + 1):

#         # LR schedule (unchanged)
#         if global_step < start_step + RESTART_WARM:
#             opt.learning_rate = cfg.max_lr * (global_step - start_step) / RESTART_WARM
#         else:
#             opt.learning_rate = cosine_lr(
#                 global_step, base=cfg.max_lr,
#                 warmup=cfg.warmup_iterations,
#                 total=cfg.max_iterations,
#                 min_lr=cfg.min_lr,
#             )

#         # forward + grad
#         loss, grads = value_and_grad(model, next(train_it)); mx.eval(loss, grads)
#         grads = tree_map(lambda g: g / ACCUM_STEPS if isinstance(g, mx.array) else g, grads)

#         accum_grads = grads if accum_grads is None else tree_map(
#             lambda a, g: a + g if isinstance(a, mx.array) else a, accum_grads, grads)
#         micro_step += 1

#         if micro_step == ACCUM_STEPS:
#             clipped = clip_global(accum_grads, cfg.grad_clip)
#             clipped = tree_map(lambda g: mx.distributed.all_sum(g), clipped)
#             opt.update(model, clipped); mx.eval(model.parameters())

#             accum_grads = None; micro_step = 0

#             # ─── checkpoint & correct global offset ──────────────
#             if rank == 0 and global_step % 5000 == 0:
#                 ckpt_path = out_dir / f"ckpt_{global_step:06d}.safetensors"
#                 model.save_weights(str(ckpt_path))
#                 processed = global_step * (size * LOCAL_BS * (cfg.context_size + 1))
#                 offset_file.write_text(str(processed))
#                 print(f"[{global_step}] ✔ saved {ckpt_path.name} | "
#                       f"offset={processed:,}", flush=True)
#             # ─────────────────────────────────────────────────────

#             acc_l += float(loss); acc_s += 1
#             if global_step % 10 == 0 and rank == 0:
#                 avg_loss = acc_l/acc_s
#                 perp = math.exp(avg_loss)
#                 print(f"[{global_step}/{cfg.max_iterations}] "
#                       f"loss={acc_l/acc_s:.3f} "
#                       f"lr={opt.learning_rate:.2e}", flush=True)
                
#                 wandb.log({
#                     "train/loss":       float(avg_loss),
#                     "train/perplexity": float(perp),
#                     "train/lr":         float(opt.learning_rate)
#                 }, step=int(global_step))
#                 acc_l = acc_s = 0


#     if rank == 0:
#         model.save_weights(str(out_dir / "ckpt_final.safetensors"))
#         offset_file.write_text(str(cfg.max_iterations *
#                            (size * LOCAL_BS * (cfg.context_size + 1))))
#         print("✅ Training complete", flush=True)


# if __name__ == "__main__":
#     main()

# model/train.py
from __future__ import annotations
import argparse, pathlib, math, json, time, itertools
from typing import Iterator, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.nn.losses as losses
from mlx.utils import tree_map
import wandb

from datasets import load_dataset, DownloadConfig
import sentencepiece as spm
import numpy as np

from model.model import OpenELM, SMLMConfig

import socket, os
print(f"[BOOT] host={socket.gethostname()} rank_env={os.getenv('MLX_RANK')}")


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


def sample_generator(dataset: Iterator[Dict[str, Any]], ctx: int, bs: int) -> Iterator[mx.array]:
    window, buf = ctx + 1, []
    while True:
        for ex in dataset:
            buf.extend(ex["ids"])
            while len(buf) >= window * bs:
                chunk = np.asarray(buf[: window * bs], dtype=np.int32)
                del buf[: window * bs]
                yield mx.array(chunk).reshape(bs, window)


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


def grad_norm_from_tree(tree) -> float:
    flats = [l for l in tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
    total_sq = sum(float((g**2).sum()) for g in flats)
    return math.sqrt(total_sq)


def get_args():
    p = argparse.ArgumentParser("OpenSLM MLX trainer")
    p.add_argument("--config",         required=True)
    p.add_argument("--tokenizer",      required=True)
    p.add_argument("--dataset",        required=True,
                   help="HF dataset id (streaming)")
    p.add_argument("--dataset-config", default=None,
                   help="HF dataset config name")
    p.add_argument("--train-split",    default="train",
                   help="split or slice, e.g. 'train', 'train[:1%]'")
    p.add_argument("--out",            required=True)
    p.add_argument("--device",         choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--resume")
    return p.parse_args()


download_config = DownloadConfig(max_retries=5)   # retryable downloads


# ─────────────────────────── main ──────────────────────────────
def main():
    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    print(f"[Rank {rank}] launcher OK ({rank+1}/{size})", flush=True)

    args = get_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    # config & tokenizer
    cfg = SMLMConfig.from_json(args.config)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") >= 0 else -100

    # per-GPU micro-batch
    LOCAL_BS    = 4
    ACCUM_STEPS = 16

    # streaming dataset load & preprocess
    for attempt in range(1, 6):
        try:
            time.sleep(5)
            print(f"[Rank {rank}] load_dataset try {attempt}/5 …", flush=True)
            ds = load_dataset(
                args.dataset,
                args.dataset_config,
                split=args.train_split,
                streaming=True,
                download_config=download_config,
                trust_remote_code=True,
            )
            print(f"[Rank {rank}] ✔ load_dataset complete", flush=True)
            break
        except Exception as e:
            print(f"[Rank {rank}] ⚠️ load_dataset failed: {e!r}", flush=True)
            if attempt == 5:
                raise
            time.sleep(5)

    ds = ds.map(lambda ex: encode_sp(ex, sp=sp, key="text"))
    ds = ds.shard(num_shards=size, index=rank, contiguous=True)
    ds = ds.shuffle(seed=42 + rank)

    # ─── offset handling (FIXED) ─────────────────────────────────
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    offset_file = out_dir / "offset.txt"
    offset = int(offset_file.read_text()) if offset_file.exists() else 0
    print(f"[Rank {rank}] skipping first {offset:,} global tokens")

    tokens_per_rank_mb = LOCAL_BS * (cfg.context_size + 1)
    rank_offset_tokens = offset // size
    skip_batches = rank_offset_tokens // tokens_per_rank_mb
    train_it = itertools.islice(
        sample_generator(ds, cfg.context_size, LOCAL_BS),
        skip_batches,
        None
    )
    # running token counter
    # tokens_per_update = size * LOCAL_BS * (cfg.context_size + 1) * ACCUM_STEPS
    # tokens_per_micro = size * LOCAL_BS * (cfg.context_size + 1)

    model = OpenELM(cfg)
    opt   = optim.AdamW(cfg.max_lr, betas=(0.9, .98), eps=1e-8,
                        weight_decay=cfg.weight_decay)
    if cfg.torch_dtype == "bfloat16" and hasattr(mx, "set_default_dtype"):
        mx.set_default_dtype(mx.bfloat16)

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

    if rank == 0:
        wandb.init(
          project="fineweb-pretrain",
          config={**cfg.__dict__, "LOCAL_BS": LOCAL_BS, "ACCUM_STEPS": ACCUM_STEPS, "world_size": size},
          name=f"pretrain-{start_step:06d}"
        )

    # loss fn
    def loss_fn(m, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = m(x)
        valid = (y != pad_id).astype(mx.float32) if pad_id >= 0 else 1.0
        ce = losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).reshape(*y.shape)
        return (ce * valid).sum() / valid.sum()

    value_and_grad = nn.value_and_grad(model, loss_fn)
    _ = value_and_grad(model, mx.array(np.zeros((1, 4), dtype=np.int32))); mx.eval(_)

    acc_l = acc_s = 0
    accum_grads = None
    micro_step  = 0
    RESTART_WARM = 10000

    for global_step in range(start_step + 1, cfg.max_iterations + 1):

        # LR schedule (restart warm + cosine)
        if global_step < start_step + RESTART_WARM:
            opt.learning_rate = cfg.max_lr * (global_step - start_step) / RESTART_WARM
        else:
            opt.learning_rate = cosine_lr(
                global_step, base=cfg.max_lr,
                warmup=cfg.warmup_iterations,
                total=cfg.max_iterations,
                min_lr=cfg.min_lr,
            )

        # forward + grad
        loss, grads = value_and_grad(model, next(train_it)); mx.eval(loss, grads)
        # scale per-micro-step for accumulation
        grads = tree_map(lambda g: g / ACCUM_STEPS if isinstance(g, mx.array) else g, grads)

        accum_grads = grads if accum_grads is None else tree_map(
            lambda a, g: a + g if isinstance(a, mx.array) else a, accum_grads, grads)
        micro_step += 1

        if micro_step == ACCUM_STEPS:
            # aggregate across ranks
            global_grads = tree_map(lambda g: mx.distributed.all_sum(g), accum_grads)
            # compute gradient norm before clipping
            grad_norm = grad_norm_from_tree(global_grads)
            # clip
            clipped = clip_global(global_grads, cfg.grad_clip)
            # update
            opt.update(model, clipped); mx.eval(model.parameters())

            accum_grads = None; micro_step = 0

            # ─── checkpoint & offset bookkeeping ──────────────
            if rank == 0 and global_step % 5000 == 0:
                ckpt_path = out_dir / f"ckpt_{global_step:06d}.safetensors"
                model.save_weights(str(ckpt_path))
                processed = global_step * (size * LOCAL_BS * (cfg.context_size + 1))
                offset_file.write_text(str(processed))
                print(f"[{global_step}] ✔ saved {ckpt_path.name} | "
                      f"offset={processed:,}", flush=True)

            acc_l += float(loss); acc_s += 1
            if global_step % 10 == 0 and rank == 0:
                avg_loss = acc_l / acc_s
                perp = math.exp(avg_loss)
                print(f"[{global_step}/{cfg.max_iterations}] "
                      f"loss={avg_loss:.3f} "
                      f"perplexity={perp:.2f} "
                      f"lr={opt.learning_rate:.2e} "
                      f"grad_norm={grad_norm:.3f}", flush=True)

                wandb.log({
                    "train/loss":       float(avg_loss),
                    "train/perplexity": float(perp),
                    "train/lr":         float(opt.learning_rate),
                    "train/grad_norm":  float(grad_norm),
                }, step=int(global_step))
                acc_l = acc_s = 0

    if rank == 0:
        model.save_weights(str(out_dir / "ckpt_final.safetensors"))
        offset_file.write_text(str(cfg.max_iterations *
                           (size * LOCAL_BS * (cfg.context_size + 1))))
        print("✅ Training complete", flush=True)


if __name__ == "__main__":
    main()