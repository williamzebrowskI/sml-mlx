#!/usr/bin/env python3
"""
TinyStories – 100 M-param GPT-style model on MLX ≥ 0.25
• reads pre-tokenised data/encoded.txt (space-separated ids)
• vectorised batches of shape (B, L)
• infinite stream → exactly total_steps batches
• plain nn.value_and_grad (no JIT) for stability
• tqdm over batches + periodic loss prints + checkpoints
"""

import argparse
import json
import pathlib
import itertools

import mlx.core        as mx
import mlx.nn          as nn
import mlx.nn.losses   as losses
import mlx.optimizers as optim
from mlx.utils        import tree_map
from tqdm             import tqdm

from model.model      import Transformer, TransformerConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--out",         required=True)
    p.add_argument("--seq_len",     type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--warmup",      type=int,   default=5_000)
    p.add_argument("--total_steps", type=int,   default=8_000)
    return p.parse_args()


def lr_schedule(step, *, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / warmup
    pct = (step - warmup) / (total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + mx.cos(mx.pi * pct)))


def batch_iterator(path, seq_len, batch_size):
    """
    Read space-separated token-ids from `path`, yield mx.array(B, L).
    """
    buff = []
    with open(path) as fh:
        for line in fh:
            buff += [int(t) for t in line.split()]
            while len(buff) >= seq_len * batch_size:
                chunk = buff[: seq_len * batch_size]
                buff  = buff[seq_len * batch_size :]
                arr   = mx.array(chunk, dtype=mx.int32)
                yield arr.reshape(batch_size, seq_len)


def infinite_batch_iterator(path, seq_len, batch_size):
    """
    Cycle the dataset forever.
    """
    while True:
        yield from batch_iterator(path, seq_len, batch_size)


def main():
    args = get_args()

    # load model config & initialize
    cfg   = TransformerConfig(**json.load(open(args.config)))
    model = Transformer(cfg)

    # build an infinite iterator and slice exactly `total_steps` batches
    inf_iter = infinite_batch_iterator(
        "data/encoded.txt",
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    seq_iter = itertools.islice(inf_iter, args.total_steps)
    seq_iter = tqdm(
        seq_iter,
        total=args.total_steps,
        desc="batches",
        smoothing=0.01
    )

    # optimizer: hyper-params only
    opt = optim.AdamW(
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # loss function on a batch ⇒ scalar
    def loss_fn(batch):
        x      = batch[:, :-1]                       # (B, L-1)
        y      = batch[:, 1:]                        # (B, L-1)
        logits = model(x)                            # (B, L-1, Vocab)
        return losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1)
        ).mean()

    # forward + backward
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # ensure output dir exists
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # training loop
    for step, batch in enumerate(seq_iter):
        # scheduler
        opt.learning_rate = lr_schedule(
            step,
            base_lr=args.lr,
            warmup=args.warmup,
            total=args.total_steps
        )

        # compute loss & gradients, apply update
        loss, grads = loss_and_grad(batch)
        opt.update(model, grads)

        # periodic logging
        if step % 100 == 0:
            seq_iter.write(f"{step:6d} | loss {float(loss):.3f}")

        # checkpoint every 5k steps
        if step and step % 5000 == 0:
            ck = out_dir / f"ckpt_{step:06d}.safetensors"
            model.save_weights(str(ck))            # <— convert to str
            seq_iter.write(f"saved {ck}")

    # final checkpoint
    final_ck = out_dir / "ckpt_final.safetensors"
    model.save_weights(str(final_ck))
    print(f"✅ saved final checkpoint {final_ck}")


if __name__ == "__main__":
    main()