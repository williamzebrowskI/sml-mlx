# #!/usr/bin/env python3
# """
# TinyStories – 100 M-param GPT-style model on MLX ≥ 0.25
# • reads pre-tokenised data/encoded.txt (space-separated ids)
# • vectorised batches of shape (B, L)
# • infinite stream → exactly total_steps batches
# • plain nn.value_and_grad (no JIT) for stability
# • tqdm over batches + periodic loss prints + checkpoints
# """

# import argparse
# import json
# import pathlib
# import itertools

# import mlx.core        as mx
# import mlx.nn          as nn
# import mlx.nn.losses   as losses
# import mlx.optimizers as optim
# from mlx.utils        import tree_map
# from tqdm             import tqdm

# from model.model      import Transformer, TransformerConfig


# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--config",    required=True, help="Path to model/config.json")
#     p.add_argument("--tokenizer", required=True, help="Path to SentencePiece .model file")
#     p.add_argument("--dataset",   required=True, help="Path to JSONL dataset (one {'text':...} per line)")
#     p.add_argument("--out",       required=True, help="Directory to save checkpoints")
#     p.add_argument("--seq_len",    type=int, default=512, help="Context length (sequence length)")
#     p.add_argument("--batch_size", type=int, default=8, help="Number of sequences per batch (streamed)")
#     p.add_argument("--grad_accum", type=int, default=32, help="Gradient accumulation steps")
#     p.add_argument("--lr",         type=float, default=2e-4, help="Base learning rate")
#     p.add_argument("--warmup",     type=int,   default=5_000, help="Warmup steps for LR schedule")
#     p.add_argument("--total_steps",type=int,   default=8_000, help="Total training steps")
#     # p.add_argument("--config",      required=True)
#     # p.add_argument("--out",         required=True)
#     # p.add_argument("--seq_len",     type=int,   default=512)
#     # p.add_argument("--batch_size",  type=int,   default=4)
#     # p.add_argument("--lr",          type=float, default=2e-4)
#     # p.add_argument("--warmup",      type=int,   default=5_000)
#     # p.add_argument("--total_steps", type=int,   default=8_000)
#     return p.parse_args()


# def lr_schedule(step, *, base_lr, warmup, total):
#     if step < warmup:
#         return base_lr * step / warmup
#     pct = (step - warmup) / (total - warmup)
#     return base_lr * (0.1 + 0.9 * 0.5 * (1 + mx.cos(mx.pi * pct)))


# def batch_iterator(path, seq_len, batch_size):
#     """
#     Read space-separated token-ids from `path`, yield mx.array(B, L).
#     """
#     buff = []
#     with open(path) as fh:
#         for line in fh:
#             buff += [int(t) for t in line.split()]
#             while len(buff) >= seq_len * batch_size:
#                 chunk = buff[: seq_len * batch_size]
#                 buff  = buff[seq_len * batch_size :]
#                 arr   = mx.array(chunk, dtype=mx.int32)
#                 yield arr.reshape(batch_size, seq_len)


# def infinite_batch_iterator(path, seq_len, batch_size):
#     """
#     Cycle the dataset forever.
#     """
#     while True:
#         yield from batch_iterator(path, seq_len, batch_size)


# def main():
#     args = get_args()

#     # load model config & initialize
#     cfg   = TransformerConfig(**json.load(open(args.config)))
#     model = Transformer(cfg)

#     # build an infinite iterator and slice exactly `total_steps` batches
#     inf_iter = infinite_batch_iterator(
#         "data/encoded.txt",
#         seq_len=args.seq_len,
#         batch_size=args.batch_size
#     )
#     seq_iter = itertools.islice(inf_iter, args.total_steps)
#     seq_iter = tqdm(
#         seq_iter,
#         total=args.total_steps,
#         desc="batches",
#         smoothing=0.01
#     )

#     # optimizer: hyper-params only
#     opt = optim.AdamW(
#         learning_rate=args.lr,
#         betas=(0.9, 0.95),
#         weight_decay=0.1,
#     )

#     # loss function on a batch ⇒ scalar
#     def loss_fn(batch):
#         x      = batch[:, :-1]                       # (B, L-1)
#         y      = batch[:, 1:]                        # (B, L-1)
#         logits = model(x)                            # (B, L-1, Vocab)
#         return losses.cross_entropy(
#             logits.reshape(-1, cfg.vocab_size),
#             y.reshape(-1)
#         ).mean()

#     # forward + backward
#     loss_and_grad = nn.value_and_grad(model, loss_fn)

#     # ensure output dir exists
#     out_dir = pathlib.Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # training loop
#     for step, batch in enumerate(seq_iter):
#         # scheduler
#         opt.learning_rate = lr_schedule(
#             step,
#             base_lr=args.lr,
#             warmup=args.warmup,
#             total=args.total_steps
#         )

#         # compute loss & gradients, apply update
#         loss, grads = loss_and_grad(batch)
#         opt.update(model, grads)

#         # periodic logging
#         if step % 100 == 0:
#             seq_iter.write(f"{step:6d} | loss {float(loss):.3f}")

#         # checkpoint every 5k steps
#         if step and step % 5000 == 0:
#             ck = out_dir / f"ckpt_{step:06d}.safetensors"
#             model.save_weights(str(ck))            # <— convert to str
#             seq_iter.write(f"saved {ck}")

#     # final checkpoint
#     final_ck = out_dir / "ckpt_final.safetensors"
#     model.save_weights(str(final_ck))
#     print(f"✅ saved final checkpoint {final_ck}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Train a GPT-style LM with MLX only (no NumPy), with debug logs:

• Loads a JSONL dataset of {"text": ...}
• Tokenizes with your SentencePiece model
• Packs into (context_size+1)-length samples: inputs → targets
• Streams shuffled minibatches via MLX (mx.random)
• Uses AdamW, cosine LR schedule, tqdm, and periodic checkpoints
"""

import argparse
import json
import math
import time
import pathlib

import datasets
import mlx.core         as mx
import mlx.nn           as nn
import mlx.nn.losses    as losses
import mlx.optimizers  as optim
from tqdm              import tqdm

from model.model       import Transformer, TransformerConfig
from tokenizer         import sp_tokenizer


def to_samples(ids: mx.array, context_size: int) -> mx.array:
    """
    Reshape a 1D array of token-ids into shape (N, context_size+1)
    where each row is [x_0 ... x_{L-1}, x_1 ... x_L].
    """
    window = context_size + 1
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
    Yield (batch_size, context+1) slices forever, shuffling each epoch.
    """
    N = samples.shape[0]
    key = mx.random.key(seed)
    perm = mx.random.permutation(N, key=key)
    idx = 0
    while True:
        if idx + batch_size > N:
            key, = mx.random.split(key, 1)
            perm = mx.random.permutation(N, key=key)
            idx = 0
        batch_idx = perm[idx : idx + batch_size]
        yield samples[batch_idx]
        idx += batch_size


def lr_schedule(step, *, base_lr, warmup, total_steps):
    if step < warmup:
        return base_lr * step / warmup
    pct = (step - warmup) / (total_steps - warmup)
    # cosine decays down to 0.1×
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pct)))


def get_args():
    p = argparse.ArgumentParser("MLX GPT-LM trainer (debug)")
    p.add_argument("--config",           required=True, help="Path to model/config.json")
    p.add_argument("--dataset",          required=True, help="JSONL file, one {'text':...} per line")
    p.add_argument("--tokenizer",        required=True, help="SentencePiece .model file")
    p.add_argument("--out",              required=True, help="Directory to save checkpoints")
    p.add_argument("--context_size",     type=int,   default=512, help="Sequence length")
    p.add_argument("--batch_size",       type=int,   default=4,   help="Per-step batch size")
    p.add_argument("--total_steps",      type=int,   default=8000, help="Total training iterations")
    p.add_argument("--lr",               type=float, default=2e-4, help="Base learning rate")
    p.add_argument("--warmup",           type=int,   default=5000, help="Linear warmup iters")
    p.add_argument("--steps_report",     type=int,   default=100,  help="Log every N steps")
    p.add_argument("--steps_checkpoint", type=int,   default=5000, help="Checkpoint every N steps")
    return p.parse_args()


def main():
    args = get_args()
    print(f"[DEBUG] Parsed args: {args}")

    # 1) load SentencePiece
    print(f"[DEBUG] Loading tokenizer from {args.tokenizer}")
    tok = sp_tokenizer.load(args.tokenizer)

    # 2) load & tokenize dataset
    print(f"[DEBUG] Loading JSONL dataset from {args.dataset}")
    ds = datasets.load_dataset("json", data_files={"train": args.dataset}, split="train")
    print(f"[DEBUG] Tokenizing...")
    ds = ds.map(lambda ex: {"ids": tok.encode(ex["text"])}, remove_columns=["text"])
    print(f"[DEBUG] Tokenization done: {len(ds)} examples")

    # 3) flatten all IDs → 1D array, fast
    t0 = time.perf_counter()
    id_arrays = [mx.array(ids, dtype=mx.int32) for ids in ds["ids"]]
    ids_arr = mx.concatenate(id_arrays, axis=0)
    t1 = time.perf_counter()
    print(f"[DEBUG] Flatten+concatenate took {(t1-t0):.2f}s; total tokens = {ids_arr.shape[0]}")

    # 4) pack into samples of length context_size+1
    print(f"[DEBUG] Packing into context+1 samples with context_size={args.context_size}")
    samples = to_samples(ids_arr, args.context_size)
    print(f"[DEBUG] Produced {samples.shape[0]} samples of length {args.context_size+1}")

    # 5) build our MLX batch iterator
    train_iter = iterate_batches(samples, args.batch_size, seed=42)
    print(f"[DEBUG] Built batch iterator (batch_size={args.batch_size})")

    # 6) load model config & initialize
    print(f"[DEBUG] Instantiating model from config {args.config}")
    cfg   = TransformerConfig(**json.load(open(args.config)))
    model = Transformer(cfg)
    print(f"[DEBUG] Model instantiated.")

    # 7) optimizer
    optim_obj = optim.AdamW(learning_rate=args.lr, weight_decay=0.1)
    print(f"[DEBUG] Optimizer (AdamW) created with lr={args.lr}")

    # 8) prepare loss+grad function
    def loss_fn(mdl, batch):
        x = batch[:, :-1]  # (B, L)
        y = batch[:, 1:]   # (B, L)
        logits = mdl(x)    # (B, L, V)
        return losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1)
        ).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    print(f"[DEBUG] loss_and_grad function ready")

    # ensure output dir exists
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Checkpoints will be saved to {out_dir}")

    # 9) training loop
    losses_buf = []
    t_start = time.perf_counter()
    for step_idx in range(1, args.total_steps + 1):
        # update LR
        optim_obj.learning_rate = lr_schedule(
            step_idx, base_lr=args.lr, warmup=args.warmup, total_steps=args.total_steps
        )

        batch = next(train_iter)
        loss, grads = loss_and_grad(model, batch)
        optim_obj.update(model, grads)

        losses_buf.append(float(loss))

        # report
        if step_idx % args.steps_report == 0:
            t_now = time.perf_counter()
            avg = sum(losses_buf) / len(losses_buf)
            print(
                f"[{step_idx:5d}/{args.total_steps}] "
                f"loss={avg:.3f}  it/s={(args.steps_report/(t_now-t_start)):.2f}"
            )
            losses_buf.clear()
            t_start = t_now

        # checkpoint
        if step_idx % args.steps_checkpoint == 0:
            ck = out_dir / f"ckpt_{step_idx:06d}.safetensors"
            model.save_weights(str(ck))
            print(f"[DEBUG] Saved checkpoint {ck}")

    # final
    final_ck = out_dir / "ckpt_final.safetensors"
    model.save_weights(str(final_ck))
    print(f"✅ Done — saved final checkpoint {final_ck}")


if __name__ == "__main__":
    main()