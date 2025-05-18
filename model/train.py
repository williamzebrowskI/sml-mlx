
#!/usr/bin/env python3
"""Minimal MLX training loop for 200 M‑param Python model."""
import argparse, json, math, pathlib, itertools, time
import mlx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from tokenizer import sp_tokenizer  # local helper
from model.model import Transformer, TransformerConfig

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--total_steps", type=int, default=500000)
    return ap.parse_args()

def main():
    args = get_args()
    cfg = TransformerConfig(**json.load(open(args.config)))
    tok = sp_tokenizer.load(args.tokenizer)

    model = Transformer(cfg)
    model.to(dtype=getattr(mlx, cfg.dtype))

    ds = load_dataset("json", data_files=args.dataset, streaming=True)["train"]
    def encode(ex):
        ids = tok.encode(ex["text"]) + [tok.eos_id]
        return {"ids": ids}
    ds = ds.map(encode).shuffle(buffer_size=10000)

    def batch_iter(ds):
        buff = []
        for ex in ds:
            buff += ex["ids"]
            while len(buff) >= args.seq_len:
                yield buff[:args.seq_len]
                buff = buff[args.seq_len:]

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_fn(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        pct = (step - args.warmup) / (args.total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * pct)) * 0.1 + args.lr * 0.9

    step = 0
    acc_loss = 0.0
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in itertools.islice(batch_iter(ds), args.total_steps * args.grad_accum):
        x = mlx.array(seq[:-1], dtype=mlx.int32)
        y = mlx.array(seq[1:], dtype=mlx.int32)
        logits = model(x[None, :])
        loss = nn.cross_entropy(logits.reshape(-1, cfg.vocab_size), y)
        loss.backward()
        acc_loss += loss.item()
        if (step + 1) % args.grad_accum == 0:
            opt.step(lr_fn(step // args.grad_accum))
            opt.zero_grad()
            if step % (100 * args.grad_accum) == 0:
                print(f"{step//args.grad_accum:>7} | loss {acc_loss/args.grad_accum:.3f}")
                acc_loss = 0.0
            if step % (10000 * args.grad_accum) == 0 and step:
                ckpt = out_dir / f"ckpt_{step//args.grad_accum:06d}.safetensors"
                model.save_weights(ckpt)
                print("saved", ckpt)
        step += 1
        if step // args.grad_accum >= args.total_steps:
            break

if __name__ == "__main__":
    main()
