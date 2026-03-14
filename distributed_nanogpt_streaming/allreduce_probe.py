#!/usr/bin/env python3
"""Minimal distributed allreduce probe for MLX ring debugging."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train import _all_sum
from distributed_nanogpt_streaming.train_gpt2_streaming import _allreduce_tree_chunked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ring")
    parser.add_argument("--stream-mode", type=str, default="cpu", choices=["cpu", "default"])
    parser.add_argument("--elems", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--chunk-mb", type=int, default=0)
    args = parser.parse_args()

    try:
        group = mx.distributed.init(backend=args.backend, strict=True)
    except TypeError:
        group = mx.distributed.init(backend=args.backend)

    rank = int(group.rank() if callable(getattr(group, "rank", None)) else group.rank)
    world = int(group.size() if callable(getattr(group, "size", None)) else group.size)
    dtype = getattr(mx, args.dtype)
    print(
        f"[rank {rank}] world={world} elems={args.elems} dtype={args.dtype} chunk_mb={args.chunk_mb}",
        flush=True,
    )

    arr = mx.ones((args.elems,), dtype=dtype) * float(rank + 1)
    mx.eval(arr)
    print(f"[rank {rank}] ready", flush=True)

    t0 = time.perf_counter()
    if args.chunk_mb > 0:
        out = _allreduce_tree_chunked(
            {"x": arr},
            world,
            stream_mode=args.stream_mode,
            sync_bytes=args.chunk_mb * 1024 * 1024,
        )["x"]
    else:
        out = _all_sum(arr, stream_mode=args.stream_mode)
    mx.eval(out)
    dt = time.perf_counter() - t0
    print(f"[rank {rank}] done dt={dt:.3f}s first={float(out[0].item()):.3f}", flush=True)


if __name__ == "__main__":
    main()
