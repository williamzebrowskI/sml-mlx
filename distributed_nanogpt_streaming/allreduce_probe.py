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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ring")
    parser.add_argument("--stream-mode", type=str, default="cpu", choices=["cpu", "default"])
    parser.add_argument("--elems", type=int, required=True)
    args = parser.parse_args()

    try:
        group = mx.distributed.init(backend=args.backend, strict=True)
    except TypeError:
        group = mx.distributed.init(backend=args.backend)

    rank = int(group.rank() if callable(getattr(group, "rank", None)) else group.rank)
    world = int(group.size() if callable(getattr(group, "size", None)) else group.size)
    print(f"[rank {rank}] world={world} elems={args.elems}", flush=True)

    arr = mx.ones((args.elems,), dtype=mx.float32) * float(rank + 1)
    mx.eval(arr)
    print(f"[rank {rank}] ready", flush=True)

    t0 = time.perf_counter()
    out = _all_sum(arr, stream_mode=args.stream_mode)
    mx.eval(out)
    dt = time.perf_counter() - t0
    print(f"[rank {rank}] done dt={dt:.3f}s first={float(out[0].item()):.3f}", flush=True)


if __name__ == "__main__":
    main()
