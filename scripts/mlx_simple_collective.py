#!/usr/bin/env python3

from __future__ import annotations

import argparse
import faulthandler
import os
import socket
import sys

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ring")
    parser.add_argument("--expected-world", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    # If a collective wedges, print Python stacks and exit instead of hanging forever.
    faulthandler.dump_traceback_later(args.timeout, exit=True)

    rank_env = os.getenv("MLX_RANK", "?")
    print(
        f"[pre-init] host={socket.gethostname()} rank_env={rank_env} "
        f"exe={sys.executable}",
        flush=True,
    )

    group = mx.distributed.init(backend=args.backend, strict=True)
    rank = int(group.rank())  # type: ignore[attr-defined]
    world = int(group.size())  # type: ignore[attr-defined]

    print(
        f"[post-init] host={socket.gethostname()} rank={rank} world={world}",
        flush=True,
    )

    if args.expected_world and world != args.expected_world:
        raise SystemExit(f"expected world={args.expected_world}, got {world}")

    x = mx.array([rank + 1], dtype=mx.int32)
    print(f"[before-all-sum] rank={rank} value={int(x.item())}", flush=True)
    y = mx.distributed.all_sum(x, group=group, stream=mx.cpu)
    mx.eval(y)
    print(f"[after-all-sum] rank={rank} value={int(y.item())}", flush=True)

    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
