#!/usr/bin/env python3
"""
MLX Distributed Connectivity Check
=================================

This script verifies that MLX distributed can form a multi-host group and that
basic collectives and point-to-point ops work across all ranks.

It is intended for 4-node setups (e.g. 4 Mac Studios), but works for any N.

Key idea: we call `mx.distributed.init(..., strict=True)` so the script FAILS
if your env isn't actually configured for distributed (MLX otherwise falls back
to a singleton group).

Backends & setup (high level)
-----------------------------

ring backend (TCP, often used for Thunderbolt rings):
  - Requires env vars:
      - MLX_RANK
      - MLX_HOSTFILE  (path to a JSON file describing the ring links)
  - Recommended: use MLX's helpers to generate config + launch.

jaccl backend (RDMA / ibverbs):
  - Requires env vars:
      - MLX_RANK
      - MLX_JACCL_COORDINATOR  (ip:port for coordinator)
      - MLX_IBV_DEVICES        (path to a JSON file describing IBV devices)

Recommended way to launch (MLX utilities)
-----------------------------------------

1) Create a hostfile for `mlx.launch` (ssh + IPs) with `mlx.distributed_config`.
   See MLX docs for details:
     https://ml-explore.github.io/mlx/build/html/usage/distributed.html
     https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html

2) Launch one process per host with `mlx.launch`, which sets MLX_* env vars:

   mlx.launch --hostfile hosts.json --backend ring -- \
     python scripts/mlx_dist_check.py --backend ring --expected-world 4

Manual launch (ring) looks like:
  export MLX_HOSTFILE=/path/to/ring_links.json
  # On each host, set a unique rank 0..N-1 and run the script:
  MLX_RANK=0 python scripts/mlx_dist_check.py --backend ring --expected-world 4
  MLX_RANK=1 python scripts/mlx_dist_check.py --backend ring --expected-world 4
  ...

Exit codes:
  0 = success
  2 = init/config error
  3 = collective/p2p test failure
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional
import mlx.core as mx


def _stream_from_flag(flag: str):
    # Most MLX distributed transports are host-side; forcing CPU streams tends
    # to make debugging simpler and avoids surprises from device selection.
    if flag == "cpu":
        return mx.cpu
    return None


def _die(code: int, msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(code)


def _env_snapshot() -> dict[str, str]:
    keys = [
        "MLX_RANK",
        "MLX_HOSTFILE",
        "MLX_JACCL_COORDINATOR",
        "MLX_IBV_DEVICES",
        "MLX_LOCAL_RANK",
        "MLX_WORLD_SIZE",
    ]
    out = {}
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            out[k] = v
    return out


@dataclass(frozen=True)
class DistInfo:
    group: object
    rank: int
    world: int


def init_group(backend: str, strict: bool) -> DistInfo:
    try:
        g = mx.distributed.init(backend=backend, strict=strict)
    except Exception as e:
        _die(
            2,
            "\n".join(
                [
                    f"[init] FAILED backend={backend!r} strict={strict}: {e}",
                    f"[init] env={_env_snapshot()}",
                    "[init] Tip: for multi-host, prefer `mlx.launch` (it sets MLX_* env vars).",
                ]
            ),
        )
    try:
        rank = int(g.rank())  # type: ignore[attr-defined]
        world = int(g.size())  # type: ignore[attr-defined]
    except Exception as e:
        _die(2, f"[init] group missing rank/size? {e!r}")
    return DistInfo(group=g, rank=rank, world=world)


def _all_sum_scalar(di: DistInfo, stream, *, verbose: bool) -> None:
    # mx.array doesn't accept a stream arg; use an op that does so we can force CPU.
    x = mx.ones((1,), dtype=mx.float32, stream=stream) * float(di.rank + 1)
    y = mx.distributed.all_sum(x, group=di.group, stream=stream)
    mx.eval(y)
    got = float(y.item())
    expected = float(di.world * (di.world + 1) / 2)
    if abs(got - expected) > 1e-3:
        _die(
            3,
            f"[all_sum] FAIL rank={di.rank} got={got} expected={expected} world={di.world}",
        )
    if verbose and di.rank == 0:
        print(f"[all_sum] ok expected={expected} got={got}", flush=True)


def _all_gather_ranks(di: DistInfo, stream, *, verbose: bool) -> None:
    x = mx.ones((1,), dtype=mx.int32, stream=stream) * int(di.rank)
    y = mx.distributed.all_gather(x, group=di.group, stream=stream)
    mx.eval(y)
    ranks = [int(v) for v in y.tolist()]
    expected = list(range(di.world))
    if ranks != expected:
        _die(
            3,
            f"[all_gather] FAIL rank={di.rank} got={ranks} expected={expected}",
        )
    if verbose and di.rank == 0:
        print(f"[all_gather] ok ranks={ranks}", flush=True)


def _all_gather_hostnames(di: DistInfo, stream, *, max_len: int, verbose: bool) -> None:
    host = socket.gethostname().encode("utf-8", errors="replace")[:max_len]
    buf = list(host) + [0] * (max_len - len(host))
    x = mx.array(buf, dtype=mx.uint8).reshape(1, max_len)
    y = mx.distributed.all_gather(x, group=di.group, stream=stream)
    mx.eval(y)
    if di.rank != 0:
        return
    rows = y.tolist()
    names: list[str] = []
    for row in rows:
        b = bytes(int(v) for v in row)
        b = b.split(b"\x00", 1)[0]
        try:
            names.append(b.decode("utf-8", errors="replace"))
        except Exception:
            names.append(repr(b))
    if verbose:
        print("[hosts] rank mapping:", flush=True)
        for r, n in enumerate(names):
            print(f"  rank {r}: {n}", flush=True)


def _p2p_ring(di: DistInfo, stream, *, verbose: bool) -> None:
    if di.world <= 1:
        if verbose and di.rank == 0:
            print("[p2p] skipped (world_size=1)", flush=True)
        return

    # Neighbor-only p2p is supported even for the ring backend.
    src = (di.rank - 1) % di.world
    dst = (di.rank + 1) % di.world

    payload = mx.ones((1,), dtype=mx.int32, stream=stream) * int(di.rank)
    sent = mx.distributed.send(payload, dst=dst, group=di.group, stream=stream)
    recvd = mx.distributed.recv_like(payload, src=src, group=di.group, stream=stream)
    mx.eval(sent, recvd)

    got = int(recvd.item())
    if got != src:
        _die(
            3,
            f"[p2p] FAIL rank={di.rank} recv_from={src} got={got} (sent_to={dst})",
        )
    if verbose and di.rank == 0:
        print("[p2p] ok (neighbor ring send/recv)", flush=True)


def _bandwidth_all_sum(di: DistInfo, stream, *, tensor_mb: int, iters: int, verbose: bool) -> None:
    if tensor_mb <= 0:
        return

    n_f32 = (tensor_mb * 1024 * 1024) // 4
    x = mx.ones((n_f32,), dtype=mx.float32, stream=stream)

    # Warmup
    y = mx.distributed.all_sum(x, group=di.group, stream=stream)
    mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        y = mx.distributed.all_sum(x, group=di.group, stream=stream)
        mx.eval(y)
    dt = time.perf_counter() - t0

    avg = dt / max(1, iters)
    size_bytes = tensor_mb * 1024 * 1024
    gbps = (size_bytes / max(1e-9, avg)) / (1024**3)

    if di.rank == 0:
        print(
            f"[perf] all_sum payload={tensor_mb}MB iters={iters} avg={avg*1e3:.2f}ms approx={gbps:.2f}GiB/s",
            flush=True,
        )
    elif verbose:
        print(
            f"[perf] rank={di.rank} avg={avg*1e3:.2f}ms",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Check multi-host MLX distributed connectivity")
    ap.add_argument(
        "--backend",
        type=str,
        default=os.getenv("MLX_BACKEND", "ring"),
        help="Distributed backend: ring, jaccl, mpi, nccl, any",
    )
    ap.add_argument(
        "--expected-world",
        type=int,
        default=None,
        help="Fail if group size != this value (e.g. 4).",
    )
    ap.add_argument(
        "--non-strict",
        action="store_true",
        help="If set, allow MLX to fall back to a singleton group (NOT recommended).",
    )
    ap.add_argument(
        "--stream",
        choices=["default", "cpu"],
        default="cpu",
        help="Which stream/device to use for tensors + communication.",
    )
    ap.add_argument(
        "--no-p2p",
        action="store_true",
        help="Skip send/recv ring test.",
    )
    ap.add_argument(
        "--hostnames",
        action="store_true",
        help="All-gather hostnames and print rank mapping on rank 0.",
    )
    ap.add_argument(
        "--hostname-len",
        type=int,
        default=64,
        help="Max bytes per hostname to gather (only used with --hostnames).",
    )
    ap.add_argument(
        "--tensor-mb",
        type=int,
        default=0,
        help="If >0, run a small all_sum perf test with this payload size (MB).",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Iterations for perf test (used with --tensor-mb).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="More logs (rank 0 prints summary even without this).",
    )

    args = ap.parse_args()
    strict = not args.non_strict
    stream = _stream_from_flag(args.stream)

    di = init_group(args.backend, strict=strict)
    if di.rank == 0:
        print(
            f"[init] ok backend={args.backend} world={di.world} strict={strict} stream={args.stream}",
            flush=True,
        )
        if args.verbose:
            print(f"[init] env={_env_snapshot()}", flush=True)

    if args.expected_world is not None and di.world != int(args.expected_world):
        _die(
            3,
            "\n".join(
                [
                    f"[world] FAIL expected_world={args.expected_world} got={di.world}",
                    f"[world] backend={args.backend} strict={strict} env={_env_snapshot()}",
                    "[world] Tip: if strict=False, MLX can silently fall back to world=1.",
                ]
            ),
        )

    # Collectives act as synchronization points.
    _all_sum_scalar(di, stream, verbose=args.verbose)
    _all_gather_ranks(di, stream, verbose=args.verbose)
    if args.hostnames:
        _all_gather_hostnames(di, stream, max_len=max(8, int(args.hostname_len)), verbose=args.verbose)
    if not args.no_p2p:
        _p2p_ring(di, stream, verbose=args.verbose)

    _bandwidth_all_sum(
        di,
        stream,
        tensor_mb=int(args.tensor_mb),
        iters=max(1, int(args.iters)),
        verbose=args.verbose,
    )

    if di.rank == 0:
        print("[ok] distributed connectivity looks good", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
