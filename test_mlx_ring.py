#!/usr/bin/env python3
# test_mlx_ring.py
import os, time, socket, json, math, argparse
import mlx.core as mx

def barrier():
    mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32)))

def fmt_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu","cpu"], default="gpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sizes", default="1e4,1e6,4e6", help="comma-separated float counts (elements)")
    args = ap.parse_args()

    # device
    try:
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        dev = "gpu" if args.device == "gpu" else "cpu"
    except Exception as e:
        dev = f"cpu_fallback({e})"

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()

    host = socket.gethostname()
    env_info = {
        "rank": rank,
        "size": size,
        "host": host,
        "device": dev,
        "MLX_HOSTS": os.getenv("MLX_HOSTS"),
        "MLX_PORT": os.getenv("MLX_PORT"),
    }
    print(f"[R{rank}] init: {json.dumps(env_info)}", flush=True)

    # quick barrier
    barrier()

    # sanity: small all_sum
    x = mx.array([rank + 1], dtype=mx.float32)
    y = mx.distributed.all_sum(x)
    mx.eval(y)
    expected = float(size * (size + 1) // 2)
    ok_small = float(y.item()) == expected
    print(f"[R{rank}] small all_sum -> {float(y.item())} (expected {expected}) ok={ok_small}", flush=True)
    barrier()

    # sizes to test
    sizes = []
    for s in args.sizes.split(","):
        s = s.strip()
        if not s:
            continue
        sizes.append(int(float(s)))

    # timing runs (float32)
    results = []
    for n in sizes:
        payload_bytes = n * 4  # float32
        # initialize distinct values per rank to make check meaningful
        arr = mx.ones((n,), dtype=mx.float32) * (rank + 1)
        # warmup
        _ = mx.distributed.all_sum(arr); mx.eval(_)
        barrier()
        times = []
        for _rep in range(args.repeats):
            t0 = time.perf_counter()
            out = mx.distributed.all_sum(arr)
            mx.eval(out)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        # correctness check on a few elements
        val0 = float(out[0].item())
        ok = math.isclose(val0, expected, rel_tol=1e-4, abs_tol=1e-4)
        # simple per-rank "throughput" (note: all-reduce traffic is > payload)
        avg_t = sum(times) / len(times)
        thr = payload_bytes / avg_t  # bytes/sec (payload)
        print(f"[R{rank}] n={n:,} ({fmt_bytes(payload_bytes)}), avg={avg_t*1e3:.2f} ms, payload_rate≈{fmt_bytes(thr)}/s, ok={ok}", flush=True)
        results.append({
            "n": n,
            "payload_bytes": payload_bytes,
            "avg_sec": avg_t,
            "payload_bytes_per_sec": thr,
            "ok": ok,
        })
        barrier()

    if rank == 0:
        summary = {
            "hosts": os.getenv("MLX_HOSTS"),
            "size": size,
            "device": dev,
            "results": results,
        }
        print("\n[R0] SUMMARY:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()