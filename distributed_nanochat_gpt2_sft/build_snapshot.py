#!/usr/bin/env python3
"""Build a local mixed SFT snapshot from the nanochat-style task recipe."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distributed_nanochat_gpt2_sft.tasks import build_cursor, parse_source_specs
from train.train import _load_config_defaults


def _weighted_pick(specs: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    total = sum(max(1, int(item["weight"])) for item in specs)
    threshold = rng.randrange(total)
    running = 0
    for item in specs:
        running += max(1, int(item["weight"]))
        if threshold < running:
            return item
    return specs[-1]


def _write_mix(
    *,
    raw_sources: Any,
    output_path: Path,
    count: int,
    seed: int,
    shuffle_buffer: int,
    trust_remote_code: bool,
) -> None:
    specs = parse_source_specs(raw_sources, default_shuffle_buffer=shuffle_buffer)
    source_states: List[Dict[str, Any]] = []
    for i, spec in enumerate(specs):
        cursor = build_cursor(
            spec,
            rank=0,
            world=1,
            seed=seed + i * 1_000_003,
            trust_remote_code=trust_remote_code,
        )
        source_states.append({"cursor": cursor, "weight": int(spec.weight), "kind": spec.kind})

    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(count):
            src = _weighted_pick(source_states, rng)
            row = src["cursor"].next_conversation()
            payload = {"messages": row["messages"], "source": src["kind"], "index": idx}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local SFT snapshot JSONL files")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/config.json",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/data/train_snapshot.jsonl",
    )
    parser.add_argument(
        "--val-out",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/data/val_snapshot.jsonl",
    )
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = _load_config_defaults(str(Path(args.config).resolve()))
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    shuffle_buffer = int(cfg.get("shuffle_buffer", 1024))

    _write_mix(
        raw_sources=cfg.get("train_sources", []),
        output_path=Path(args.train_out).resolve(),
        count=args.train_count,
        seed=args.seed,
        shuffle_buffer=shuffle_buffer,
        trust_remote_code=trust_remote_code,
    )
    _write_mix(
        raw_sources=cfg.get("val_sources", []),
        output_path=Path(args.val_out).resolve(),
        count=args.val_count,
        seed=args.seed + 10_000_000,
        shuffle_buffer=shuffle_buffer,
        trust_remote_code=trust_remote_code,
    )
    print(f"[done] train={args.train_out} val={args.val_out}")


if __name__ == "__main__":
    main()
