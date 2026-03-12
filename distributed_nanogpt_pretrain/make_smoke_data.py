#!/usr/bin/env python3
"""Create a tiny GPT-2 token dataset for distributed smoke tests."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    out_dir = Path("/Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/data/smoke_gpt2")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1337)
    vocab_size = 50257
    padded_vocab_size = 50304
    train = rng.integers(0, vocab_size, size=262144, dtype=np.uint16)
    val = rng.integers(0, vocab_size, size=32768, dtype=np.uint16)
    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    meta = {
        "dataset_name": "synthetic_smoke",
        "tokenizer_name": "gpt2",
        "vocab_size": vocab_size,
        "padded_vocab_size": padded_vocab_size,
        "token_dtype": "uint16",
        "train_tokens": int(train.shape[0]),
        "val_tokens": int(val.shape[0]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(out_dir)


if __name__ == "__main__":
    main()
