#!/usr/bin/env python3
"""
Fetch a slice of OpenWebText from HuggingFace and dump it as JSONL.

Example
-------
python fetch_openwebtext.py \
    --dataset Skylion007/openwebtext \
    --split   train \
    --n_paragraphs 1_000_000 \
    --output  data/raw/openwebtext_1M.jsonl
"""

import argparse, json, pathlib, textwrap
from datasets import load_dataset

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream OpenWebText and save a clean JSONL file."
    )
    p.add_argument("--dataset",  default="Skylion007/openwebtext",
                   help="HF dataset repo (e.g. 'Skylion007/openwebtext').")
    p.add_argument("--split",    default="train",
                   help="Dataset split to stream (default: train).")
    p.add_argument("--n_paragraphs", type=int, default=1_000_000,
                   help="How many non-blank paragraphs to save.")
    p.add_argument("--output",   required=True,
                   help="Target JSONL path.")
    return p.parse_args()

def main() -> None:
    args = cli()
    out  = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,   # no local download; iterates over the tar shards
    )

    keep = 0
    with out.open("w") as f:
        for rec in ds:
            txt = rec["text"].strip()
            if not txt:                 # skip blanks
                continue
            f.write(json.dumps({"text": txt}) + "\n")
            keep += 1
            if keep >= args.n_paragraphs:
                break

    print(f"✅  Saved {keep:,} paragraphs → {out}")

if __name__ == "__main__":
    main()