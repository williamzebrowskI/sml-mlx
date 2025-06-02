#!/usr/bin/env python3
"""
Fetch ~2 billion tokens worth of OpenWebText from HuggingFace and dump it as JSONL.

Example
-------
python fetch_openwebtext.py \
    --dataset       Skylion007/openwebtext \
    --split         train \
    --token_budget  2000000000 \
    --output        data/raw/openwebtext_2B.jsonl
"""

import argparse
import json
import pathlib
import sys

# Ensure that “tokenizer/” (and other project modules) are on PYTHONPATH
# (this script lives in data/scripts/, so go up two levels to the repo root)
proj_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(proj_root))

from tokenizer import sp_tokenizer  # now importable

from datasets import load_dataset


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream OpenWebText and save ~N tokens to a JSONL file."
    )
    p.add_argument(
        "--dataset",
        default="Skylion007/openwebtext",
        help="HF dataset repo (e.g. 'Skylion007/openwebtext')."
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (default: train)."
    )
    p.add_argument(
        "--token_budget",
        type=int,
        default=2_000_000_000,
        help="Approximate number of tokens to save (e.g. 2 billion)."
    )
    p.add_argument(
        "--output",
        required=True,
        help="Target JSONL path."
    )
    return p.parse_args()


def main() -> None:
    args = cli()
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load SentencePiece tokenizer
    tok = sp_tokenizer.load("tokenizer/spm.model")

    # 2) Stream the chosen split of OpenWebText
    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,   # iterate over shards without full download
    )

    total_tokens = 0
    kept_docs    = 0

    with out_path.open("w") as fout:
        for rec in ds:
            text = rec["text"].strip()
            if not text:
                continue

            # Tokenize and count
            ids = tok.encode(text)
            n_tok = len(ids)
            if total_tokens + n_tok > args.token_budget:
                # Exiting once we've reached ~2 billion tokens
                break

            # Write one JSON line per document
            fout.write(json.dumps({"text": text}) + "\n")
            total_tokens += n_tok
            kept_docs    += 1

            # Print progress every 1 million documents
            if kept_docs % 1_000_000 == 0:
                print(f"⏳ Kept {kept_docs:,} docs, ~{total_tokens:,} tokens…")

    giB = total_tokens / 1_024**3
    print(f"✅  Saved {kept_docs:,} documents → {out_path}")
    print(f"    Total tokens: {total_tokens:,} (~{giB:.2f} GiB)")


if __name__ == "__main__":
    main()