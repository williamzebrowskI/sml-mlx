#!/usr/bin/env python3
"""Prepare nanoGPT-style train.bin/val.bin token files from HF text or token datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def _iter_texts(dataset, text_field: str):
    for ex in dataset:
        if not isinstance(ex, dict):
            continue
        text = ex.get(text_field)
        if isinstance(text, str):
            text = text.strip()
            if text:
                yield text


def _iter_token_lists(dataset, token_field: str):
    for ex in dataset:
        if not isinstance(ex, dict):
            continue
        tokens = ex.get(token_field)
        if isinstance(tokens, list) and tokens:
            yield [int(tok) for tok in tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GPT-2 token memmaps from a Hugging Face dataset")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--token-field", type=str, default="")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit-train-docs", type=int, default=0)
    parser.add_argument("--limit-val-docs", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(append_eos=None)
    parser.add_argument("--append-eos", action="store_true", dest="append_eos")
    parser.add_argument("--no-append-eos", action="store_false", dest="append_eos")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    eos_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    padded_vocab_size = int(math.ceil(vocab_size / 64.0) * 64)
    token_dtype = np.uint16 if padded_vocab_size <= np.iinfo(np.uint16).max else np.uint32
    append_eos = args.append_eos if args.append_eos is not None else (not args.token_field)

    def iter_examples(dataset):
        if args.token_field:
            for ids in _iter_token_lists(dataset, args.token_field):
                if append_eos:
                    ids = ids + [eos_id]
                yield np.asarray(ids, dtype=token_dtype)
        else:
            for text in _iter_texts(dataset, args.text_field):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if append_eos:
                    ids.append(eos_id)
                yield np.asarray(ids, dtype=token_dtype)

    def write_split(split_name: str, dataset, path: Path, limit_docs: int) -> tuple[int, int]:
        docs = 0
        tokens = 0
        with open(path, "wb") as f:
            for arr in iter_examples(dataset):
                arr.tofile(f)
                docs += 1
                tokens += int(arr.shape[0])
                if docs % 10000 == 0:
                    print(f"[{split_name}] docs={docs:,} tokens={tokens:,}", flush=True)
                if limit_docs > 0 and docs >= limit_docs:
                    break
        return docs, tokens

    if args.val_split:
        train_ds = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.train_split,
            streaming=True,
            trust_remote_code=args.trust_remote_code,
        )
        val_ds = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.val_split,
            streaming=True,
            trust_remote_code=args.trust_remote_code,
        )
        train_docs, train_tokens = write_split("train", train_ds, out_dir / "train.bin", args.limit_train_docs)
        val_docs, val_tokens = write_split("val", val_ds, out_dir / "val.bin", args.limit_val_docs)
    else:
        ds = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.train_split,
            streaming=True,
            trust_remote_code=args.trust_remote_code,
        )
        train_path = out_dir / "train.bin"
        val_path = out_dir / "val.bin"
        train_docs = train_tokens = val_docs = val_tokens = 0
        val_every = max(1, int(round(1.0 / max(args.val_ratio, 1e-9))))
        with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
            for i, arr in enumerate(iter_examples(ds), start=1):
                target = f_val if (args.val_ratio > 0 and i % val_every == 0) else f_train
                arr.tofile(target)
                if target is f_val:
                    val_docs += 1
                    val_tokens += int(arr.shape[0])
                else:
                    train_docs += 1
                    train_tokens += int(arr.shape[0])
                if i % 10000 == 0:
                    print(
                        f"[stream] docs={i:,} train_tokens={train_tokens:,} val_tokens={val_tokens:,}",
                        flush=True,
                    )
                if args.limit_train_docs > 0 and train_docs >= args.limit_train_docs:
                    break

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "val_split": args.val_split or None,
        "text_field": args.text_field,
        "token_field": args.token_field or None,
        "tokenizer_name": args.tokenizer_name,
        "eos_token_id": eos_id,
        "append_eos": append_eos,
        "vocab_size": vocab_size,
        "padded_vocab_size": padded_vocab_size,
        "token_dtype": np.dtype(token_dtype).name,
        "train_docs": train_docs,
        "train_tokens": train_tokens,
        "val_docs": val_docs,
        "val_tokens": val_tokens,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
