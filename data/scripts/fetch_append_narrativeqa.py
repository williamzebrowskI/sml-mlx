#!/usr/bin/env python3
"""
fetch_append_narrativeqa.py ──────────────────────────────────────────────
Append long-answer QA pairs from NarrativeQA to an existing JSONL
corpus that already contains WebGPT + TriviaQA.

Example
-------
python fetch_append_narrativeqa.py \
    --train_jsonl data/fine_tune/combined_train.jsonl \
    --valid_jsonl data/fine_tune/combined_valid.jsonl \
    --n_long 30000 \
    --min_len 15 \
    --valid_frac 0.05 \
    --seed 42
"""

import argparse, json, random, pathlib, sys
from pathlib import Path
from datasets import load_dataset, disable_caching

disable_caching()

# ─────────────────────────────────────────────────────────────────────────
def iter_narrativeqa():
    """
    Yield (question, answer) tuples from the *default* NarrativeQA split.
    Each dataset example contains a *list* of Q/A dicts.
    """
    ds = load_dataset("sapienzanlp/narrativeqa", split="train")  # only “default”
    for ex in ds:
        for qa in ex["question"]:
            q = qa.get("question", "").strip()
            answers = qa.get("answers", [])
            a0 = answers[0].strip() if answers else ""
            if q and a0:
                yield q, a0

def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            yield obj["question"], obj["answer"]

def export_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q, a in records:
            f.write(json.dumps({"question": q, "answer": a},
                               ensure_ascii=False) + "\n")
    print(f"Wrote {len(records):,} → {path}")

# ─────────────────────────────────────────────────────────────────────────
def fetch_long_pairs(n_long, min_len, seed):
    """
    Sample up to `n_long` NarrativeQA pairs whose answers meet `min_len`.
    """
    rnd = random.Random(seed)
    candidates = [
        (q, a) for q, a in iter_narrativeqa()
        if len(a) >= min_len
    ]
    rnd.shuffle(candidates)
    chosen = candidates[:n_long]
    print(f"  • selected {len(chosen):,} long-answer pairs "
          f"(≥{min_len} chars) from NarrativeQA")
    return chosen

# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--valid_jsonl", required=True)
    ap.add_argument("--n_long", type=int, default=30000,
                    help="# long-answer pairs to add")
    ap.add_argument("--min_len", type=int, default=15,
                    help="minimum answer length (characters)")
    ap.add_argument("--valid_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    print("Scanning NarrativeQA …")
    long_pairs = fetch_long_pairs(args.n_long, args.min_len, args.seed)

    # ---------------------------------------------------------------------
    print("Loading current QA corpus …")
    tr_path = Path(args.train_jsonl)
    va_path = Path(args.valid_jsonl)
    train_pairs = list(read_jsonl(tr_path))
    valid_pairs = list(read_jsonl(va_path))
    print(f"  • existing: {len(train_pairs):,} train / {len(valid_pairs):,} valid")

    # append & reshuffle ---------------------------------------------------
    all_pairs = train_pairs + valid_pairs + long_pairs
    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * args.valid_frac)
    valid_new, train_new = all_pairs[:n_val], all_pairs[n_val:]
    print(f"After appending + reshuffle:")
    print(f"  • train : {len(train_new):,}")
    print(f"  • valid : {len(valid_new):,}")

    # overwrite files ------------------------------------------------------
    export_jsonl(train_new, tr_path)
    export_jsonl(valid_new, va_path)
    print("✅ Done – appended NarrativeQA long answers.")

if __name__ == "__main__":
    main()