#!/usr/bin/env python3
"""
fetch_combined_qa_jsonl.py
Merge WebGPT-comparisons + TriviaQA into train/valid JSONL.
"""

import json, random, argparse
from pathlib import Path
from datasets import load_dataset, disable_caching

disable_caching()

# ---------- WebGPT helpers (unchanged) ----------
def smart_question_string(q_field):
    if isinstance(q_field, str):
        return q_field.strip()
    if isinstance(q_field, dict):
        for k in ("full_text", "text"):
            if k in q_field and isinstance(q_field[k], str):
                return q_field[k].strip()
        for v in q_field.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def shorter_answer(a0, a1):  # choose shorter, non-empty
    a0, a1 = a0.strip(), a1.strip()
    return a0 if a0 and (not a1 or len(a0) <= len(a1)) else a1

def iter_webgpt():
    ds = load_dataset("openai/webgpt_comparisons", split="train", streaming=True)
    for ex in ds:
        q = smart_question_string(ex["question"])
        a = shorter_answer(ex["answer_0"], ex["answer_1"])
        if q and a:
            yield q, a

# ---------- TriviaQA helpers (fixed) ----------
def iter_triviaqa():
    # use 'unfiltered.nocontext' to avoid large context blobs
    ds = load_dataset("trivia_qa", "unfiltered.nocontext", split="train", streaming=True)
    for ex in ds:
        q = ex["question"].strip()
        aliases = [t.strip() for t in ex["answer"]["aliases"] if t.strip()]
        if not aliases:
            continue
        yield q, min(aliases, key=len)

# ---------- I/O ----------
def export_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q, a in records:
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records):,} → {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/fine_tune")
    ap.add_argument("--valid_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    print("Collecting WebGPT QA …")
    webgpt = list(iter_webgpt())
    print(f"  kept {len(webgpt):,}")

    print("Collecting TriviaQA QA …")
    trivia = list(iter_triviaqa())
    print(f"  kept {len(trivia):,}")

    pairs = webgpt + trivia
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.valid_frac)
    valid, train = pairs[:n_val], pairs[n_val:]

    out = Path(args.output_dir)
    export_jsonl(train, out / "combined_train.jsonl")
    export_jsonl(valid, out / "combined_valid.jsonl")

if __name__ == "__main__":
    main()