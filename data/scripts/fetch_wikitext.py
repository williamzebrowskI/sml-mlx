#!/usr/bin/env python3
"""
Fetch a subset of WikiText-103 and dump it as JSONL with {"text": …} entries.
Skips empty lines and section headers (=== … ===).
"""
import argparse, json, pathlib
from datasets import load_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset",      default="train",
                   help="which split of wikitext-103-v1 to fetch")
    p.add_argument("--n_articles",  type=int, default=50000,
                   help="how many non-empty paragraphs to save")
    p.add_argument("--output",      required=True,
                   help="path to write JSONL (one {'text':…} per line)")
    args = p.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the correct config name here:
    ds = load_dataset(
        "wikitext",
        "wikitext-103-v1",       # <— change to 'wikitext-103-raw-v1' if you prefer the raw split
        split=args.subset,
        streaming=True
    )

    count = 0
    with out_path.open("w") as fout:
        for rec in ds:
            txt = rec["text"].strip()
            # skip blank lines and header markers
            if not txt or txt.startswith("="):
                continue
            fout.write(json.dumps({"text": txt}) + "\n")
            count += 1
            if count >= args.n_articles:
                break

    print(f"✅  Saved {count} paragraphs → {out_path}")

if __name__ == "__main__":
    main()