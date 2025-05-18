#!/usr/bin/env python3
"""
Download TinyStories (v2) and save it as a JSONL compatible with clean_code.py
"""
import argparse, json, pathlib
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")        # single split
    ap.add_argument("--output", required=True,
                    help="Path to write tiny_stories.jsonl")
    args = ap.parse_args()

    ds = load_dataset(args.repo, split=args.split)     # no auth / gate
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for rec in ds:
            # Each record has "story" field (one short story)
            story = rec.get("text", "").strip()  
            if story:                                  # drop blanks
                f.write(json.dumps({"text": story}) + "\n")

    print(f"✅  Saved {len(ds):,} stories → {out}")

if __name__ == "__main__":
    main()