#!/usr/bin/env python3
"""
Download ≤N GB of the BigCode *stackoverflow-clean* corpus and save it
as newline-delimited JSON objects:  {"text": …}

Only the “content” field is kept (question + answers concatenated);
rows with fewer than 30 characters are skipped.
"""

import argparse, json, pathlib, sys
from datasets import load_dataset

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True,
                    help="target .jsonl file (will be created)")
    ap.add_argument("--max_gb", type=float, default=1.8,
                    help="stop after this many **compressed** GB (~≈ RAM)")
    ap.add_argument("--split", default="train",
                    help="dataset split (train / validation / test)")
    args = ap.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ≈ 1.05 GiB compressed for the default train split (10.4 M rows)  [oai_citation:0‡Hugging Face](https://huggingface.co/datasets/bigcode/stackoverflow-clean)
    ds = load_dataset("bigcode/stackoverflow-clean",
                      split=args.split,
                      streaming=True)

    budget  = int(args.max_gb * (1024 ** 3))         # bytes
    written = 0
    kept    = 0

    with out_path.open("w") as fout:
        for row in ds:
            txt = row["content"].strip()
            if len(txt) < 30:          # drop very short snippets / boiler-plate
                continue
            rec = {"text": txt}
            line = json.dumps(rec) + "\n"
            size = len(line.encode("utf-8"))
            if written + size > budget:
                break
            fout.write(line)
            written += size
            kept    += 1

    print(f"✅  Saved {kept:,} posts → {out_path}  "
          f"({written / (1024**3):.2f} GiB)")

if __name__ == "__main__":
    main()