#!/usr/bin/env python3
"""
Append extra code data (Python + Bash) to an existing JSONL file.

• Streams  up to --max_gb  of *CodeSearchNet* subsets that are public.
• Finds the column that really contains code (schema differs per language).
• Keeps only rows ≥30 chars, writes as {"text": …} — one JSON per line.
"""

import argparse, json, pathlib, sys
from datasets import load_dataset

# ───────────────────────── helpers ──────────────────────────
def find_code_field(example: dict) -> str | None:
    """Return the first key that plausibly holds source code"""
    for cand in ("code", "func_code_string", "content", "snippet"):
        txt = example.get(cand)
        if isinstance(txt, str) and any(ch in txt for ch in ("{", "def ", "#", "\n")):
            return cand
    return None

# ───────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",   required=True,
                    help="existing *.jsonl file to append to")
    ap.add_argument("--max_gb",   type=float, default=2.0,
                    help="budget in *compressed* GiB for all languages together")
    ap.add_argument("--langs",    nargs="+", default=["python", "javascript"],
                    help="CodeSearchNet languages to stream")
    args = ap.parse_args()

    out     = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    budget  = int(args.max_gb * 1024**3)       # bytes, compressed
    written = 0
    kept    = 0

    with out.open("a") as fout:
        for lang in args.langs:
            print(f"► streaming CodeSearchNet {lang} …", file=sys.stderr)
            ds = load_dataset(
                    "code_search_net",
                    name=lang,
                    split="train",
                    streaming=True,
                    trust_remote_code=True)      # avoids the prompt

            for row in ds:
                field = find_code_field(row)
                if not field:
                    continue
                txt = row[field].strip()
                if len(txt) < 30:
                    continue

                line = json.dumps({"text": txt}) + "\n"
                size = len(line.encode())
                if written + size > budget:
                    print("⏹  budget reached", file=sys.stderr)
                    return
                fout.write(line)
                written += size
                kept    += 1

    giB = written / 1024**3
    print(f"✅  Added {kept:,} snippets  ({giB:.2f} GiB)  →  {out}")

if __name__ == "__main__":
    main()