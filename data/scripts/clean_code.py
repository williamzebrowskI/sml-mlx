#!/usr/bin/env python3
"""
Deduplicate & filter text or code files WITHOUT fastdup.

* If --input is a .jsonl file      ➜ each line must be {"text": ...}
* If --input is a directory        ➜ crawls *.py files (original behaviour)

Outputs a JSONL with {"text": cleaned_string}.
"""

import argparse, pathlib, json, re, sys
from typing import Iterable

def read_py_files(raw_dir: pathlib.Path) -> Iterable[str]:
    def is_test(p: pathlib.Path) -> bool:
        return bool(re.search(r'(test_|_test|tests/)', str(p)))
    for fp in raw_dir.rglob("*.py"):
        if is_test(fp):
            continue
        txt = fp.read_text(errors="ignore")
        if 6 <= txt.count("\n") <= 20000:   # keep non-tiny / non-huge
            yield txt

def read_jsonl(file_path: pathlib.Path) -> Iterable[str]:
    with file_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                txt = (obj.get("text") or obj.get("story") or "").strip()
                if txt:
                    yield txt
            except json.JSONDecodeError:
                continue   # skip malformed lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="directory of .py files OR a *.jsonl file")
    ap.add_argument("--output", required=True,
                    help="destination JSONL with {'text': ...}")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input).expanduser()
    out_path = pathlib.Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1️⃣  Collect candidate texts
    if in_path.is_dir():
        texts = list(read_py_files(in_path))
    elif in_path.suffix == ".jsonl":
        texts = list(read_jsonl(in_path))
    else:
        sys.exit("ERROR: --input must be a directory or .jsonl file")

    print(f"Collected {len(texts):,} texts before dedup")

    # 2️⃣  Hash-set dedup -- the Pythonic way
    unique, seen = [], set()
    for t in texts:
        h = hash(t)
        if h not in seen:
            seen.add(h)
            unique.append(t)

    print(f"Kept {len(unique):,} unique items")

    # 3️⃣  Write output JSONL
    with out_path.open("w") as f:
        for t in unique:
            f.write(json.dumps({"text": t}) + "\n")

    print("✅ wrote", out_path)

if __name__ == "__main__":
    main()