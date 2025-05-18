#!/usr/bin/env python3
"""
Download CodeSearchNet-Python (and optionally LCC_Python),
merge, de-dup, and emit a single JSONL for the rest of the pipeline.
"""
import argparse, json, pathlib
from datasets import load_dataset, concatenate_datasets

def load_csn():
    return load_dataset("AhmedSSoliman/CodeSearchNet-Python",
                        split="train", trust_remote_code=True)

def load_lcc():
    return load_dataset("microsoft/LCC_python",
                        split="train", trust_remote_code=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_lcc", action="store_true",
                    help="also include LCC_Python (~55 M tokens)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds_list = [load_csn()]
    if args.with_lcc:
        ds_list.append(load_lcc())
    ds = concatenate_datasets(ds_list)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    kept = 0
    with out.open("w") as f:
        for rec in ds:
            code = (rec.get("code") or
                    rec.get("code_tokens") or
                    rec.get("context") or "")
            code = code.strip()
            if len(code) < 20 or code in seen:
                continue
            seen.add(code)
            f.write(json.dumps({"text": code}) + "\n")
            kept += 1
    print(f"✅ wrote {kept:,} unique snippets to {out}")

if __name__ == "__main__":
    main()