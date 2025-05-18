
#!/usr/bin/env python3
"""Deduplicate & filter Python files.

Uses fastdup for near‑duplicate removal + simple heuristics to strip tests and large files.
"""
import argparse, pathlib, json, re, sys
from fastdup import fastdup

def is_test(path):
    return bool(re.search(r'(test_|_test|tests/)', str(path)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw code dir")
    ap.add_argument("--output", required=True, help="output JSONL file with {{'text': code}}")
    args = ap.parse_args()

    raw_dir = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.rglob("*.py"))
    print(f"Scanning {len(files)} .py files")

    # simple heuristics
    kept = []
    for fp in files:
        if is_test(fp): continue
        txt = fp.read_text(errors="ignore")
        if txt.count("\n") < 6 or len(txt) > 20000: continue
        kept.append(txt)

    print("running fastdup deduplication…")
    unique = fastdup.remove_duplicates(kept, min_unique_lines=50)

    with out_path.open("w") as f:
        for code in unique:
            f.write(json.dumps({"text": code}) + "\n")

    print("✅ wrote", out_path)

if __name__ == "__main__":
    main()
