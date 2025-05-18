#!/usr/bin/env python3
"""
Tokenise JSONL → space-separated ids with a tqdm bar.
Writes data/encoded.txt (stdout) while all logs go to stderr.
"""

import sys, json, pathlib
from tqdm import tqdm
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from tokenizer import sp_tokenizer

jsonl_path, tok_path = sys.argv[1:3]
out_path = "data/encoded.txt"

tok = sp_tokenizer.load(tok_path)

# quick line-count for a nice progress bar
with open(jsonl_path, "rb") as f:
    n_lines = sum(1 for _ in f)

with open(jsonl_path) as fin, open(out_path, "w") as fout:
    bar = tqdm(fin, total=n_lines, desc="encode", unit="line", smoothing=0.05)
    for row in bar:
        ids = tok.encode(json.loads(row)["text"]) + [tok.eos_id]
        fout.write(" ".join(map(str, ids)) + "\n")

print(f"✅ wrote {out_path}", file=sys.stderr)