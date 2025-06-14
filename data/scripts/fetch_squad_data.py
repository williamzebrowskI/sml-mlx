#!/usr/bin/env python3
"""
scripts/fetch_squad_jsonl.py

Download SQuAD v1.1 and convert to JSONL for fine-tuning.

Outputs:
  data/squad_train.jsonl
  data/squad_valid.jsonl
"""

import os
import json
from datasets import load_dataset

def main():
    # ensure output directory exists
    os.makedirs("data", exist_ok=True)

    # load SQuAD v1.1
    ds = load_dataset("squad")

    # write train split
    with open("data/squad_train.jsonl", "w") as ftrain:
        for ex in ds["train"]:
            question = ex["question"].strip()
            # take the first answer text
            answer   = ex["answers"]["text"][0].strip()
            ftrain.write(json.dumps({"question": question, "answer": answer}) + "\n")

    # write validation split
    with open("data/squad_valid.jsonl", "w") as fval:
        for ex in ds["validation"]:
            question = ex["question"].strip()
            answer   = ex["answers"]["text"][0].strip()
            fval.write(json.dumps({"question": question, "answer": answer}) + "\n")

    print("Wrote data/squad_train.jsonl and data/squad_valid.jsonl")

if __name__ == "__main__":
    main()