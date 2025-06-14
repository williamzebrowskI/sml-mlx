#!/usr/bin/env python3
"""
prepare_squad_v2_jsonl.py

Download and preprocess the SQuAD v2 dataset from Hugging Face into JSONL files
with fields: id, question, context, answers (list of {text, answer_start}).
"""

import argparse
import os
import json
from datasets import load_dataset  # Hugging Face datasets library

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SQuAD v2 JSONL files for fine-tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fine_tune",
        help="Directory to write JSONL files (train.jsonl, valid.jsonl)",
    )
    parser.add_argument(
        "--filter_unanswerable",
        action="store_true",
        help="If set, exclude examples with no answers (unanswerable) from the output",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the SQuAD v2 dataset
    # This will download if not cached locally
    dataset = load_dataset("rajpurkar/squad_v2")  # train has ~130K, validation ~11K  [oai_citation:4‡huggingface.co](https://huggingface.co/datasets/rajpurkar/squad_v2?utm_source=chatgpt.com)

    for split in ["train", "validation"]:
        split_name = "train" if split == "train" else "valid"
        out_path = os.path.join(args.output_dir, f"squad_v2_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            for example in dataset[split]:
                # example fields: id, title, context, question, answers
                answers = example.get("answers", {})
                texts = answers.get("text", [])
                starts = answers.get("answer_start", [])
                # If filtering unanswerable, skip if no answer texts
                if args.filter_unanswerable and (not texts or len(texts)==0):
                    continue
                # Build answer list of dicts
                ans_list = []
                for txt, st in zip(texts, starts):
                    ans_list.append({"text": txt, "answer_start": st})
                out_obj = {
                    "id": example.get("id"),
                    "question": example.get("question", "").strip(),
                    "context": example.get("context", "").strip(),
                    "answers": ans_list,  # may be empty list
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        print(f"Wrote {split} split to {out_path}, filter_unanswerable={args.filter_unanswerable}")

if __name__ == "__main__":
    main()