#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer for the TinyStories (or any) corpus.

Usage example
-------------
python tokenizer/train_tokenizer.py \
    --input data/clean_text.jsonl \
    --model-dir tokenizer \
    --model-type bpe \
    --vocab-size 50096
"""

import argparse
import pathlib
import sentencepiece as spm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True,
                    help="Path to cleaned corpus (.txt or .jsonl)")
    ap.add_argument("--model-dir",  required=True,
                    help="Directory to write <prefix>.model / .vocab")
    ap.add_argument("--model-type", default="bpe",
                    choices=("bpe", "unigram"),
                    help="SentencePiece algorithm")
    ap.add_argument("--vocab-size", type=int, default=50_096)
    args = ap.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = model_dir / f"py50k_{args.model_type}"

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=str(prefix),
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        unk_id=3, bos_id=1, eos_id=2, pad_id=0,
        bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        verbose=True
    )

    print(f"✅ tokenizer saved to {prefix}.model / {prefix}.vocab")


if __name__ == "__main__":
    main()