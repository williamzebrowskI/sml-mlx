# #!/usr/bin/env python3
# """
# Train a SentencePiece tokenizer for the TinyStories (or any) corpus.

# Usage example
# -------------
# python tokenizer/train_tokenizer.py \
#     --input data/clean_text.jsonl \
#     --model-dir tokenizer \
#     --model-type bpe \
#     --vocab-size 50096
# """

# import argparse
# import pathlib
# import sentencepiece as spm


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input",      required=True,
#                     help="Path to cleaned corpus (.txt or .jsonl)")
#     ap.add_argument("--model-dir",  required=True,
#                     help="Directory to write <prefix>.model / .vocab")
#     ap.add_argument("--model-type", default="bpe",
#                     choices=("bpe", "unigram"),
#                     help="SentencePiece algorithm")
#     ap.add_argument("--vocab-size", type=int, default=50_096)
#     args = ap.parse_args()

#     model_dir = pathlib.Path(args.model_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)

#     prefix = model_dir / f"py50k_{args.model_type}"

#     spm.SentencePieceTrainer.Train(
#         input=args.input,
#         model_prefix=str(prefix),
#         model_type=args.model_type,
#         vocab_size=args.vocab_size,
#         character_coverage=1.0,
#         unk_id=3, bos_id=1, eos_id=2, pad_id=0,
#         bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
#         verbose=True
#     )

#     print(f"✅ tokenizer saved to {prefix}.model / {prefix}.vocab")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer on your raw JSONL text.

Writes out:
  <model-dir>/spm.model
  <model-dir>/spm.vocab
"""
import argparse
import pathlib
import sentencepiece as spm

def main():
    p = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    p.add_argument(
        "--input", required=True,
        help="Path to JSONL (one {'text':…} per line) to train on"
    )
    p.add_argument(
        "--model-dir", required=True,
        help="Directory where spm.model & spm.vocab will be saved"
    )
    p.add_argument(
        "--vocab-size", type=int, default=50096,
        help="Number of tokens in the SentencePiece vocab"
    )
    p.add_argument(
        "--model-type", choices=["bpe","unigram","word","char"],
        default="bpe",
        help="Which SentencePiece model algorithm to use"
    )
    p.add_argument(
        "--character-coverage", type=float, default=1.0,
        help="Amount of characters covered; 1.0 means 100%%"
    )
    args = p.parse_args()

    out_dir = pathlib.Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / "spm"
    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=str(prefix),
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        split_by_whitespace=True
    )

    print(f"✅ tokenizer saved to {prefix}.model (+ .vocab)")

if __name__ == "__main__":
    main()