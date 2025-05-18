
#!/usr/bin/env python3
"""Train a 50 K SentencePiece unigram tokenizer on cleaned Python corpus."""
import argparse, sentencepiece as spm, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=50096)
    args = ap.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = model_dir / "py50k"

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=str(prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        split_digits=True,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )

    print("✅ tokenizer saved to", model_dir)

if __name__ == "__main__":
    main()
