
#!/usr/bin/env python3
"""Quick HumanEval / MBPP harness."""
import argparse, json, pathlib, random, gc, math
import mlx
from model.model import Transformer, TransformerConfig
from tokenizer import sp_tokenizer
from eval_suites import humaneval, mbpp   # assume tiny wrappers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    tok = sp_tokenizer.load(args.tokenizer)
    cfg = TransformerConfig(**json.load(open('model/config.json')))
    model = Transformer(cfg)
    model.load_weights(args.checkpoint)
    model.eval()

    results = {
        "humaneval": humaneval.run(model, tok),
        "mbpp": mbpp.run(model, tok)
    }
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
