#!/usr/bin/env python3
"""
Evaluate *one* checkpoint against a small prompt suite.

It prints a Markdown table that you can paste into your training-notes
or commit message.

Usage
-----
PYTHONPATH=. python scripts/eval_suite.py \
    --checkpoint model/checkpoints/ckpt_041000.safetensors \
    --config     model/config.json \
    --tokenizer  tokenizer/spm.model
"""
import argparse, json, textwrap, pathlib
import mlx.core as mx

from model.model   import Transformer, TransformerConfig
from tokenizer     import sp_tokenizer
from model.eval    import run_mlx_sample   # reuse the sampler

# --------------------------------------------------------------------------- #
# 10-prompt evaluation suite                                                  #
# --------------------------------------------------------------------------- #

# ▸ scripts/eval_suite.py  (replace SUITE = [...] block)
SUITE = [
    # factual
    dict(prompt="Define **entropy** in one concise sentence:",
         args=dict(sampling_method="top-p", top_p=0.92, temperature=0.6,
                   rep_penalty=1.1, ngram_block=3, max_new_tokens=40, key_seed=1)),
    dict(prompt="Explain in one short paragraph how photosynthesis works.",
         args=dict(sampling_method="top-k", top_k=50, temperature=0.45,
                   rep_penalty=1.2, ngram_block=3, max_new_tokens=80, key_seed=2)),
    dict(prompt="Q: What causes the phases of the Moon?\nA:",
         args=dict(sampling_method="top-p", top_p=0.9, temperature=0.4,
                   rep_penalty=1.15, ngram_block=3, max_new_tokens=60, key_seed=3)),

    # creative
    dict(prompt="Write a haiku about autumn rain and fading lanterns.",
         args=dict(sampling_method="top-k", top_k=20, temperature=1.0,
                   rep_penalty=1.05, ngram_block=2, max_new_tokens=35, key_seed=4)),
    dict(prompt="Invent a new sport suitable for low-gravity lunar colonies.",
         args=dict(sampling_method="top-k", top_k=80, temperature=0.9,
                   rep_penalty=1.1, ngram_block=4, max_new_tokens=120, key_seed=5)),
    dict(prompt="On the outskirts of Neo-Tokyo, the abandoned observatory still whispered about the stars. Tonight,",
         args=dict(sampling_method="top-p", top_p=0.92, temperature=0.8,
                   rep_penalty=1.15, ngram_block=3, max_new_tokens=120, key_seed=6)),

    # dialogue / instruction following
    dict(prompt="User: I'm feeling stressed about exams. Any quick tips?\nAssistant:",
         args=dict(sampling_method="top-p", top_p=0.85, temperature=0.6,
                   rep_penalty=1.2, ngram_block=3, max_new_tokens=80, key_seed=7)),
    dict(prompt="Alice: Did you remember to calibrate the quantum scanner?\nBob:",
         args=dict(sampling_method="top-k", top_k=40, temperature=0.6,
                   rep_penalty=1.15, ngram_block=3, max_new_tokens=90, key_seed=8)),

    # code
    dict(prompt="```python\ndef fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
         args=dict(sampling_method="greedy", temperature=0,
                   rep_penalty=1.3, ngram_block=4, max_new_tokens=60, key_seed=9)),

    # bullet list
    dict(prompt="Summarize the benefits of regular exercise in three bullet points:",
         args=dict(sampling_method="top-k", top_k=30, temperature=0.45,
                   rep_penalty=1.2, ngram_block=3, max_new_tokens=60, key_seed=10)),
    # cloze
    dict(prompt="The capital of France is",
         args=dict(sampling_method="top-p", top_p=0.8, temperature=0.4,
                   rep_penalty=1.05, ngram_block=2, max_new_tokens=10, key_seed=11)),
    # numeric reasoning (cheap check)
    dict(prompt="What is 17 × 23?",
         args=dict(sampling_method="top-p", top_p=0.9, temperature=0.3,
                   rep_penalty=1.1, ngram_block=2, max_new_tokens=15, key_seed=12)),
]

SEQ_LEN = 512   # context length fed into the model

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_model(ckpt: str, cfg_path: str):
    cfg   = TransformerConfig(**json.load(open(cfg_path)))
    model = Transformer(cfg)
    model.load_weights(ckpt)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser("Mini evaluation suite")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config",     default="model/config.json")
    ap.add_argument("--tokenizer",  required=True)
    args = ap.parse_args()

    tok   = sp_tokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, args.config)

    rows = []
    for item in SUITE:
        gen = run_mlx_sample(
            model, tok,
            prompt   = item["prompt"],
            seq_len  = SEQ_LEN,
            **item["args"],
        )
        excerpt = textwrap.shorten(
            gen[len(item["prompt"]):].strip(),
            width=120, placeholder=" …"
        )
        rows.append((item["prompt"].split("\n")[0][:32] + "…", excerpt))

    ck_name = pathlib.Path(args.checkpoint).stem
    print(f"### Sample outputs – `{ck_name}`\n")
    print("| prompt | completion |")
    print("|--------|------------|")
    for ptxt, comp in rows:
        print(f"| *{ptxt}* | {comp} |")


if __name__ == "__main__":
    main()