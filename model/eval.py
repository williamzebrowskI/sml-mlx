#!/usr/bin/env python
"""
Offline eval script for your TinyGPLM checkpoints.

Usage (single Mac):

    PYTHONPATH=. python -m model.eval \
        --ckpt model/checkpoints_spm/ckpt_987500.safetensors \
        --spm-model tokenizer/fineweb_spm/spm.model \
        --seq-len 3072 \
        --max-new-tokens 200
"""

import argparse
import os

import mlx.core as mx

# Import model + helpers; prefer absolute import for `python -m model.eval`.
# Keep a module handle so globals like TOK reflect updates after set_tokenizer().
import importlib
try:
    mdl = importlib.import_module("model.model")
except ImportError:
    mdl = importlib.import_module("model")

TinyGPLM = mdl.TinyGPLM
TinyGPConfig = mdl.TinyGPConfig
set_tokenizer = mdl.set_tokenizer
load_safetensors_model = mdl.load_safetensors_model
generate_greedy_nocache = mdl.generate_greedy_nocache
generate_topk = mdl.generate_topk
generate_topp = mdl.generate_topp

# ---------- Prompt set ----------

DEFAULT_PROMPTS = [
    ("continuation", "The old library smelled of dust and secrets. As I opened the hidden door, "),
    ("story",        "On a rainy afternoon, a cardboard box was left on the porch. Inside, "),
    ("facts",        "Saturn's rings are made of ice and rock. They formed because "),
    ("code",         "def fibonacci(n: int) -> list[int]:\n    "),
    ("reasoning",    "Alice has 12 marbles. She gives Bob 4 and finds 3 more in her pocket. Now she "),
    ("howto",        "For a smooth cup of pour-over coffee, start by "),
]


# ---------- Eval driver ----------

def eval_checkpoint(
    ckpt_path: str,
    spm_model: str,
    seq_len: int,
    max_new_tokens: int,
    prompts=None,
):
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    # 1. Initialize tokenizer globals
    set_tokenizer(spm_model)

    # IMPORTANT: get vocab size from the module-global TOK after set_tokenizer()
    vocab_size = mdl.TOK.vocab_size

    # 2. Build model with same config as training
    cfg = TinyGPConfig(
        vocab_size=vocab_size,
        d_model=384,
        n_heads=6,
        n_layers=12,
        max_seq=seq_len,
        max_grad_norm=1.0,
    )
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())

    # 3. Load checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ok = load_safetensors_model(ckpt_path, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}")
    mx.eval(model.parameters())
    print(f"[eval] Loaded checkpoint: {ckpt_path}")

    print(f"[eval] Using seq_len={seq_len}, max_new_tokens={max_new_tokens}")
    print(f"[eval] Vocab size={vocab_size}\n")

    # Define a richer grid of top-k / top-p variants
    topk_variants = [
        ("top-k (k=10,  T=0.6)", 0.6, 10),
        ("top-k (k=20,  T=0.7)", 0.7, 20),
        ("top-k (k=40,  T=0.8)", 0.8, 40),
        ("top-k (k=80,  T=1.0)", 1.0, 80),
        ("top-k (k=160, T=1.1)", 1.1, 160),
    ]

    topp_variants = [
        ("top-p (p=0.80, T=0.7)", 0.7, 0.80),
        ("top-p (p=0.90, T=0.8)", 0.8, 0.90),
        ("top-p (p=0.95, T=1.0)", 1.0, 0.95),
        ("top-p (p=0.98, T=1.1)", 1.1, 0.98),
    ]

    # 4. Run multiple decoding variants per prompt
    for label, prompt in prompts:
        print(f"==============================")
        print(f"[EVAL] prompt_label={label!r}")
        print(f"Prompt: {repr(prompt)}")
        print(f"==============================\n")

        # 1) Greedy (with your repetition penalty)
        greedy = generate_greedy_nocache(model, prompt, max_new_tokens)
        print(f"[greedy]\n{greedy}\n---\n")

        # 2) Top-k variants
        for desc, T, K in topk_variants:
            txt = generate_topk(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=T,
                top_k=K,
            )
            print(f"[{desc}]\n{txt}\n---\n")

        # 3) Top-p variants
        for desc, T, P in topp_variants:
            txt = generate_topp(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=T,
                top_p=P,
            )
            print(f"[{desc}]\n{txt}\n---\n")

        print("\n\n")  # extra space between prompts


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser("Offline eval for TinyGPLM checkpoints")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to .safetensors checkpoint (e.g. ckpt_987500.safetensors)",
    )
    parser.add_argument(
        "--spm-model",
        type=str,
        required=True,
        help="Path to SentencePiece model (e.g. tokenizer/fineweb_spm/spm.model)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=3072,
        help="Max sequence length to use for eval (must match or be <= training max_seq)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate per sample",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "Optional path to a text file with custom prompts.\n"
            "Each line will be used as a prompt label:prompt_text."
        ),
    )

    args = parser.parse_args()

    prompts = None
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    label, text = line.split(":", 1)
                else:
                    label, text = "custom", line
                prompts.append((label.strip(), text.strip()))

    eval_checkpoint(
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        prompts=prompts,
    )


if __name__ == "__main__":
    main()
