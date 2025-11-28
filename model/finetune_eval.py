#!/usr/bin/env python
"""
Evaluate a fine-tuned TinyGPLM checkpoint on a single prompt.

Intended for models fine-tuned on User/Assistant-style chat data
(e.g., WildChat), but works with any checkpoint you point it at.

Usage example:

    PYTHONPATH=. python -m model.eval_finetune \
      --ckpt model/checkpoints_chat_sft/chat_sft_final.safetensors \
      --spm-model tokenizer/fineweb_spm/spm.model \
      --seq-len 3072 \
      --max-new-tokens 200 \
      --prompt "User: Hello there!\\nAssistant: "
"""

import argparse
import os
import importlib

import mlx.core as mx

# Import model + helpers; keep a module handle so globals (like TOK) update after set_tokenizer().
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


def eval_checkpoint(
    ckpt_path: str,
    spm_model: str,
    seq_len: int,
    max_new_tokens: int,
    prompt: str,
):
    # 1. Initialize tokenizer globals
    set_tokenizer(spm_model)

    # 2. Get vocab size from TOK after set_tokenizer()
    vocab_size = mdl.TOK.vocab_size

    # 3. Build model with same config as training
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

    # 4. Load checkpoint (fine-tuned or base)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ok = load_safetensors_model(ckpt_path, model)
    if not ok:
        raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}")
    mx.eval(model.parameters())
    print(f"[eval-ft] Loaded checkpoint: {ckpt_path}")
    print(f"[eval-ft] Using seq_len={seq_len}, max_new_tokens={max_new_tokens}")
    print(f"[eval-ft] Vocab size={vocab_size}\n")

    print("=======================================")
    print(f"[PROMPT]\n{repr(prompt)}")
    print("=======================================\n")

    # 1) Greedy (with your repetition penalty)
    greedy = generate_greedy_nocache(model, prompt, max_new_tokens)
    print(f"[greedy]\n{greedy}\n---\n")

    # If you ever want to re-enable sampling sweeps, you can uncomment this
    # and adjust the grids as needed.

    # # 2) Top-k variants (tuned grid)
    # topk_variants = [
    #     ("top-k (k=5,   T=0.5)", 0.5, 5),
    #     ("top-k (k=10,  T=0.6)", 0.6, 10),
    #     ("top-k (k=20,  T=0.7)", 0.7, 20),
    #     ("top-k (k=40,  T=0.8)", 0.8, 40),
    #     ("top-k (k=80,  T=1.0)", 1.0, 80),
    #     ("top-k (k=160, T=1.1)", 1.1, 160),
    # ]
    #
    # for desc, T, K in topk_variants:
    #     txt = generate_topk(
    #         model,
    #         prompt,
    #         max_new_tokens=max_new_tokens,
    #         temperature=T,
    #         top_k=K,
    #     )
    #     print(f"[{desc}]\n{txt}\n---\n")
    #
    # # 3) Top-p variants (tuned grid)
    # topp_variants = [
    #     ("top-p (p=0.70, T=0.6)", 0.6, 0.70),
    #     ("top-p (p=0.80, T=0.7)", 0.7, 0.80),
    #     ("top-p (p=0.90, T=0.8)", 0.8, 0.90),
    #     ("top-p (p=0.95, T=1.0)", 1.0, 0.95),
    #     ("top-p (p=0.98, T=1.1)", 1.1, 0.98),
    # ]
    #
    # for desc, T, P in topp_variants:
    #     txt = generate_topp(
    #         model,
    #         prompt,
    #         max_new_tokens=max_new_tokens,
    #         temperature=T,
    #         top_p=P,
    #     )
    #     print(f"[{desc}]\n{txt}\n---\n")


def main():
    parser = argparse.ArgumentParser("Single-prompt eval for fine-tuned TinyGPLM checkpoints")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to fine-tuned .safetensors checkpoint (e.g. chat_sft_final.safetensors)",
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
        "--prompt",
        type=str,
        required=True,
        help="Single prompt string to evaluate (e.g. 'User: ...\\nAssistant: ').",
    )

    args = parser.parse_args()

    eval_checkpoint(
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()