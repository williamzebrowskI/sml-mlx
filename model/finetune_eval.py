#!/usr/bin/env python
"""
Evaluate a fine-tuned TinyGPLM checkpoint on a single prompt.

Intended for models fine-tuned on User/Assistant-style chat data
(e.g., Smol-SmolTalk, WildChat), but works with any checkpoint you point it at.

Usage example:

    PYTHONPATH=. python -m model.finetune_eval \
      --ckpt model/checkpoints_chat_sft/chat_sft_final.safetensors \
      --spm-model tokenizer/fineweb_spm/spm.model \
      --seq-len 3072 \
      --max-new-tokens 200 \
      --prompt "User: Hello there!\\nAssistant: "
"""

import argparse
import os
import importlib
import json
from typing import Optional, List

import mlx.core as mx
import numpy as np

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


def extract_assistant_only(full_text: str, prompt: str) -> str:
    """
    Extract only the assistant's reply from a decoded string.

    Handles both patterns:
      1) Prompt already includes 'Assistant: ':

         "User: ...\\nAssistant: <model tokens>"

      2) Prompt is just "User: ..." and the model *introduces* 'Assistant:':

         "User: ... I have a ... Assistant: <model tokens>"

    Strategy:
      - Find the *first* 'Assistant:'; return everything after it.
      - If no 'Assistant:', but the string starts with the prompt, strip the prompt.
      - Else, return the text as-is.
    """
    txt = full_text

    # Prefer to slice at the first 'Assistant:'
    first = txt.find("Assistant:")
    if first != -1:
        reply = txt[first + len("Assistant:") :]
        return reply.strip()

    # Fallback: if full text starts with the prompt, strip it off
    if prompt and txt.startswith(prompt):
        return txt[len(prompt) :].strip()

    # Fallback: nothing fancy
    return txt.strip()


def eval_checkpoint(
    ckpt_path: str,
    spm_model: str,
    seq_len: int,
    max_new_tokens: int,
    prompt: str,
    config_path: str | None = None,
    sweep: bool = False,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.5,
    penalty_window: int = 128,
    stop_phrases: list[str] | None = None,
    min_new_tokens: int = 0,
    force_max_new_tokens: bool = False,
):
    def _ban_token_id(logits, token_id: int):
        # logits: (1, vocab)
        if token_id is None:
            return logits
        vocab = int(logits.shape[-1])
        if token_id < 0 or token_id >= vocab:
            return logits
        mask = (mx.arange(vocab).reshape(1, vocab) == token_id)
        return mx.where(mask, -1e9, logits)

    def _greedy_local(
        model: TinyGPLM,
        text: str,
        max_new: int,
        rep: float,
        window: int,
        max_repeat: int = 8,
        stops: Optional[List[str]] = None,
        min_new: int = 0,
        force_max: bool = False,
    ):
        if stops is None:
            stops = ["\nUser:", "User:"]
        if force_max:
            stops = []
            max_repeat = 10**9
        # Inference prompt encoding: BOS + tokens, no EOS.
        ids = mdl.encode_prompt(text)[: model.cfg.max_seq]
        out_ids = ids[:]
        last_id = None
        repeat_run = 0
        for _ in range(max_new):
            x_ids = mx.array([out_ids[:model.cfg.max_seq]], dtype=mx.int32)
            logits = model.logits(x_ids)
            last_logits = logits[:, -1, :]
            if force_max or (len(out_ids) - len(ids)) < max(0, int(min_new)):
                last_logits = _ban_token_id(last_logits, mdl.TOK.eos_id)
            next_id = mdl._greedy_with_repetition_penalty(
                last_logits,
                generated_ids=out_ids,
                repetition_penalty=rep,
                penalty_window=window,
            )
            out_ids.append(next_id)
            repeat_run = repeat_run + 1 if next_id == last_id else 1
            last_id = next_id
            text_so_far = mdl.TOK.decode(out_ids)
            if (not force_max) and (
                next_id == mdl.TOK.eos_id
                or repeat_run >= max_repeat
                or any(text_so_far.endswith(s) for s in stops)
            ):
                break
        return mdl.TOK.decode(out_ids)

    def _sample_local(
        model: TinyGPLM,
        text: str,
        max_new: int,
        temp: float,
        top_k_val: Optional[int],
        top_p_val: Optional[float],
        rep: float,
        window: int,
        max_repeat: int = 8,
        stops: Optional[List[str]] = None,
        min_new: int = 0,
        force_max: bool = False,
    ):
        if stops is None:
            stops = ["\nUser:", "User:"]
        if force_max:
            stops = []
            max_repeat = 10**9
        # Inference prompt encoding: BOS + tokens, no EOS.
        ids = mdl.encode_prompt(text)[: model.cfg.max_seq]
        caches = None
        for tok in ids[:-1]:
            tok_id = mx.array([[tok]], dtype=mx.int32)
            _, caches = model.step(tok_id, caches)
        cur = mx.array([[ids[-1]]], dtype=mx.int32)
        out_ids = ids[:]
        last_id = None
        repeat_run = 0
        for _ in range(max_new):
            logits, caches = model.step(cur, caches)
            if force_max or (len(out_ids) - len(ids)) < max(0, int(min_new)):
                vocab = int(logits.shape[-1])
                logits = mx.where(
                    (mx.arange(vocab).reshape(1, 1, vocab) == mdl.TOK.eos_id),
                    -1e9,
                    logits,
                )
            next_id = mdl._sample_next_token(
                logits,
                temperature=temp,
                top_k=top_k_val,
                top_p=top_p_val,
                generated_ids=out_ids,
                repetition_penalty=rep,
                penalty_window=window,
            )
            out_ids.append(int(next_id))
            repeat_run = repeat_run + 1 if next_id == last_id else 1
            last_id = next_id
            text_so_far = mdl.TOK.decode(out_ids)
            if (not force_max) and (
                next_id == mdl.TOK.eos_id
                or repeat_run >= max_repeat
                or any(text_so_far.endswith(s) for s in stops)
            ):
                break
            cur = mx.array([[next_id]], dtype=mx.int32)
        return mdl.TOK.decode(out_ids)

    # 1. Initialize tokenizer globals
    set_tokenizer(spm_model)

    # 2. Load config overrides if provided
    cfg_kwargs = {
        "vocab_size": mdl.TOK.vocab_size,
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "max_seq": seq_len,
        "max_grad_norm": 1.0,
    }
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        for k in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq", "max_grad_norm"):
            if k in cfg_json:
                cfg_kwargs[k] = cfg_json[k]
        # Respect explicit seq_len flag if user passed it and it differs
        if seq_len:
            cfg_kwargs["max_seq"] = min(seq_len, cfg_kwargs.get("max_seq", seq_len))

    cfg = TinyGPConfig(**cfg_kwargs)
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    vocab_size = cfg.vocab_size  # for logging

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

    if not sweep:
        # Single-run mode: either sampled or greedy
        if temperature is not None or top_k is not None or top_p is not None:
            full = _sample_local(
                model,
                prompt,
                max_new_tokens,
                temp=temperature if temperature is not None else 0.7,
                top_k_val=top_k if top_k not in (None, 0) else None,
                top_p_val=top_p if top_p not in (None, 0) else None,
                rep=repetition_penalty,
                window=penalty_window,
                min_new=min_new_tokens,
                force_max=force_max_new_tokens,
            )
            reply = extract_assistant_only(full, prompt)
            print(f"[sampled full]\n{full}\n---\n")
            print(f"[sampled reply]\n{reply}\n---\n")
        else:
            full = _greedy_local(
                model,
                prompt,
                max_new_tokens,
                rep=repetition_penalty,
                window=penalty_window,
                min_new=min_new_tokens,
                force_max=force_max_new_tokens,
            )
            reply = extract_assistant_only(full, prompt)
            print(f"[greedy full]\n{full}\n---\n")
            print(f"[greedy reply]\n{reply}\n---\n")
        return

    # Sweep ~40 decoding variations (greedy + top-k grid + top-p grid)
    variants = []
    # Greedy with multiple repetition penalties
    for rep in (1.2, 1.5, 1.8, 2.0):
        variants.append(("greedy", None, {"rep": rep}))

    temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reps = [1.2, 1.5, 1.8]
    topks = [10, 20, 40, 60]
    topps = [0.7, 0.8, 0.9, 0.95]

    for rep in reps:
        for T in temps:
            for k in topks:
                variants.append((f"top-k (T={T}, k={k}, rep={rep})", ("topk", T, rep), k))
            for p in topps:
                variants.append((f"top-p (T={T}, p={p}, rep={rep})", ("topp", T, rep), p))
            if len(variants) >= 40:
                break
        if len(variants) >= 40:
            break

    variants = variants[:40]

    for label, kind, param in variants:
        if kind is None:
            full = _greedy_local(
                model,
                prompt,
                max_new_tokens,
                rep=param.get("rep", repetition_penalty),
                window=penalty_window,
                min_new=min_new_tokens,
                force_max=force_max_new_tokens,
            )
        elif kind[0] == "topk":
            T, rep = kind[1], kind[2]
            full = _sample_local(
                model,
                prompt,
                max_new_tokens,
                temp=T,
                top_k_val=param,
                top_p_val=None,
                rep=rep,
                window=penalty_window,
                min_new=min_new_tokens,
                force_max=force_max_new_tokens,
            )
        else:  # topp
            T, rep = kind[1], kind[2]
            full = _sample_local(
                model,
                prompt,
                max_new_tokens,
                temp=T,
                top_k_val=None,
                top_p_val=param,
                rep=rep,
                window=penalty_window,
                min_new=min_new_tokens,
                force_max=force_max_new_tokens,
            )
        reply = extract_assistant_only(full, prompt)
        print(f"[{label} full]\n{full}\n---")
        print(f"[{label} reply]\n{reply}\n---\n")


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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a JSON config (overrides d_model, n_heads, n_layers, max_seq, vocab_size).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="If set, run a grid of decoding variations (greedy + multiple top-k/top-p).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature for single-run sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k for single-run sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p for single-run sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.5, help="Repetition penalty for greedy")
    parser.add_argument("--penalty-window", type=int, default=128, help="Penalty window for greedy")
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Minimum new tokens to generate before allowing EOS/stop-phrases to end generation.",
    )
    parser.add_argument(
        "--force-max-new-tokens",
        action="store_true",
        help="If set, never early-stop (EOS/stop-phrases/repeat-run) and try to emit exactly --max-new-tokens.",
    )

    args = parser.parse_args()

    eval_checkpoint(
        ckpt_path=args.ckpt,
        spm_model=args.spm_model,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        config_path=args.config,
        sweep=args.sweep,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        penalty_window=args.penalty_window,
        min_new_tokens=args.min_new_tokens,
        force_max_new_tokens=args.force_max_new_tokens,
    )


if __name__ == "__main__":
    main()
