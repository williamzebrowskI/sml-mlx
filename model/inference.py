

# #!/usr/bin/env python3
# """
# Generate exactly 10 completions from an OpenELM checkpoint,
# each with its own (prompt, temp, top_k) trio.

# Usage
# -----
# python inference.py \
#   --config    model/config.json \
#   --tokenizer tokenizer/spm.model \
#   --ckpt-dir  runs/sml-lm           # or --checkpoint path/to/ckpt.safetensors
# """

# from __future__ import annotations
# import argparse, time, pathlib, sys
# from typing import List, Tuple

# import mlx.core as mx
# import sentencepiece as spm
# from mlx.nn import softmax

# from model.model import OpenELM, SMLMConfig


# # ─────────────────────────── sampling helpers ──────────────────────────
# def sample_token(logits, *, top_k: int, temp: float) -> int:
#     logits = logits / max(temp, 1e-6)
#     if 0 < top_k < logits.size:
#         kth = mx.topk(logits, k=top_k)[-1]
#         logits = mx.where(logits < kth, -mx.inf, logits)
#     probs = softmax(logits, axis=-1)
#     return int(mx.random.categorical(probs))


# def complete(model: OpenELM, tok: spm.SentencePieceProcessor, prompt: str,
#              *, max_new: int, top_k: int, temp: float) -> str:
#     ids = tok.encode(prompt, out_type=int)
#     x   = mx.array(ids, dtype=mx.int32)[None, :]
#     out = []

#     for _ in range(max_new):
#         logits = model(x)[0, -1]
#         nxt    = sample_token(logits, top_k=top_k, temp=temp)
#         if nxt == tok.eos_id():
#             break
#         out.append(nxt)
#         x = mx.concatenate([x, mx.array([[nxt]], dtype=mx.int32)], axis=1)

#     return tok.decode(out)


# # ─────────────────────────── main script ───────────────────────────────
# def load_latest_or_given(cfg_path: str, spm_path: str,
#                          ckpt: str | None, ckpt_dir: str):
#     cfg   = SMLMConfig.from_json(cfg_path)
#     model = OpenELM(cfg)
#     ckpt_path = pathlib.Path(ckpt) if ckpt else max(
#         pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"),
#         key=lambda p: p.stat().st_mtime,
#         default=None,
#     )
#     if ckpt_path is None:
#         sys.exit(f"❌ No checkpoint found in {ckpt_dir}")
#     print(f"[load] using {ckpt_path.name}")
#     model.load_weights(str(ckpt_path))
#     mx.eval(model.parameters())           # freeze
#     tok = spm.SentencePieceProcessor(model_file=spm_path)
#     return model, tok


# def main() -> None:
#     ap = argparse.ArgumentParser("OpenSML 10-run sampler")
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--tokenizer", required=True)
#     ap.add_argument("--checkpoint")
#     ap.add_argument("--ckpt-dir", default="runs/sml-lm")
#     ap.add_argument("--max-new", type=int, default=10)
#     args = ap.parse_args()

#     model, tok = load_latest_or_given(args.config, args.tokenizer,
#                                       args.checkpoint, args.ckpt_dir)

#     # -------- 10 prompt / sampling tuples --------
#     RUNS: List[Tuple[str, float, int]] = [
#         ("How AP reported in all ",              0.8, 10),
#         ("Log In Please enter your ECode to",  0.7, 10),
#         # ("In the future, humans will",       1.2, 50),
#         # ("A recipe for blueberry muffins:",  0.8, 30),
#         # ("Once upon a time",                 1.0, 40),
#         # ("Python list comprehension is",     0.6, 15),
#         # ("Describe the city of Paris.",      0.9, 40),
#         # ("Write a haiku about rain.",        0.8, 20),
#         # ("Pros and cons of electric cars:",  1.1, 50),
#         # ("What is the meaning of life?",     0.7, 40),
#     ]
#     # --------------------------------------------

#     for i, (prompt, temp, top_k) in enumerate(RUNS, 1):
#         t0 = time.time()
#         out = complete(model, tok, prompt,
#                        max_new=args.max_new,
#                        top_k=top_k,
#                        temp=temp)
#         dt = time.time() - t0
#         print(f"\n=== Run {i}/10 ===")
#         print(f"Prompt : {prompt!r}")
#         print(f"temp={temp}, top_k={top_k}, {args.max_new} max tokens")
#         print(f"Output : {out}")
#         print(f"⏱  {dt:.2f}s")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Generate exactly 5 completions from an OpenELM checkpoint,
each with its own (prompt, temp, top_k, top_p) quartet,
with a 5 s pause between each.
"""
from __future__ import annotations
import argparse, time, pathlib, sys
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm

from model.model import OpenELM, SMLMConfig


# ─────────────────────────── sampling helpers ──────────────────────────
def sample_token(logits, *, top_k: int = 0, top_p: float = 0.0, temp: float) -> int:
    # temperature scaling
    logits = logits / max(temp, 1e-6)

    # Top-K filtering
    if 0 < top_k < logits.size:
        kth = mx.topk(logits, k=top_k)[-1]
        logits = mx.where(logits < kth, -mx.inf, logits)

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        # sort logits descending
        sorted_logits = mx.sort(logits, axis=-1)[::-1]
        sorted_probs = nn.softmax(sorted_logits, axis=-1)
        cumprobs = mx.cumsum(sorted_probs, axis=-1)
        # find cutoff where cumprob > top_p
        cutoff_mask = (cumprobs > top_p).astype(mx.int32)
        cutoff_idx = int(mx.argmax(cutoff_mask, axis=-1))
        threshold = float(sorted_logits[cutoff_idx])
        logits = mx.where(logits < threshold, -mx.inf, logits)

    probs = nn.softmax(logits, axis=-1)
    return int(mx.random.categorical(probs))


def complete(model: OpenELM,
             tok: spm.SentencePieceProcessor,
             prompt: str,
             *,
             max_new: int,
             top_k: int,
             top_p: float,
             temp: float) -> str:
    ids = tok.encode(prompt, out_type=int)
    x = mx.array(ids, dtype=mx.int32)[None, :]
    out_ids: list[int] = []

    for _ in range(max_new):
        logits = model(x)[0, -1]
        nxt = sample_token(logits, top_k=top_k, top_p=top_p, temp=temp)
        if nxt == tok.eos_id():
            break
        out_ids.append(nxt)
        x = mx.concatenate([x, mx.array([[nxt]], dtype=mx.int32)], axis=1)

    return tok.decode(out_ids)


# ─────────────────────────── main script ───────────────────────────────
def load_latest_or_given(cfg_path: str,
                         spm_path: str,
                         ckpt: str | None,
                         ckpt_dir: str):
    cfg = SMLMConfig.from_json(cfg_path)
    model = OpenELM(cfg)
    # pick checkpoint
    if ckpt:
        ckpt_path = pathlib.Path(ckpt)
    else:
        candidates = list(pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"))
        ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not ckpt_path.exists():
        sys.exit(f"❌ No checkpoint found: {ckpt_path}")
    print(f"[load] using {ckpt_path.name}")
    model.load_weights(str(ckpt_path))
    mx.eval(model.parameters())  # freeze weights
    tok = spm.SentencePieceProcessor(model_file=spm_path)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser("OpenELM 5-run sampler")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--checkpoint", help="Path to a specific checkpoint")
    ap.add_argument("--ckpt-dir", default="runs/sml-lm")
    ap.add_argument("--max-new", type=int, default=128,
                    help="Maximum new tokens to generate per prompt")
    args = ap.parse_args()

    model, tok = load_latest_or_given(
        args.config,
        args.tokenizer,
        args.checkpoint,
        args.ckpt_dir,
    )

    # Define 5 (prompt, temp, top_k, top_p) tuples:
    # RUNS: List[Tuple[str, float, int, float]] = [
    #     ("Once upon a time in a faraway land, ", 0.0,   0, 0.0),   # greedy
    #     ("Once upon a time in a faraway land, ", 0.7,  50, 0.0),   # top-k
    #     ("Once upon a time in a faraway land, ", 0.8,   0, 0.9),   # nucleus
    #     ("The capital of France is ",             1.0, 100, 0.9), # creative
    #     ("Explain quantum computing to me: ",     0.5,   0, 0.5),  # conservative nucleus
    # ]
    # RUNS:  List[Tuple[str, float, int, float]] = [
    #     ("Once upon a time in a faraway land, ", 0.3, 0, 0.9),
    #     ("The capital of France is " ,            0.2, 0, 0.9),
    #     ("Explain quantum computing to me: " ,    0.5, 0, 0.9),
    #     ("A recipe for blueberry muffins: " ,    0.4, 0, 0.9),
    #     ("In the future, humans will "    ,       0.6, 0, 0.9),
    # ]
    # More conservative nucleus + top-k
    RUNS:  List[Tuple[str, float, int, float]] = [
        ("Once upon a time in a faraway land, there lived a young princess who…", 0.1, 50, 0.7),
        ("The capital of France is ",                                    0.1, 20, 0.5),
        ("Explain quantum computing to me: ",                            0.2, 50, 0.7),
        ("A recipe for blueberry muffins: ",                             0.1, 30, 0.6),
        ("In the future, humans will ",                                  0.2, 50, 0.7),
    ]

    for i, (prompt, temp, top_k, top_p) in enumerate(RUNS, 1):
        t0 = time.time()
        out = complete(
            model, tok, prompt,
            max_new=args.max_new,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
        )
        dt = time.time() - t0

        print(f"\n=== Run {i}/{len(RUNS)} ===")
        print(f"Prompt : {prompt!r}")
        print(f"temp={temp}, top_k={top_k}, top_p={top_p}, max_new={args.max_new}")
        print(f"Output : {out}")
        print(f"⏱  {dt:.2f}s")

        time.sleep(5)


if __name__ == "__main__":
    main()
