#!/usr/bin/env python3
"""
MLX inference for OpenELM checkpoints.

- Auto-picks latest ckpt in --ckpt-dir (or use --checkpoint)
- Aligns vocab_size to tokenizer size (avoids emb/lm_head shape errors)
- Dropout disabled via model.eval()
- Adds BOS to prompt to match training (add_bos=True)
- Softmax in fp32; sliding window uses cfg.context_size
"""

from __future__ import annotations
import argparse, pathlib, sys, time, random, os
from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm

# import model + config
try:
    from model.model import OpenELM, SMLMConfig          # run as: python -m model.inference
except Exception:
    # fallback if someone runs this file directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import model as model_mod
    OpenELM, SMLMConfig = model_mod.OpenELM, model_mod.SMLMConfig

# ────────────────────────────── RUNS ────────────────────────────────
# (prompt, temperature, top_k, top_p)
RUNS = [
    ("Headline: Breakthrough Battery Doubles EV Range\n\nArticle: ",          0.7, 50, 0.9),
    ("Paris is the capital and most populous city of France. It ",            0.7, 50, 0.9),
    ("In computer science, a binary search tree is ",                          0.7, 50, 0.9),
    ("The recipe begins by preheating the oven to 375°F. First, ",             0.7, 40, 0.9),
    ("Once upon a time in a faraway land, there lived a young princess who ",  0.7, 40, 0.9),
]

# ─────────────────────────── small utils ────────────────────────────
def log(*msg):
    print("[infer]", *msg, flush=True)

def pick_latest_ckpt(ckpt_dir: pathlib.Path) -> Optional[pathlib.Path]:
    cands = list(ckpt_dir.glob("ckpt_*.safetensors"))
    if not cands:
        final = ckpt_dir / "ckpt_final.safetensors"
        return final if final.exists() else None

    def step_of(p: pathlib.Path) -> int:
        try:
            stem = p.stem
            # Accept ckpt_000010 or ckpt_000010_tag
            if "_" in stem:
                tail = stem.split("_")[-1]
                if tail.isdigit():
                    return int(tail)
                for part in tail.replace("-", "_").split("_"):
                    if part.isdigit():
                        return int(part)
            return int(stem.split("_")[1])
        except Exception:
            return 0

    return max(cands, key=step_of)

# ───────────────────────── sampling helpers ─────────────────────────
def apply_repetition_penalty(logits: mx.array, generated: List[int], penalty: float) -> mx.array:
    if penalty <= 1.0 or not generated:
        return logits
    out = logits.astype(mx.float32)
    for tid in set(generated):
        out[tid] = out[tid] / penalty
    return out

def filter_top_k(logits: mx.array, k: int) -> mx.array:
    if k is None or k <= 0 or k >= int(logits.size):
        return logits
    topk_vals = mx.topk(logits, k=k)   # sorted descending
    kth = topk_vals[-1]
    return mx.where(logits < kth, -mx.inf, logits)

def filter_top_p(logits: mx.array, top_p: float) -> mx.array:
    if top_p is None or not (0.0 < top_p < 1.0):
        return logits
    sorted_logits = mx.sort(logits, axis=-1)[::-1]
    probs = nn.softmax(sorted_logits.astype(mx.float32), axis=-1)
    cum = mx.cumsum(probs, axis=-1)
    mask = (cum > top_p).astype(mx.int32)
    cutoff_idx = int(mx.argmax(mask, axis=-1))
    threshold = float(sorted_logits[cutoff_idx])
    return mx.where(logits < threshold, -mx.inf, logits)

def sample_next_id(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated: List[int],
) -> int:
    t = max(1e-5, float(temperature))
    x = (logits.astype(mx.float32) / t)
    x = apply_repetition_penalty(x, generated, repetition_penalty)
    x = filter_top_k(x, top_k)
    x = filter_top_p(x, top_p)
    probs = nn.softmax(x, axis=-1)
    return int(mx.random.categorical(probs))

# ───────────────────────── model loading ────────────────────────────
def load_model_and_tok(cfg_path: str, spm_path: str, ckpt_path: Optional[str], ckpt_dir: str, device: str):
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    cfg = SMLMConfig.from_json(cfg_path)

    # Load tokenizer FIRST and align vocab_size to it
    tok = spm.SentencePieceProcessor(model_file=spm_path)
    sp_vocab = int(tok.get_piece_size())
    if cfg.vocab_size != sp_vocab:
        log(f"vocab_size mismatch: config={cfg.vocab_size} tokenizer={sp_vocab} → overriding config")
        cfg.vocab_size = sp_vocab

    if hasattr(mx, "set_default_dtype"):
        if cfg.torch_dtype in ("float16", "fp16"):
            mx.set_default_dtype(mx.float16)
        elif cfg.torch_dtype == "bfloat16":
            mx.set_default_dtype(mx.bfloat16)

    model = OpenELM(cfg)
    # Disable dropout etc. during generation
    if hasattr(model, "eval"):
        model.eval()

    ckpt = pathlib.Path(ckpt_path) if ckpt_path else pick_latest_ckpt(pathlib.Path(ckpt_dir))
    if ckpt is None or not ckpt.exists():
        sys.exit(f"❌ No valid checkpoint found (looked in {ckpt_dir})")
    log(f"Loading weights: {ckpt}")
    model.load_weights(str(ckpt))
    mx.eval(model.parameters())  # materialize weights

    return model, tok, cfg

# ───────────────────────── generation loop ──────────────────────────
def generate(
    model: OpenELM,
    tok: spm.SentencePieceProcessor,
    cfg: SMLMConfig,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    stop_at_eos: bool = True,
) -> str:
    # IMPORTANT: add BOS to match training
    input_ids: List[int] = tok.encode(prompt, out_type=int, add_bos=True)
    generated: List[int] = []
    ctx = int(cfg.context_size)

    for _ in range(max_new_tokens):
        full = input_ids + generated
        window = full[-ctx:] if len(full) > ctx else full
        x = mx.array(window, dtype=mx.int32)[None, :]  # (1, L)

        logits = model(x)[0, -1]  # (vocab,)
        next_id = sample_next_id(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated=generated,
        )
        if stop_at_eos and next_id == tok.eos_id():
            break
        generated.append(next_id)

    return tok.decode(generated)

# ─────────────────────────── CLI entrypoint ─────────────────────────
def main():
    ap = argparse.ArgumentParser("OpenELM / MLX inference")
    ap.add_argument("--config", required=True, help="Path to training config.json")
    ap.add_argument("--tokenizer", required=True, help="SentencePiece model file (spm.model)")
    ap.add_argument("--ckpt-dir", default="runs/sml-lm", help="Directory with ckpt_*.safetensors")
    ap.add_argument("--checkpoint", default=None, help="Specific ckpt path (overrides --ckpt-dir)")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")

    # Single-prompt path
    ap.add_argument("--prompt", default=None, help="If given, run a single prompt instead of built-in RUNS")

    # Sampling knobs
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--no-eos-stop", action="store_true", help="Do not stop on EOS token")

    # Repro
    ap.add_argument("--seed", type=int, default=0, help="Set >0 for deterministic sampling")

    args = ap.parse_args()

    if args.seed and args.seed > 0:
        random.seed(args.seed)
        mx.random.seed(args.seed)

    model, tok, cfg = load_model_and_tok(
        args.config, args.tokenizer, args.checkpoint, args.ckpt_dir, args.device
    )

    stop_at_eos = not args.no_eos_stop
    rep = args.repetition_penalty
    mnew = args.max_new

    def run_one(prompt: str, temp: float, top_k: int, top_p: float):
        print(f"\nPrompt: {prompt!r}")
        t0 = time.time()
        out = generate(
            model, tok, cfg, prompt,
            max_new_tokens=mnew,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep,
            stop_at_eos=stop_at_eos,
        )
        dt = time.time() - t0
        ntoks = len(tok.encode(out, out_type=int))
        print(f"[infer] gen speed ≈ {ntoks/dt:.1f} tok/s ({ntoks} tokens in {dt:.2f}s)")
        print(out)

    if args.prompt is not None:
        run_one(args.prompt, args.temperature, args.top_k, args.top_p)
    else:
        for i, (prompt, temp, top_k, top_p) in enumerate(RUNS, 1):
            print(f"\n=== Run {i}/{len(RUNS)} ===")
            print(f"temp={temp} top_k={top_k} top_p={top_p} rep={rep} max_new={mnew}")
            run_one(prompt, temp, top_k, top_p)

if __name__ == "__main__":
    main()