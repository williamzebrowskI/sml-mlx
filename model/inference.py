#!/usr/bin/env python3
import argparse
import time
import pathlib

import mlx.core as mx
import mlx.nn   as nn
import sentencepiece as spm

from model.model import OpenELM, SMLMConfig

# ───────────────────────────────────────────────────────────────────────────────
# 1) Persistent PRNG key for sampling
# ───────────────────────────────────────────────────────────────────────────────
_global_key = mx.random.key(int(time.time() * 1e6))


# ───────────────────────────────────────────────────────────────────────────────
# 2) Decode config, tokenizer, model + checkpoint
# ───────────────────────────────────────────────────────────────────────────────
def load_model(config_path, tokenizer_path, ckpt_path=None, ckpt_dir="runs/sml-lm"):
    # 2.1) load model config
    cfg = SMLMConfig.from_json(config_path)

    # 2.2) tokenizer
    tok = spm.SentencePieceProcessor(model_file=tokenizer_path)

    # 2.3) build model
    model = OpenELM(cfg)

    # 2.4) pick checkpoint
    if ckpt_path:
        ckpt = pathlib.Path(ckpt_path)
    else:
        # find latest in ckpt_dir
        ckpts = list(pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"))
        if not ckpts:
            raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
        def step_num(p): return int(p.stem.split("_")[-1])
        ckpt = max(ckpts, key=step_num)
    print(f"[inference] loading weights from {ckpt}")
    model.load_weights(str(ckpt))

    # 2.5) freeze + move to GPU if available
    mx.eval(model.parameters())

    return cfg, tok, model


# ───────────────────────────────────────────────────────────────────────────────
# 3) Top-k + temperature sampling
# ───────────────────────────────────────────────────────────────────────────────
def generate(
    model,
    tok,
    prompt: str,
    max_new: int = 128,
    top_k:   int = 5,
    temp:    float = 0.2,
) -> str:
    global _global_key

    # encode prompt (without EOS)
    prompt_ids = tok.encode(prompt, out_type=int)
    context = mx.array(prompt_ids, dtype=mx.int32)

    # we'll collect only the newly generated IDs
    out_ids: list[int] = []

    for _ in range(max_new):
        # split RNG
        _global_key, subkey = mx.random.split(_global_key)

        # forward
        logits = model(mx.expand_dims(context, 0))[0, -1]
        logits = logits / max(temp, 1e-6)

        # top-k filter
        if 0 < top_k < logits.size:
            kth = mx.topk(logits, k=top_k)[-1] if mx.topk(logits, k=top_k).ndim else mx.topk(logits, k=top_k)
            logits = mx.where(logits < kth, -mx.inf, logits)

        # sample
        probs = nn.softmax(logits, axis=-1)
        next_id = int(mx.random.categorical(probs, key=subkey))

        # append & break on EOS
        out_ids.append(next_id)
        if next_id == tok.eos_id():
            break

        # extend context
        context = mx.concatenate([context, mx.array([next_id], dtype=mx.int32)], axis=0)

    # decode only the generated portion
    return tok.decode(out_ids)


# ───────────────────────────────────────────────────────────────────────────────
# 4) CLI
# ───────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Interactive inference with your SMLM model"
    )
    p.add_argument("--config",    required=True,  help="path/to/model/config.json")
    p.add_argument("--tokenizer", required=True,  help="path/to/spm.model")
    p.add_argument("--checkpoint",              help="exact ckpt to load (optional)")
    p.add_argument("--ckpt-dir",   default="runs/sml-lm",
                   help="where to look for latest if --checkpoint is unset")
    p.add_argument("--max-new",    type=int, default=20,   help="max new tokens")
    p.add_argument("--top-k",      type=int, default=40,   help="top-k sampling")
    p.add_argument("--temp",       type=float, default=0.7, help="temperature")
    args = p.parse_args()

    cfg, tok, model = load_model(
        args.config,
        args.tokenizer,
        ckpt_path=args.checkpoint,
        ckpt_dir=args.ckpt_dir,
    )

    print("🦜‍🦉  Ready! Type your prompt below. Ctrl-C to quit.\n")
    while True:
        prompt = input("User > ").strip()
        if not prompt:
            continue
        t0 = time.time()
        out = generate(
            model, tok, prompt,
            max_new=args.max_new,
            top_k=args.top_k,
            temp=args.temp,
        )
        print("Model >", out)
        print(f"⏱️  {time.time() - t0:.2f}s\n")


if __name__ == "__main__":
    main()