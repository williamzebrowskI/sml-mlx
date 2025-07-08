# #!/usr/bin/env python3
# import argparse, time, pathlib

# import mlx.core as mx
# import sentencepiece as spm
# from mlx.nn import softmax
# from model.model import OpenELM, SMLMConfig

# def load_model(config_path, tokenizer_path, checkpoint=None, ckpt_dir="runs/sml-lm"):
#     # 1) load config & build model
#     cfg = SMLMConfig.from_json(config_path)
#     model = OpenELM(cfg)

#     # 2) pick checkpoint
#     if checkpoint:
#         ckpt = pathlib.Path(checkpoint)
#     else:
#         ckpts = sorted(pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"))
#         if not ckpts:
#             raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
#         ckpt = ckpts[-1]
#     print(f"[inference] loading weights from {ckpt.name}")
#     model.load_weights(str(ckpt))
#     mx.eval(model.parameters())  # freeze

#     # 3) tokenizer
#     tok = spm.SentencePieceProcessor(model_file=tokenizer_path)
#     return model, tok

# def sample_next(logits, top_k, temp):
#     # scale
#     logits = logits / max(temp, 1e-6)
#     # top-k cutoff
#     if 0 < top_k < logits.size:
#         kth = mx.topk(logits, k=top_k)[-1]
#         logits = mx.where(logits < kth, -mx.inf, logits)
#     # softmax & sample
#     probs = softmax(logits, axis=-1)
#     return int(mx.random.categorical(probs))

# def generate(model, tok, prompt, max_new=128, top_k=40, temp=0.7):
#     # encode + batch-dim
#     ids = tok.encode(prompt, out_type=int)
#     x = mx.array(ids, dtype=mx.int32)[None, :]
#     out = []
#     for _ in range(max_new):
#         logits = model(x)[0, -1]           # (vocab,)
#         nxt    = sample_next(logits, top_k, temp)
#         out.append(nxt)
#         if nxt == tok.eos_id():
#             break
#         # append and continue
#         x = mx.concatenate([x, mx.array([[nxt]], dtype=mx.int32)], axis=1)
#     return tok.decode(out)

# def main():
#     p = argparse.ArgumentParser("simple SMLM inference")
#     p.add_argument("--config",    required=True)
#     p.add_argument("--tokenizer", required=True)
#     p.add_argument("--checkpoint",help="specific ckpt to load")
#     p.add_argument("--ckpt-dir",  default="runs/sml-lm")
#     p.add_argument("--max-new",   type=int,   default=20)
#     p.add_argument("--top-k",     type=int,   default=40)
#     p.add_argument("--temp",      type=float, default=0.7)
#     args = p.parse_args()

#     model, tok = load_model(
#         args.config, args.tokenizer,
#         checkpoint=args.checkpoint, ckpt_dir=args.ckpt_dir
#     )
#     print("🦜‍🦉 Ready! Type prompts (Ctrl-C to quit).")
#     while True:
#         prompt = input("▶ ").strip()
#         if not prompt:
#             continue
#         t0 = time.time()
#         out = generate(
#             model, tok, prompt,
#             max_new=args.max_new,
#             top_k=args.top_k,
#             temp=args.temp
#         )
#         print(out)
#         print(f"⏱️ {time.time()-t0:.2f}s\n")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
import time
import pathlib

import mlx.core as mx
import sentencepiece as spm
from mlx.nn import softmax

from model.model import OpenELM, SMLMConfig

def load_model(config_path: str,
               tokenizer_path: str,
               checkpoint: str = None,
               ckpt_dir: str = "runs/sml-lm"):
    # 1) load config & build
    cfg   = SMLMConfig.from_json(config_path)
    model = OpenELM(cfg)

    # 2) pick checkpoint
    if checkpoint:
        ckpt = pathlib.Path(checkpoint)
    else:
        ckpts = sorted(pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"))
        if not ckpts:
            raise FileNotFoundError(f"No ckpt in {ckpt_dir}")
        ckpt = ckpts[-1]

    print(f"[inference] loading weights from {ckpt.name}")
    model.load_weights(str(ckpt))
    mx.eval(model.parameters())  # freeze

    # 3) tokenizer
    tok = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return model, tok

def sample_next(logits, top_k: int, temp: float):
    # temperature + top-k filtering + sample
    logits = logits / max(temp, 1e-6)
    if 0 < top_k < logits.size:
        kth    = mx.topk(logits, k=top_k)[-1]
        logits = mx.where(logits < kth, -mx.inf, logits)
    probs = softmax(logits, axis=-1)
    return int(mx.random.categorical(probs))

def generate(model, tok, prompt: str,
             max_new: int = 128,
             top_k:   int = 40,
             temp:    float = 0.7) -> str:
    # encode prompt + add batch dim
    ids = tok.encode(prompt, out_type=int)
    x   = mx.array(ids, dtype=mx.int32)[None, :]  # (1, L)
    out = []

    for _ in range(max_new):
        logits = model(x)[0, -1]            # full forward every step
        nxt    = sample_next(logits, top_k, temp)
        out.append(nxt)
        if nxt == tok.eos_id():
            break
        # append new token and repeat
        x = mx.concatenate([x, mx.array([[nxt]], dtype=mx.int32)], axis=1)

    return tok.decode(out)

def main():
    p = argparse.ArgumentParser("simple SMLM inference")
    p.add_argument("--config",    required=True,  help="path/to/config.json")
    p.add_argument("--tokenizer", required=True,  help="path/to/spm.model")
    p.add_argument("--checkpoint",help="specific ckpt to load")
    p.add_argument("--ckpt-dir",  default="runs/sml-lm",
                   help="dir to search for latest ckpt")
    p.add_argument("--max-new",   type=int,   default=20)
    p.add_argument("--top-k",     type=int,   default=40)
    p.add_argument("--temp",      type=float, default=0.7)
    args = p.parse_args()

    model, tok = load_model(
        args.config, args.tokenizer,
        checkpoint=args.checkpoint,
        ckpt_dir=args.ckpt_dir
    )

    print("🦜 ‍Ready! Enter prompts (Ctrl-C to quit).")
    while True:
        prompt = input("▶ ").strip()
        if not prompt:
            continue
        t0  = time.time()
        out = generate(model, tok, prompt,
                       max_new=args.max_new,
                       top_k=args.top_k,
                       temp=args.temp)
        print(out)
        print(f"⏱️ {time.time() - t0:.2f}s\n")

if __name__ == "__main__":
    main()