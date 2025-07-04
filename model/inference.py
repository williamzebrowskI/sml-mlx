# #!/usr/bin/env python3
# import argparse
# import time
# import pathlib

# import mlx.core as mx
# import mlx.nn   as nn
# import sentencepiece as spm

# from model.model import OpenELM, SMLMConfig

# # ───────────────────────────────────────────────────────────────────────────────
# # 1) Persistent PRNG key for sampling
# # ───────────────────────────────────────────────────────────────────────────────
# _global_key = mx.random.key(int(time.time() * 1e6))


# # ───────────────────────────────────────────────────────────────────────────────
# # 2) Decode config, tokenizer, model + checkpoint
# # ───────────────────────────────────────────────────────────────────────────────
# def load_model(config_path, tokenizer_path, ckpt_path=None, ckpt_dir="runs/sml-lm"):
#     # 2.1) load model config
#     cfg = SMLMConfig.from_json(config_path)

#     # 2.2) tokenizer
#     tok = spm.SentencePieceProcessor(model_file=tokenizer_path)

#     # 2.3) build model
#     model = OpenELM(cfg)

#     # 2.4) pick checkpoint
#     if ckpt_path:
#         ckpt = pathlib.Path(ckpt_path)
#     else:
#         # find latest in ckpt_dir
#         ckpts = list(pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"))
#         if not ckpts:
#             raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
#         def step_num(p): return int(p.stem.split("_")[-1])
#         ckpt = max(ckpts, key=step_num)
#     print(f"[inference] loading weights from {ckpt}")
#     model.load_weights(str(ckpt))

#     # 2.5) freeze + move to GPU if available
#     mx.eval(model.parameters())

#     return cfg, tok, model


# # ───────────────────────────────────────────────────────────────────────────────
# # 3) Top-k + temperature sampling
# # ───────────────────────────────────────────────────────────────────────────────
# def generate(
#     model,
#     tok,
#     prompt: str,
#     max_new: int = 128,
#     top_k:   int = 5,
#     temp:    float = 0.2,
# ) -> str:
#     global _global_key

#     # encode prompt (without EOS)
#     prompt_ids = tok.encode(prompt, out_type=int)
#     context = mx.array(prompt_ids, dtype=mx.int32)

#     # we'll collect only the newly generated IDs
#     out_ids: list[int] = []

#     for _ in range(max_new):
#         # split RNG
#         _global_key, subkey = mx.random.split(_global_key)

#         # forward
#         logits = model(mx.expand_dims(context, 0))[0, -1]
#         logits = logits / max(temp, 1e-6)

#         # top-k filter
#         if 0 < top_k < logits.size:
#             kth = mx.topk(logits, k=top_k)[-1] if mx.topk(logits, k=top_k).ndim else mx.topk(logits, k=top_k)
#             logits = mx.where(logits < kth, -mx.inf, logits)

#         # sample
#         probs = nn.softmax(logits, axis=-1)
#         next_id = int(mx.random.categorical(probs, key=subkey))

#         # append & break on EOS
#         out_ids.append(next_id)
#         if next_id == tok.eos_id():
#             break

#         # extend context
#         context = mx.concatenate([context, mx.array([next_id], dtype=mx.int32)], axis=0)

#     # decode only the generated portion
#     return tok.decode(out_ids)


# # ───────────────────────────────────────────────────────────────────────────────
# # 4) CLI
# # ───────────────────────────────────────────────────────────────────────────────
# def main():
#     p = argparse.ArgumentParser(
#         description="Interactive inference with your SMLM model"
#     )
#     p.add_argument("--config",    required=True,  help="path/to/model/config.json")
#     p.add_argument("--tokenizer", required=True,  help="path/to/spm.model")
#     p.add_argument("--checkpoint",              help="exact ckpt to load (optional)")
#     p.add_argument("--ckpt-dir",   default="runs/sml-lm",
#                    help="where to look for latest if --checkpoint is unset")
#     p.add_argument("--max-new",    type=int, default=20,   help="max new tokens")
#     p.add_argument("--top-k",      type=int, default=40,   help="top-k sampling")
#     p.add_argument("--temp",       type=float, default=0.7, help="temperature")
#     args = p.parse_args()

#     cfg, tok, model = load_model(
#         args.config,
#         args.tokenizer,
#         ckpt_path=args.checkpoint,
#         ckpt_dir=args.ckpt_dir,
#     )

#     print("🦜‍🦉  Ready! Type your prompt below. Ctrl-C to quit.\n")
#     while True:
#         prompt = input("User > ").strip()
#         if not prompt:
#             continue
#         t0 = time.time()
#         out = generate(
#             model, tok, prompt,
#             max_new=args.max_new,
#             top_k=args.top_k,
#             temp=args.temp,
#         )
#         print("Model >", out)
#         print(f"⏱️  {time.time() - t0:.2f}s\n")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse, time, pathlib

import mlx.core as mx
import sentencepiece as spm
from mlx.nn import softmax
from model.model import OpenELM, SMLMConfig

def load_model(config_path, tokenizer_path, checkpoint=None, ckpt_dir="runs/sml-lm"):
    # 1) load config & build model
    cfg = SMLMConfig.from_json(config_path)
    model = OpenELM(cfg)

    # 2) pick checkpoint
    if checkpoint:
        ckpt = pathlib.Path(checkpoint)
    else:
        ckpts = sorted(pathlib.Path(ckpt_dir).glob("ckpt_*.safetensors"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        ckpt = ckpts[-1]
    print(f"[inference] loading weights from {ckpt.name}")
    model.load_weights(str(ckpt))
    mx.eval(model.parameters())  # freeze

    # 3) tokenizer
    tok = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return model, tok

def sample_next(logits, top_k, temp):
    # scale
    logits = logits / max(temp, 1e-6)
    # top-k cutoff
    if 0 < top_k < logits.size:
        kth = mx.topk(logits, k=top_k)[-1]
        logits = mx.where(logits < kth, -mx.inf, logits)
    # softmax & sample
    probs = softmax(logits, axis=-1)
    return int(mx.random.categorical(probs))

def generate(model, tok, prompt, max_new=128, top_k=40, temp=0.7):
    # encode + batch-dim
    ids = tok.encode(prompt, out_type=int)
    x = mx.array(ids, dtype=mx.int32)[None, :]
    out = []
    for _ in range(max_new):
        logits = model(x)[0, -1]           # (vocab,)
        nxt    = sample_next(logits, top_k, temp)
        out.append(nxt)
        if nxt == tok.eos_id():
            break
        # append and continue
        x = mx.concatenate([x, mx.array([[nxt]], dtype=mx.int32)], axis=1)
    return tok.decode(out)

def main():
    p = argparse.ArgumentParser("simple SMLM inference")
    p.add_argument("--config",    required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint",help="specific ckpt to load")
    p.add_argument("--ckpt-dir",  default="runs/sml-lm")
    p.add_argument("--max-new",   type=int,   default=20)
    p.add_argument("--top-k",     type=int,   default=40)
    p.add_argument("--temp",      type=float, default=0.7)
    args = p.parse_args()

    model, tok = load_model(
        args.config, args.tokenizer,
        checkpoint=args.checkpoint, ckpt_dir=args.ckpt_dir
    )
    print("🦜‍🦉 Ready! Type prompts (Ctrl-C to quit).")
    while True:
        prompt = input("▶ ").strip()
        if not prompt:
            continue
        t0 = time.time()
        out = generate(
            model, tok, prompt,
            max_new=args.max_new,
            top_k=args.top_k,
            temp=args.temp
        )
        print(out)
        print(f"⏱️ {time.time()-t0:.2f}s\n")

if __name__ == "__main__":
    main()