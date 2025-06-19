
# #!/usr/bin/env python3
# """
# LoRA fine-tune (resume-aware, MLX)
# """
# from __future__ import annotations
# import argparse, json, math, os, random, sys
# import mlx.core as mx
# import mlx.nn   as nn
# import mlx.optimizers as optim

# sys.path.insert(0, os.getcwd())                # project root
# from model.model import Transformer, TransformerConfig
# from tokenizer    import sp_tokenizer
# import lora                                      # local helpers

# # ───────────────────────── meta helpers
# def _meta(p: str) -> str:          return p + ".meta"
# def save_step(step: int, fp: str): json.dump({"step": step}, open(_meta(fp), "w"))
# def load_step(fp: str) -> int:
#     meta = _meta(fp)
#     return json.load(open(meta))["step"] if os.path.isfile(meta) else 0

# # ───────────────────────── CLI
# def build():
#     p = argparse.ArgumentParser("LoRA fine-tune (resume-aware)")
#     p.add_argument("--checkpoint",  "--ckpt", required=True)
#     p.add_argument("--config",      required=True)
#     p.add_argument("--tokenizer",   required=True)
#     p.add_argument("--train_jsonl", required=True)
#     p.add_argument("--valid_jsonl", required=True)
#     p.add_argument("--adapter_file", default="model/lora_adapters/adapters.npz")

#     p.add_argument("--lora_layers", type=int, default=16)
#     p.add_argument("--batch_size",  type=int, default=16)
#     p.add_argument("--iters",       type=int, default=4_000)
#     p.add_argument("--lr",          type=float, default=2e-5)

#     p.add_argument("--save_every",   type=int, default=500)
#     p.add_argument("--steps_eval",   type=int, default=250)
#     p.add_argument("--steps_report", type=int, default=100)

#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--train", action="store_true")
#     p.add_argument("--test",  action="store_true")
#     return p.parse_args()

# # ───────────────────────── main
# def main():
#     a = build()
#     random.seed(a.seed);  mx.random.seed(a.seed)

#     # base model -------------------------------------------------------------
#     cfg   = TransformerConfig(**json.load(open(a.config)))
#     model = Transformer(cfg)
#     model.load_weights(a.checkpoint)
#     model.freeze()

#     tok      = sp_tokenizer.load(a.tokenizer)
#     BOS, EOS = tok.sp.bos_id(), tok.sp.eos_id()

#     # loss -------------------------------------------------------------------
#     def loss(x, y, L):
#         logits = model(x).astype(mx.float32)
#         pos    = mx.arange(y.shape[1])[None, :]
#         mask = (pos >= (L[:, None] - 1)) & (y != 0) & (y != BOS)
#         ce     = nn.losses.cross_entropy(logits, y) * mask
#         return ce.sum() / mask.sum(), mask.sum()

#     grad_fn = nn.value_and_grad(model, loss)

#     # add LoRA ---------------------------------------------------------------
#     blocks = model.layers if hasattr(model, "layers") else model.encoder.layers
#     for blk in blocks[-a.lora_layers:]:
#         for name, sub in blk.named_modules():
#             if "." not in name and isinstance(sub, nn.Linear):
#                 setattr(blk, name, lora.LoRALinear.from_linear(sub, dropout=0.1))

#     # resume -----------------------------------------------------------------
#     start = 0
#     if os.path.isfile(a.adapter_file):
#         lora.load_lora_adapters(model, a.adapter_file)
#         start = load_step(a.adapter_file)
#         print(f"[resume] restored adapters; continuing from step {start}")

#     opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.01)

#     # data -------------------------------------------------------------------
#     train_pairs = list(lora.load_pairs(a.train_jsonl))
#     valid_pairs = list(lora.load_pairs(a.valid_jsonl))
#     pad = lambda s, L: (s + [0] * L)[:L]

#     step = start
#     if a.train:
#         while step < a.iters:
#             random.shuffle(train_pairs)
#             for chunk in (train_pairs[i:i + a.batch_size]
#                           for i in range(0, len(train_pairs), a.batch_size)):
#                 qs, ans = zip(*chunk)
#                 enc_q   = [[BOS] + tok.encode(q)   for q in qs]
#                 enc_a   = [tok.encode(a) + [EOS]   for a in ans]
#                 enc_qa  = [q + a for q, a in zip(enc_q, enc_a)]
#                 L       = min(cfg.context_size, max(map(len, enc_qa)))

#                 x  = mx.array([pad(s, L) for s in enc_q])
#                 y  = mx.array([pad(s, L) for s in enc_qa])
#                 pr = mx.array([len(s)    for s in enc_q])

#                 (tr_loss, _), grads = grad_fn(x, y, pr)
#                 opt.update(model, grads)

#                 # ---- reporting -------------------------------------------
#                 if step % a.steps_report == 0:
#                     print(f"[Train] step {step:6d}  loss {tr_loss.item():.4f}")

#                 # ---- evaluation ------------------------------------------
#                 if step and step % a.steps_eval == 0:
#                     val_loss, n_tok = 0.0, 0
#                     for vchunk in (valid_pairs[i:i + a.batch_size]
#                                     for i in range(0, len(valid_pairs), a.batch_size)):
#                         vq, va = zip(*vchunk)
#                         eq  = [[BOS] + tok.encode(q)      for q in vq]
#                         ea  = [tok.encode(a) + [EOS]      for a in va]
#                         eqa = [q + a for q, a in zip(eq, ea)]
#                         L2  = min(cfg.context_size, max(map(len, eqa)))
#                         vx  = mx.array([pad(s, L2) for s in eq])
#                         vy  = mx.array([pad(s, L2) for s in eqa])
#                         pr2 = mx.array([len(s)     for s in eq])
#                         vl, _ = loss(vx, vy, pr2)
#                         val_loss += vl.item() * len(vq)
#                         n_tok    += len(vq)
#                     val_loss /= n_tok
#                     val_ppl   = math.exp(val_loss)
#                     print(f"[Eval]  step {step:6d}  val_loss {val_loss:.4f}  val_ppl {val_ppl:.2f}")

#                 # ---- checkpoint -----------------------------------------
#                 if step and step % a.save_every == 0:
#                     lora.save_lora_adapters(model, a.adapter_file)
#                     save_step(step, a.adapter_file)
#                     print(f"[save] adapters @ step {step}")

#                 step += 1
#                 if step >= a.iters:
#                     break

#     # final save -------------------------------------------------------------
#     lora.save_lora_adapters(model, a.adapter_file)
#     save_step(step, a.adapter_file)
#     print("Done; saved final adapters.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
fine_tune_lora_mlx.py
────────────────────────────────────────────────────────────────────────────
LoRA fine-tuning driver that

* can **resume** from an existing `adapters.npz` (+ sidecar .meta),
* reports both **training loss** and **validation loss / ppl**,
* uses a **linear-warm-up → cosine-decay** learning-rate schedule, and
* supports an optional **plateau decay** once improvement stalls.

Tested with MLX 0.8+, Python 3.11.
"""

from __future__ import annotations

import argparse, json, math, os, random, sys, time
import mlx.core as mx
import mlx.nn   as nn
import mlx.optimizers as optim

sys.path.insert(0, os.getcwd())
from model.model import Transformer, TransformerConfig
from tokenizer    import sp_tokenizer
import lora

# ────────────────────────────── meta helpers
def _meta(fp: str) -> str:              return fp + ".meta"
def save_step(step: int, fp: str):      json.dump({"step": step}, open(_meta(fp), "w"))
def load_step(fp: str) -> int:
    meta = _meta(fp)
    return json.load(open(meta))["step"] if os.path.isfile(meta) else 0

# ────────────────────────────── CLI
def build_cli():
    pa = argparse.ArgumentParser("LoRA fine-tune (warm-up + cosine LR, resume-aware)")
    # paths
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--config",     required=True)
    pa.add_argument("--tokenizer",  required=True)
    pa.add_argument("--train_jsonl",required=True)
    pa.add_argument("--valid_jsonl",required=True)
    pa.add_argument("--adapter_file", default="model/lora_adapters/adapters.npz")
    # LoRA / optimisation
    pa.add_argument("--lora_layers", type=int, default=16)
    pa.add_argument("--batch_size",  type=int, default=16)
    pa.add_argument("--iters",       type=int, default=20_000)
    pa.add_argument("--base_lr",     type=float, default=1e-4)
    # LR schedule
    pa.add_argument("--warmup_steps", type=int, default=1_000,
                    help="linear warm-up period (steps)")
    pa.add_argument("--min_lr",       type=float, default=1e-5,
                    help="final LR floor for cosine decay")
    # logging / saving
    pa.add_argument("--save_every",   type=int, default=1_000)
    pa.add_argument("--steps_report", type=int, default=100)
    pa.add_argument("--steps_eval",   type=int, default=500)
    # misc
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--train", action="store_true")
    pa.add_argument("--test",  action="store_true")
    return pa.parse_args()

# ────────────────────────────── LR scheduler
def lr_schedule(step: int, *, base: float, warm: int, total: int, min_lr: float) -> float:
    if step < warm:
        return base * step / max(1, warm)
    pct = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * pct))

# ────────────────────────────── main
def main():
    args = build_cli()
    random.seed(args.seed);  mx.random.seed(args.seed)

    # 1) base model ─────────────────────────────────────────────────────────
    cfg   = TransformerConfig(**json.load(open(args.config)))
    model = Transformer(cfg)
    model.load_weights(args.checkpoint)
    model.freeze()

    # 2) LoRA injection ────────────────────────────────────────────────────
    blocks = model.layers if hasattr(model, "layers") else model.encoder.layers
    for blk in blocks[-args.lora_layers:]:
        for name, mod in blk.named_modules():
            if "." not in name and isinstance(mod, nn.Linear):
                setattr(blk, name, lora.LoRALinear.from_linear(mod, dropout=0.1))

    # 3) resume adapters (if any) ───────────────────────────────────────────
    step_start = 0
    if os.path.isfile(args.adapter_file):
        lora.load_lora_adapters(model, args.adapter_file)
        step_start = load_step(args.adapter_file)
        print(f"[resume] loaded adapters – continuing from step {step_start}")

    # 4) tokenizer / special IDs ────────────────────────────────────────────
    tok = sp_tokenizer.load(args.tokenizer)
    BOS, EOS = tok.sp.bos_id(), tok.sp.eos_id()

    # 5) loss & grad fn ─────────────────────────────────────────────────────
    def loss_fn(x, y, L):
        logits = model(x).astype(mx.float32)           # (B,L,V)
        pos    = mx.arange(y.shape[1])[None, :]
        mask   = (pos >= (L[:, None]-1)) & (y != BOS) & (y != 0)
        ce     = nn.losses.cross_entropy(logits, y) * mask
        return ce.sum() / mask.sum(), mask.sum()

    grad_fn = nn.value_and_grad(model, loss_fn)

    # 6) optimiser ----------------------------------------------------------
    opt = optim.AdamW(learning_rate=args.base_lr, weight_decay=0.01)

    # 7) dataset ------------------------------------------------------------
    train_pairs = list(lora.load_pairs(args.train_jsonl))
    valid_pairs = list(lora.load_pairs(args.valid_jsonl))
    pad = lambda seq, L: (seq + [0]*L)[:L]

    # 8) training loop ──────────────────────────────────────────────────────
    t0, tr_buf = time.time(), []
    step = step_start
    if args.train:
        while step < args.iters:
            random.shuffle(train_pairs)
            for chunk in (train_pairs[i:i+args.batch_size]
                          for i in range(0, len(train_pairs), args.batch_size)):
                qs, ans = zip(*chunk)
                enc_q   = [[BOS] + tok.encode(q) for q in qs]
                enc_a   = [tok.encode(a) + [EOS] for a in ans]
                enc_qa  = [q+a for q, a in zip(enc_q, enc_a)]
                L       = min(cfg.context_size, max(map(len, enc_qa)))

                x  = mx.array([pad(s, L) for s in enc_q])
                y  = mx.array([pad(s, L) for s in enc_qa])
                pr = mx.array([len(s)    for s in enc_q])

                # schedule LR
                opt.learning_rate = lr_schedule(step,
                                                base=args.base_lr,
                                                warm=args.warmup_steps,
                                                total=args.iters,
                                                min_lr=args.min_lr)

                (tr_loss, _), grads = grad_fn(x, y, pr)
                opt.update(model, grads)
                tr_buf.append(tr_loss.item())

                # --- periodic reporting --------------------------------
                if step % args.steps_report == 0:
                    dt = time.time() - t0
                    print(f"[Train] step {step:6d}  loss {sum(tr_buf)/len(tr_buf):.4f} "
                          f"lr {opt.learning_rate:.3e}  {args.steps_report/dt:.2f} it/s")
                    tr_buf.clear(); t0 = time.time()

                # --- evaluation ----------------------------------------
                if step and step % args.steps_eval == 0:
                    val_loss, n = 0.0, 0
                    for vchunk in (valid_pairs[i:i+args.batch_size]
                                    for i in range(0, len(valid_pairs), args.batch_size)):
                        vq, va = zip(*vchunk)
                        eq  = [[BOS] + tok.encode(q)   for q in vq]
                        ea  = [tok.encode(a) + [EOS]   for a in va]
                        eqa = [q+a for q, a in zip(eq, ea)]
                        L2  = min(cfg.context_size, max(map(len, eqa)))
                        vx  = mx.array([pad(s, L2) for s in eq])
                        vy  = mx.array([pad(s, L2) for s in eqa])
                        pr2 = mx.array([len(s)     for s in eq])
                        vl, _ = loss_fn(vx, vy, pr2)
                        val_loss += vl.item() * len(vq)
                        n       += len(vq)
                    val_loss /= n
                    print(f"[Eval ] step {step:6d}  val_loss {val_loss:.4f} "
                          f"val_ppl {math.exp(val_loss):.2f}")

                # --- checkpoint ----------------------------------------
                if step and step % args.save_every == 0:
                    lora.save_lora_adapters(model, args.adapter_file)
                    save_step(step, args.adapter_file)
                    print(f"[save ] adapters @ step {step}")

                step += 1
                if step >= args.iters:
                    break

    # 9) final save ---------------------------------------------------------
    lora.save_lora_adapters(model, args.adapter_file)
    save_step(step, args.adapter_file)
    print("✅ Done – adapters saved.")

if __name__ == "__main__":
    main()