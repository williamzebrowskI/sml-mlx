#!/usr/bin/env python3
"""
fine_tune_lora_squad.py

LoRA fine-tuning driver for SQuAD-style QA on Apple MLX.
Reads JSONL files with fields "question", "context", "answers".
"""

import argparse
import json
import math
import os
import sys
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# ensure project root on import path
sys.path.insert(0, os.getcwd())

from model.model        import Transformer, TransformerConfig
from tokenizer          import sp_tokenizer
import lora

def build_parser():
    p = argparse.ArgumentParser("Fine-tune Transformer with LoRA on SQuAD")
    p.add_argument("--checkpoint",   required=True, help="Base model safetensors")
    p.add_argument("--config",       required=True, help="TransformerConfig JSON")
    p.add_argument("--tokenizer",    required=True, help="SentencePiece model")
    p.add_argument("--train_jsonl",  required=True, help="SQuAD v2 train JSONL")
    p.add_argument("--valid_jsonl",  required=True, help="SQuAD v2 valid JSONL")
    p.add_argument("--lora_layers",  type=int, default=16, help="Number of final layers to LoRA")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--iters",        type=int, default=2000, help="Total training steps")
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--adapter_file", default="model/lora_adapters/adapters.npz",
                   help="Path to save/load LoRA adapters")
    p.add_argument("--save_every",   type=int, default=500, help="Steps between saving adapters")
    p.add_argument("--steps_report", type=int, default=100, help="Steps between loss reports")
    p.add_argument("--steps_eval",   type=int, default=500, help="Steps between evaluations")
    p.add_argument("--train",        action="store_true")
    p.add_argument("--test",         action="store_true")
    p.add_argument("--seed",         type=int, default=0)
    return p

def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    mx.random.seed(args.seed)

    # Load & freeze base model
    cfg   = TransformerConfig(**json.load(open(args.config)))
    model = Transformer(cfg)
    model.load_weights(args.checkpoint)  # base weights
    model.freeze()

    # Loss function capturing model
    def loss_fn(x: mx.array, y: mx.array, L: mx.array):
        """
        x: input IDs array shape (B, T)
        y: target IDs array shape (B, T)
        L: lengths array shape (B,)
        Returns: (loss_scalar, token_count)
        """
        logits = model(x)  # shape (B, T, vocab)  [oai_citation:6‡hyper.ai](https://hyper.ai/cn/news/28378?utm_source=chatgpt.com)
        logits = logits.astype(mx.float32)
        # Create mask: positions < length
        mask = mx.arange(x.shape[1])[None, :] < L[:, None]  # shape (B, T)  [oai_citation:7‡hyper.ai](https://hyper.ai/cn/news/28378?utm_source=chatgpt.com)
        # Cross-entropy: nn.losses.cross_entropy expects shape (B, T, vocab) & targets (B, T)
        ce = nn.losses.cross_entropy(logits, y) * mask  # masked loss per token
        # Sum and normalize
        total = ce.sum()
        ntoks = mask.sum()
        return total / ntoks, ntoks

    grad_fn = nn.value_and_grad(model, loss_fn)  # create grad function  [oai_citation:8‡hyper.ai](https://hyper.ai/cn/news/28378?utm_source=chatgpt.com)

    # Identify transformer blocks: depends on your Transformer implementation
    if hasattr(model, "layers"):
        blocks = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        blocks = model.encoder.layers
    else:
        raise AttributeError("Transformer has no .layers or encoder.layers")

    # Inject LoRA into final N layers
    for blk in blocks[-args.lora_layers:]:
        # Traverse submodules: named_modules returns (name, module) pairs
        for name, sub in blk.named_modules():
            # Only top-level Linear modules in block (no dot in name)
            if "." not in name and isinstance(sub, nn.Linear):
                # Replace with LoRA version
                setattr(blk, name, lora.LoRALinear.from_linear(sub))

    # Optionally resume existing adapters
    if args.train and os.path.exists(args.adapter_file):
        os.makedirs(os.path.dirname(args.adapter_file), exist_ok=True)
        lora.load_lora_adapters(model, args.adapter_file)

    opt = optim.Adam(learning_rate=args.lr)  # MLX Adam optimizer  [oai_citation:9‡hyper.ai](https://hyper.ai/cn/news/28378?utm_source=chatgpt.com)
    tok = sp_tokenizer.load(args.tokenizer)

    # Load all pairs into memory (be mindful of memory; SQuAD ~130k examples)
    train_pairs = list(lora.load_pairs(args.train_jsonl))
    valid_pairs = list(lora.load_pairs(args.valid_jsonl))
    print(f"Loaded {len(train_pairs)} training examples, {len(valid_pairs)} validation examples.")

    step = 0
    # TRAINING LOOP
    if args.train:
        while step < args.iters:
            # Shuffle each epoch
            random.shuffle(train_pairs)
            # Iterate batches
            for chunk in (
                train_pairs[i : i + args.batch_size]
                for i in range(0, len(train_pairs), args.batch_size)
            ):
                qs, ans = zip(*chunk)
                # Encode question prompt and question+answer target
                enc_q   = [tok.encode(q) for q in qs]
                enc_qa  = [tok.encode(q + a) for q, a in zip(qs, ans)]
                # Determine dynamic max length (capped by context_size)
                max_len = min(cfg.context_size, max(len(s) for s in enc_qa))
                # Pad/truncate
                inp = [(s + [0] * max_len)[:max_len] for s in enc_q]
                tgt = [(s + [0] * max_len)[:max_len] for s in enc_qa]
                # Lengths = original question lengths (for masking)
                Ls  = [len(s) for s in enc_q]

                x = mx.array(inp)
                y = mx.array(tgt)
                L = mx.array(Ls)

                (lv, _), grads = grad_fn(x, y, L)
                opt.update(model, grads)

                if step % args.steps_report == 0:
                    print(f"[Train] step {step}, loss {lv.item():.4f}")

                if step and step % args.steps_eval == 0:
                    # Validation pass
                    vl, nt = 0.0, 0
                    random.shuffle(valid_pairs)
                    for vchunk in (
                        valid_pairs[i : i + args.batch_size]
                        for i in range(0, len(valid_pairs), args.batch_size)
                    ):
                        vq, va = zip(*vchunk)
                        eq     = [tok.encode(q) for q in vq]
                        eqa    = [tok.encode(q + a) for q, a in zip(vq, va)]
                        ml     = min(cfg.context_size, max(len(s) for s in eqa))
                        vx     = mx.array([(s + [0]*ml)[:ml] for s in eq])
                        vy     = mx.array([(s + [0]*ml)[:ml] for s in eqa])
                        vL     = mx.array([len(s) for s in eq])
                        lv2, ct2 = loss_fn(vx, vy, vL)
                        vl += (lv2 * ct2).item()
                        nt += ct2.item()
                    print(f"[Eval] step {step}, val_ppl {math.exp(vl/nt):.2f}")

                if step and step % args.save_every == 0:
                    os.makedirs(os.path.dirname(args.adapter_file), exist_ok=True)
                    lora.save_lora_adapters(model, args.adapter_file)
                    print(f"Saved adapters → {args.adapter_file}")

                step += 1
                if step >= args.iters:
                    break

    # TEST / FINAL EVALUATION
    if args.test:
        tl, nt = 0.0, 0
        for chunk in (
            valid_pairs[i : i + args.batch_size]
            for i in range(0, len(valid_pairs), args.batch_size)
        ):
            vq, va = zip(*chunk)
            eq     = [tok.encode(q) for q in vq]
            eqa    = [tok.encode(q + a) for q, a in zip(vq, va)]
            ml     = min(cfg.context_size, max(len(s) for s in eqa))
            vx     = mx.array([(s + [0]*ml)[:ml] for s in eq])
            vy     = mx.array([(s + [0]*ml)[:ml] for s in eqa])
            vL     = mx.array([len(s) for s in eq])
            lv3, ct3 = loss_fn(vx, vy, vL)
            tl += (lv3 * ct3).item()
            nt += ct3.item()
        print(f"[Test] ppl {math.exp(tl/nt):.2f}")

    # Final save
    os.makedirs(os.path.dirname(args.adapter_file), exist_ok=True)
    lora.save_lora_adapters(model, args.adapter_file)
    print(f"Adapters saved to {args.adapter_file}")

if __name__ == "__main__":
    main()