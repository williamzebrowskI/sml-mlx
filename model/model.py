# model/model.py
# A ~25M-parameter LM in MLX with HF streaming + ring DDP,
# using a simple byte-level tokenizer (no external SPM files).

import os
os.environ.setdefault("MX_MAX_INFLIGHT_CMDS", "1")

import math, time, json, gzip, requests, argparse
from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim

# ---------------------------
# Byte-level tokenizer + specials
# ---------------------------

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

class ByteTokenizer:
    def __init__(self):
        self.special = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.offset = len(SPECIAL_TOKENS)
        self.vocab_size = self.offset + 256  # byte tokens

    def encode(self, s: str) -> List[int]:
        ids = [self.special["<BOS>"]]
        ids += [self.offset + b for b in s.encode("utf-8")]
        ids += [self.special["<EOS>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_out: List[int] = []
        for t in ids:
            if t < self.offset:
                continue
            b = t - self.offset
            if 0 <= b < 256:
                bytes_out.append(b)
        return bytes(bytes_out).decode("utf-8", errors="ignore")

TOK = ByteTokenizer()
VOCAB_SIZE = TOK.vocab_size
PAD_ID = PAD_ID
mask_id_for_loss = PAD_ID

# ---------------------------
# Sinusoidal positions (param-free)
# ---------------------------

def sinusoidal_positions(T: int, D: int) -> mx.array:
    pos = mx.arange(T, dtype=mx.float32)[:, None]
    i   = mx.arange(D, dtype=mx.float32)[None, :]
    angle = pos / (10000 ** (2 * (i // 2) / D))
    return mx.where((i % 2) == 0, mx.sin(angle), mx.cos(angle))  # (T, D)

# ---------------------------
# Transformer (fast SDPA + cached additive mask)
# ---------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dh = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,    bias=False)
        self.max_seq = max_seq
        # cached additive causal mask (T,T) → broadcastable
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(max_seq)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        qkv = self.qkv(x)                              # (B,T,3D)
        q, k, v = mx.split(qkv, 3, axis=-1)           # (B,T,D) each

        def split_heads(t):
            t = t.reshape(B, T, self.h, self.dh)      # (B,T,H,dh)
            return t.transpose(0, 2, 1, 3)            # (B,H,T,dh)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        mask = self._mask[:T, :T]                     # (T,T) additive
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.dh), mask=mask
        )                                             # (B,H,T,dh)
        ctx = attn.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int, mlp_mult=4):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model, bias=False),
        )
    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

@dataclass
class TinyGPConfig:
    vocab_size: int
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 12
    max_seq: int = 256
    max_grad_norm: float = 1.0

class TinyGPLM(nn.Module):
    """
    Causal LM with tied output head (~25–50M params depending on vocab/dim).
    """
    def __init__(self, cfg: TinyGPConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.max_seq) for _ in range(cfg.n_layers)
        ])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.out_bias = mx.zeros((cfg.vocab_size,))
        self.pos_cache = sinusoidal_positions(cfg.max_seq, cfg.d_model)

    def logits(self, x_ids: mx.array) -> mx.array:
        B, T = x_ids.shape
        x = self.tok_emb(x_ids) + self.pos_cache[:T, :][None, :, :]
        x = self.blocks(x)
        x = self.ln_f(x)                               # (B,T,D)
        W = self.tok_emb.weight                        # (V,D)
        return mx.matmul(x, W.transpose(1, 0)) + self.out_bias  # (B,T,V)

    def __call__(self, x_ids: mx.array, targets: Optional[mx.array] = None):
        lg = self.logits(x_ids)
        if targets is None:
            return {"logits": lg}
        ce = losses.cross_entropy(lg, targets, reduction="none")   # (B,T)
        # Mask PADs (or EOS fallback set in batcher)
        mask = (targets != mask_id_for_loss).astype(mx.float32)
        loss = (ce * mask).sum() / (mask.sum() + 1e-6)
        return {"logits": lg, "loss": loss}

# ---------------------------
# HF streaming helpers
# ---------------------------

def hf_text_iterator(name, config, split, field, world_size, rank, trust_remote_code=False):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=trust_remote_code)
    i = 0
    for ex in ds:
        if i % world_size == rank:
            yield {"text": ex.get(field, "")}
        i += 1

# ---------------------------
# Distributed + utils
# ---------------------------

def init_dist(backend="ring", expected_world=None):
    group = mx.distributed.init(backend=backend)
    size = group.size() if callable(getattr(group,"size",None)) else int(getattr(group,"size",1))
    rank = group.rank() if callable(getattr(group,"rank",None)) else int(getattr(group,"rank",0))
    print(f"[boot] rank={rank} size={size} backend={backend}", flush=True)
    if expected_world is not None and size != expected_world:
        raise RuntimeError(f"Expected world size {expected_world}, got {size}")
    if rank == 0:
        print(f"[dist] backend={backend} world={size} (expected={expected_world})")
    return group, size, rank

def allreduce_grads(grads, world):
    if world == 1: return grads
    def _reduce(g):
        if isinstance(g, mx.array):
            try:
                return mx.distributed.all_sum(g, stream=mx.cpu) / world
            except TypeError:
                return mx.distributed.all_sum(g) / world
        return g
    return nn.utils.tree_map(_reduce, grads)

def clip_global(tree, max_norm):
    flats = [l for l in nn.utils.tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
    total = math.sqrt(sum(float((g**2).sum()) for g in flats))
    if max_norm <= 0 or total <= max_norm:
        return tree, total
    scale = max_norm / (total + 1e-6)
    return nn.utils.tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree), total

# ---------------------------
# Training
# ---------------------------

def train_hf_distributed(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    trust_remote_code: bool = False,
    max_steps: int = 100_000,
    seq_len: int = 256,
    batch_size: int = 1,
    accum_steps: int = 1,
    lr: float = 3e-4,
    wd: float = 0.1,
    backend: str = "ring",
    expected_world: int | None = None,
    log_every: int = 10,
    per_rank_logs: bool = False,
    save_dir: str = "model/checkpoints_spm",
    save_every: int = 5_000,
    eval_every: int = 0,
    eval_prompt: str = "Hello, world. ",
    eval_tokens: int = 50,
):
    _, world, rank = init_dist(backend=backend, expected_world=expected_world)
    print(f"[rank {rank}] entering train_hf_distributed", flush=True)

    cfg = TinyGPConfig(vocab_size=VOCAB_SIZE, d_model=384, n_heads=6, n_layers=12,
                       max_seq=seq_len, max_grad_norm=1.0)
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    print(f"[rank {rank}] model initialized (vocab={VOCAB_SIZE}, d_model={cfg.d_model}, layers={cfg.n_layers})", flush=True)

    # sync weights across ranks
    if world > 1:
        print(f"[rank {rank}] syncing initial weights across ranks", flush=True)
        synced = nn.utils.tree_map(lambda p: (mx.distributed.all_sum(p) / world) if isinstance(p, mx.array) else p,
                                   model.parameters())
        model.update(synced); mx.eval(model.parameters())

    opt = optim.AdamW(lr, weight_decay=wd)
    get_vg = nn.value_and_grad(model, lambda m, x, y: m(x, y)["loss"])

    print(f"[rank {rank}] creating HF iterator: dataset={dataset_name} config={dataset_config} split={split}", flush=True)
    sample_iter = hf_text_iterator(dataset_name, dataset_config, split, text_field, world, rank, trust_remote_code)
    print(f"[rank {rank}] HF iterator ready, starting batch loop", flush=True)

    # Batcher with SPM, padding with PAD_ID (or EOS fallback) and masking in loss
    def batch_iterator():
        X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        filled = 0
        for ex in sample_iter:
            text = ex.get("text","")
            if not text: continue
            ids = TOK.encode(text)
            if len(ids) < 2: continue
            ids = ids[: seq_len + 1]
            x_ids, y_ids = ids[:-1], ids[1:]
            if len(x_ids) < seq_len:
                pad = seq_len - len(x_ids)
                x_ids = x_ids + [PAD_ID] * pad
                y_ids = y_ids + [PAD_ID] * pad
            X[filled] = mx.array(x_ids, dtype=mx.int32)
            Y[filled] = mx.array(y_ids, dtype=mx.int32)
            filled += 1
            if filled == batch_size:
                yield X, Y
                X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
                Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
                filled = 0

    os.makedirs(save_dir, exist_ok=True)

    update_step = 0
    micro_accum = 0
    accum_grads = None
    last = time.time()

    for X, Y in batch_iterator():
        if update_step == 0 and micro_accum == 0:
            print(f"[rank {rank}] first batch received, shapes X={X.shape} Y={Y.shape}", flush=True)
        loss, grads = get_vg(model, X, Y)
        mx.eval(loss, grads)

        # NaN/Inf guards
        if bool(mx.isnan(loss)) or bool(mx.isinf(loss)):
            if rank == 0:
                print(f"[{update_step}] ⚠️ NaN/Inf loss; skipping micro-step")
            continue
        def _bad(g):
            return isinstance(g, mx.array) and (bool(mx.isnan(g).any()) or bool(mx.isinf(g).any()))
        if any(_bad(g) for g in nn.utils.tree_map(lambda x: x, grads)):
            if rank == 0:
                print(f"[{update_step}] ⚠️ NaN/Inf grads; skipping micro-step")
            continue

        # accumulate
        grads = nn.utils.tree_map(lambda g: g / accum_steps if isinstance(g, mx.array) else g, grads)
        accum_grads = grads if accum_grads is None else nn.utils.tree_map(
            lambda a, g: a + g if isinstance(a, mx.array) else a, accum_grads, grads
        )
        micro_accum += 1

        if micro_accum == accum_steps:
            global_grads = allreduce_grads(accum_grads, world)
            clipped, gnorm = clip_global(global_grads, model.cfg.max_grad_norm)
            opt.update(model, clipped)
            mx.eval(model.parameters(), opt.state)
            if update_step == 0:
                print(f"[rank {rank}] first optimizer step applied", flush=True)

            now = time.time()
            dt = now - last; last = now
            if update_step % log_every == 0:
                toks_local_s = (batch_size * seq_len * accum_steps) / max(1e-9, dt)
                ppl = math.exp(min(20.0, float(loss.item())))
                if per_rank_logs:
                    print(f"[{update_step}] rank={rank} loss={loss.item():.4f} ppl={ppl:.2f} grad_norm={gnorm:.3f} tok/s={toks_local_s:.0f}")
                if rank == 0 and not per_rank_logs:
                    print(f"[{update_step}] loss={loss.item():.4f} ppl={ppl:.2f} grad_norm={gnorm:.3f} tok/s≈{toks_local_s*world:.0f}")

            if rank == 0 and eval_every and update_step % eval_every == 0:
                try:
                    print(f"[{update_step}] sample:\n{generate(model, eval_prompt, 64)}\n---")
                except Exception as e:
                    print(f"[{update_step}] eval failed: {e}")

            if rank == 0 and update_step > 0 and update_step % save_every == 0:
                path = os.path.join(save_dir, f"ckpt_{update_step:06d}.safetensors")
                try:
                    mx.save_safetensors(path, model.parameters())
                    print(f"[{update_step}] saved {path}")
                except Exception as e:
                    print(f"[{update_step}] safetensors save failed ({e}); checkpoint NOT written")

            update_step += 1
            micro_accum = 0
            accum_grads = None
            if update_step >= max_steps:
                break

    if rank == 0:
        final_path = os.path.join(save_dir, "ckpt_final.safetensors")
        try:
            mx.save_safetensors(final_path, model.parameters())
            print(f"[final] saved {final_path}")
        except Exception as e:
            print(f"[final] safetensors save failed ({e}); final checkpoint NOT written")

# ---------------------------
# Simple greedy generator
# ---------------------------

def generate(model: TinyGPLM, prompt: str, max_new_tokens=128):
    ids = TOK.encode(prompt)[: model.cfg.max_seq]
    stop_id = EOS_ID
    x = mx.array([ids], dtype=mx.int32)
    for _ in range(max_new_tokens):
        logits = model.logits(x)[:, -1, :]            # (1,V)
        next_id = int(mx.argmax(logits, axis=-1).item())
        ids.append(next_id)
        if stop_id is not None and next_id == stop_id:
            break
        x = mx.array([ids[-model.cfg.max_seq:]], dtype=mx.int32)
    return TOK.decode(ids)

# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser("Distributed MLX LM training (byte-level tokenizer)")
    p.add_argument("--dataset", type=str, default="Skylion007/openwebtext")
    p.add_argument("--dataset-config", type=str, default="plain_text")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--trust-remote-code", action="store_true", default=True)

    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)

    p.add_argument("--backend", type=str, default="ring")
    p.add_argument("--expected-world", type=int, default=None)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--per-rank-logs", action="store_true")
    p.add_argument("--save-dir", type=str, default="model/checkpoints_spm")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-prompt", type=str, default="Hello, world. ")
    p.add_argument("--eval-tokens", type=int, default=50)
    args = p.parse_args()

    cfg = None if args.dataset_config in (None,"","None","none") else args.dataset_config
    train_hf_distributed(
        dataset_name=args.dataset,
        dataset_config=cfg,
        split=args.split,
        text_field=args.text_field,
        trust_remote_code=args.trust_remote_code,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        wd=args.wd,
        backend=args.backend,
        expected_world=args.expected_world,
        log_every=args.log_every,
        per_rank_logs=args.per_rank_logs,
        save_dir=args.save_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_prompt=args.eval_prompt,
        eval_tokens=args.eval_tokens,
    )

if __name__ == "__main__":
    main()
