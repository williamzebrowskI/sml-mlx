# tinygp_train_mlx.py
# A ~25–50M parameter LM in MLX with HF streaming + ring-distributed training across multiple Macs.
# Focused on stability + avoiding Metal GPU timeouts:
#  - Uses mx.fast.scaled_dot_product_attention (optimized Metal kernel)
#  - Frequent mx.eval() to flush lazy ops
#  - Gradient all-reduce BEFORE clipping; global grad-norm clipping
#  - NaN/Inf guards on loss/gradients; skip bad micro-steps
#  - Optional smaller seq_len / accum_steps to shorten per-kernel work
#  - All-reduce on CPU stream when available (less pressure on GPU command buffers)

import math, time, json, gzip, io, os, random, threading, queue, requests, argparse
from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import mlx.optimizers as optim

# ---------------------------
# Vocab / special tokens
# ---------------------------

SPECIAL_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>",
    "<QUESTION>", "</QUESTION>",
    "<PLAN>", "</PLAN>",
    "<SEARCH>", "</SEARCH>",
    "<DOC>", "</DOC>",
    "<QUOTE>", "</QUOTE>",
    "<CALC>", "</CALC>",
    "<ANSWER>", "</ANSWER>"
]
PAD, BOS, EOS = 0, 1, 2

# Stub tokenizer: replace with SentencePiece/BPE for real training.
class ByteTokenizer:
    def __init__(self):
        self.special = {tok:i for i,tok in enumerate(SPECIAL_TOKENS)}
        self.offset = len(SPECIAL_TOKENS)
        self.vocab_size = self.offset + 256  # byte tokens
    def encode(self, s: str) -> List[int]:
        return [self.special["<BOS>"]] + [self.offset + b for b in s.encode("utf-8")] + [self.special["<EOS>"]]
    def decode(self, ids: List[int]) -> str:
        bytes_out = []
        for t in ids:
            if t < self.offset: continue
            b = t - self.offset
            if 0 <= b < 256:
                bytes_out.append(b)
        return bytes(bytes_out).decode("utf-8", errors="ignore")

TOK = ByteTokenizer()
VOCAB = TOK.vocab_size

# ---------------------------
# Positional encoding (sinusoidal: no params)
# ---------------------------

def sinusoidal_positions(T: int, D: int) -> mx.array:
    pos = mx.arange(T, dtype=mx.float32)[:, None]
    i = mx.arange(D, dtype=mx.float32)[None, :]
    angle = pos / (10000 ** (2 * (i // 2) / D))
    pe = mx.where((i % 2) == 0, mx.sin(angle), mx.cos(angle))
    return pe  # (T, D)

# ---------------------------
# Transformer (optimized SDPA)
# ---------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # cache a big additive causal mask; slice per T
        self._mask_cache_T = None
        self._mask_cache = None
        self.max_seq = max_seq

    def _get_mask(self, T: int) -> mx.array:
        if self._mask_cache is None or self._mask_cache_T < T:
            # additive causal mask (broadcastable), shape (T, T) -> (1,1,T,T)
            m = nn.MultiHeadAttention.create_additive_causal_mask(self.max_seq)
            self._mask_cache = m  # (max_seq, max_seq) additive
            self._mask_cache_T = self.max_seq
        return self._mask_cache[:T, :T]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each (B,T,D)

        def reshape_heads(t):
            t = t.reshape(B, T, self.h, self.dh)     # (B,T,H,dh)
            return t.transpose(0, 2, 1, 3)           # (B,H,T,dh)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        mask = self._get_mask(T)  # (T,T) additive mask
        # mx.fast.scaled_dot_product_attention expects (B,H,L,D), additive mask broadcastable
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=1.0 / math.sqrt(self.dh),
            mask=mask
        )  # (B,H,T,dh)

        ctx = attn.transpose(0, 2, 1, 3).reshape(B, T, D)  # (B,T,D)
        return self.proj(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
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
    max_seq: int = 512
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0

class TinyGPLM(nn.Module):
    """
    Causal LM with tied output head (~25M params with ByteTokenizer; larger with BPE/SPM).
    """
    def __init__(self, cfg: TinyGPConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.max_seq) for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out_bias = mx.zeros((cfg.vocab_size,))
        # cache sinusoidal positions (non-trainable)
        self.pos_cache = sinusoidal_positions(cfg.max_seq, cfg.d_model)

    def logits(self, x_ids: mx.array) -> mx.array:
        B, T = x_ids.shape
        tok = self.tok_emb(x_ids)                           # (B,T,D)
        pos = self.pos_cache[:T, :]                         # (T,D)
        x = tok + pos[None, :, :]
        x = self.blocks(x)
        x = self.ln_f(x)                                    # (B,T,D)
        W = self.tok_emb.weight                             # (V,D)
        return mx.matmul(x, W.transpose(1, 0)) + self.out_bias  # (B,T,V)

    def __call__(self, x_ids: mx.array, targets: Optional[mx.array] = None):
        lg = self.logits(x_ids)
        if targets is None:
            return {"logits": lg}
        ce = losses.cross_entropy(lg, targets, reduction="none")  # (B,T)
        mask = (targets != PAD).astype(mx.float32)
        denom = mask.sum() + 1e-6
        loss = (ce * mask).sum() / denom
        return {"logits": lg, "loss": loss}

# ---------------------------
# Remote JSONL streamer (HTTP)
# ---------------------------

class RemoteJSONLStream:
    def __init__(self, urls: List[str], shuffle: bool = True):
        self.urls = urls
        self.shuffle = shuffle

    def _iter_url(self, url: str) -> Iterator[dict]:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            raw = r.raw
            if url.endswith(".gz"):
                gz = gzip.GzipFile(fileobj=raw)
                for line in gz:
                    if not line.strip(): continue
                    yield json.loads(line.decode("utf-8"))
            else:
                for line in raw:
                    if not line.strip(): continue
                    yield json.loads(line.decode("utf-8"))

    def __iter__(self) -> Iterator[dict]:
        while True:
            urls = self.urls[:]
            if self.shuffle:
                random.shuffle(urls)
            for u in urls:
                for obj in self._iter_url(u):
                    yield obj

class PrefetchBatches:
    def __init__(self, sample_iter: Iterator[dict], batch_tokens: int, max_prefetch: int = 4):
        self.sample_iter = sample_iter
        self.batch_tokens = batch_tokens
        self.q = queue.Queue(max_prefetch)
        self.th = threading.Thread(target=self._runner, daemon=True)
        self.th.start()

    def _encode(self, txt: str) -> List[int]:
        return TOK.encode(txt)

    def _runner(self):
        for obj in self.sample_iter:
            text = obj.get("trace") or obj.get("text")
            if not text: continue
            ids = self._encode(text)
            # chunk into fixed windows
            for i in range(0, len(ids), self.batch_tokens):
                seq = ids[i : i + self.batch_tokens]
                self.q.put(seq)
        self.q.put(None)

    def __iter__(self):
        while True:
            seq = self.q.get()
            if seq is None: break
            yield seq

# ---------------------------
# Distributed helpers
# ---------------------------

def init_dist(backend: str = "ring", expected_world: int | None = None):
    try:
        group = mx.distributed.init(backend=backend)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MLX distributed backend '{backend}': {e}")

    world_attr = getattr(group, "size", None)
    rank_attr  = getattr(group, "rank", None)
    world = int(world_attr()) if callable(world_attr) else int(world_attr or 1)
    rank  = int(rank_attr())  if callable(rank_attr)  else int(rank_attr or 0)

    if expected_world is not None and world != expected_world:
        raise RuntimeError(f"Expected world size {expected_world}, but got {world}")
    if rank == 0:
        print(f"[dist] backend={backend} world={world} (expected={expected_world})")
        if os.environ.get("MX_MAX_INFLIGHT_CMDS") is None:
            print("[dist] hint: export MX_MAX_INFLIGHT_CMDS=2 to reduce Metal queue depth")

    return group, world, rank

def allreduce_grads(grads, world):
    if world == 1:
        return grads
    def _reduce(g):
        if isinstance(g, mx.array):
            try:
                return mx.distributed.all_sum(g, stream=mx.cpu) / world
            except TypeError:
                return mx.distributed.all_sum(g) / world
        return g
    return nn.utils.tree_map(_reduce, grads)

def clip_global(tree, max_norm: float):
    flats = [l for l in nn.utils.tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
    total = math.sqrt(sum(float((g**2).sum()) for g in flats))
    if total <= max_norm or max_norm <= 0:
        return tree, total
    scale = max_norm / (total + 1e-6)
    return nn.utils.tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree), total

# ---------------------------
# Training (HF streaming + ring DDP)
# ---------------------------

def _save_checkpoint(params, path: str):
    try:
        mx.save_safetensors(path, params); return path, None
    except Exception as e:
        alt_path = path + ".npz"; mx.save(alt_path, params); return alt_path, e

def train_hf_distributed_50m(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    trust_remote_code: bool = False,
    max_steps: int = 100_000,
    seq_len: int = 512,
    batch_size: int = 8,
    accum_steps: int = 1,
    lr: float = 3e-4,
    wd: float = 0.1,
    backend: str = "ring",
    expected_world: int | None = None,
    log_every: int = 10,
    per_rank_logs: bool = False,
    save_dir: str = "model/checkpoints_50m",
    save_every: int = 5_000,
    eval_every: int = 2_000,
    eval_prompt: str = "def fib(n):",
    eval_tokens: int = 64,
):
    _, world, rank = init_dist(backend=backend, expected_world=expected_world)

    cfg = TinyGPConfig(
        vocab_size=VOCAB,
        d_model=384,
        n_layers=12,
        n_heads=6,
        max_seq=seq_len,
        max_grad_norm=1.0,
    )
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())

    # synchronize weights
    if world > 1:
        synced = nn.utils.tree_map(
            lambda p: (mx.distributed.all_sum(p) / world) if isinstance(p, mx.array) else p,
            model.parameters()
        )
        model.update(synced); mx.eval(model.parameters())

    opt = optim.AdamW(lr, weight_decay=wd)
    # numerically stable masked loss
    def masked_loss(m: TinyGPLM, X: mx.array, Y: mx.array) -> mx.array:
        out = m(X, Y)  # already masked internally
        return out["loss"]

    get_val_and_grad = nn.value_and_grad(model, masked_loss)

    # Build stream-sharded iterator (user-provided util recommended; fallback below)
    try:
        from .utils import hf_text_iterator
        sample_iter = hf_text_iterator(
            name=dataset_name, config=dataset_config, split=split,
            field=text_field, world_size=world, rank=rank, trust_remote_code=trust_remote_code,
        )
    except Exception:
        # Fallback: simple HF load_dataset(streaming=True) with local shard filter
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True, trust_remote_code=trust_remote_code)
        # naive sharding: take every world-th sample starting from rank
        def _fallback_iter():
            i = 0
            for ex in ds:
                if i % world == rank:
                    yield {"text": ex.get(text_field, "")}
                i += 1
        sample_iter = _fallback_iter()

    # batcher
    def batch_iterator():
        X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        filled = 0
        for ex in sample_iter:
            text = ex.get("text", "")
            if not text:
                continue
            ids = TOK.encode(text)
            if len(ids) < 2:
                continue
            ids = ids[: seq_len + 1]
            x_ids, y_ids = ids[:-1], ids[1:]
            if len(x_ids) < seq_len:
                pad_len = seq_len - len(x_ids)
                x_ids = x_ids + [PAD] * pad_len
                y_ids = y_ids + [PAD] * pad_len
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
    last_update_time = time.time()
    micro_accum = 0
    accum_grads = None

    for X, Y in batch_iterator():
        loss, grads = get_val_and_grad(model, X, Y)
        mx.eval(loss, grads)  # flush to catch issues early

        # NaN/Inf guard on loss
        if bool(mx.isnan(loss)) or bool(mx.isinf(loss)):
            if rank == 0:
                print(f"[{update_step}] ⚠️ NaN/Inf loss, skipping micro-step")
            continue

        # Optional: NaN/Inf guard on a few grad leaves
        def _has_bad(g):
            if not isinstance(g, mx.array): return False
            return bool(mx.isnan(g).any()) or bool(mx.isinf(g).any())
        bad_grad = any(_has_bad(g) for g in nn.utils.tree_map(lambda x: x, grads))
        if bad_grad:
            if rank == 0:
                print(f"[{update_step}] ⚠️ NaN/Inf gradients, skipping micro-step")
            continue

        # accumulate (scale by accum_steps to keep update magnitude)
        grads = nn.utils.tree_map(lambda g: g / accum_steps if isinstance(g, mx.array) else g, grads)
        accum_grads = grads if accum_grads is None else nn.utils.tree_map(
            lambda a, g: a + g if isinstance(a, mx.array) else a, accum_grads, grads
        )
        micro_accum += 1

        if micro_accum == accum_steps:
            # global all-reduce on CPU stream if available
            global_grads = allreduce_grads(accum_grads, world)
            # clip global grad
            clipped, gnorm = clip_global(global_grads, model.cfg.max_grad_norm)
            # apply update
            opt.update(model, clipped)
            mx.eval(model.parameters(), opt.state)

            # logging
            now = time.time()
            step_time = now - last_update_time
            last_update_time = now
            if update_step % log_every == 0:
                local_tok_s = (batch_size * seq_len * accum_steps) / max(1e-9, step_time)
                ppl = math.exp(min(20.0, float(loss.item())))  # safe exp
                if per_rank_logs:
                    print(f"[{update_step}] rank={rank} loss={loss.item():.4f} ppl={ppl:.2f} grad_norm={gnorm:.3f} local_toks/s={local_tok_s:.0f}")
                if rank == 0:
                    global_tok_s = local_tok_s * world
                    print(f"[{update_step}] loss={loss.item():.4f} ppl={ppl:.2f} grad_norm={gnorm:.3f} global_toks/s={global_tok_s:.0f}")

            # periodic sampling (rank 0 only)
            if rank == 0 and eval_every and update_step > 0 and update_step % eval_every == 0:
                try:
                    sample = generate(model, eval_prompt, max_new_tokens=eval_tokens)
                    print(f"[{update_step}] sample:\n{sample}\n---")
                except Exception as e:
                    print(f"[{update_step}] eval sample failed: {e}")

            # checkpoint (rank 0)
            if rank == 0 and update_step > 0 and update_step % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_{update_step:06d}.safetensors")
                saved_path, err = _save_checkpoint(model.parameters(), ckpt_path)
                if err:
                    print(f"[{update_step}] safetensors save failed ({err}); wrote fallback {saved_path}")
                else:
                    print(f"[{update_step}] Saved checkpoint {saved_path}")

            update_step += 1
            micro_accum = 0
            accum_grads = None

            if update_step >= max_steps:
                break

    if rank == 0:
        final_path = os.path.join(save_dir, "ckpt_final.safetensors")
        saved_path, err = _save_checkpoint(model.parameters(), final_path)
        if err:
            print(f"[final] safetensors save failed ({err}); wrote fallback {saved_path}")
        else:
            print(f"Saved final checkpoint {saved_path}")

# ---------------------------
# Tiny agent runner (inference)
# ---------------------------

def generate(model: TinyGPLM, prompt: str, max_new_tokens=128):
    ids = TOK.encode(prompt)[: model.cfg.max_seq]
    x = mx.array([ids], dtype=mx.int32)
    for _ in range(max_new_tokens):
        logits = model.logits(x)[:, -1, :]  # (1,V)
        next_id = int(mx.argmax(logits, axis=-1).item())
        ids.append(next_id)
        if next_id == TOK.special["<EOS>"]: break
        x = mx.array([ids[-model.cfg.max_seq:]], dtype=mx.int32)
    return TOK.decode(ids)

def answer_with_tools(model: TinyGPLM, question: str, search_fn, fetch_snippet_fn, calc_fn, max_rounds=4):
    context = f"<QUESTION>{question}</QUESTION>\n<PLAN>"
    for _ in range(max_rounds):
        out = generate(model, context + "</PLAN>\n")
        if "<SEARCH>" in out and "</SEARCH>" in out:
            q = out.split("<SEARCH>")[1].split("</SEARCH>")[0][:256]
            urls = search_fn(q)[:3]
            snippets = [fetch_snippet_fn(u)[:800] for u in urls]
            docs = "\n".join([f"<DOC>{s}</DOC>" for s in snippets if s.strip()])
            context += f"\n<SEARCH>{q}</SEARCH>\n{docs}\n"; continue
        if "<CALC>" in out and "</CALC>" in out:
            expr = out.split("<CALC>")[1].split("</CALC>")[0][:128]
            try: val = calc_fn(expr)
            except Exception: val = "ERR"
            context += f"\n<CALC>{expr}</CALC>\n<DOC>RESULT={val}</DOC>\n"; continue
        if "<ANSWER>" in out and "</ANSWER>" in out:
            return out.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        context += "\n" + out
    return "I could not answer."

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Distributed MLX ~25–50M LM training on HF streaming")
    parser.add_argument("--dataset", type=str, default="Skylion007/openwebtext")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")

    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--seq-len", type=int, default=512)        # Consider 256 to reduce per-kernel time
    parser.add_argument("--batch-size", type=int, default=8)       # Per-rank
    parser.add_argument("--accum-steps", type=int, default=1)      # Increase to smooth gradients; keep small to reduce kernel time
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.1)

    parser.add_argument("--backend", type=str, default="ring")
    parser.add_argument("--expected-world", type=int, default=None)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--per-rank-logs", action="store_true")
    parser.add_argument("--save-dir", type=str, default="model/checkpoints_50m")
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--eval-prompt", type=str, default="def fib(n):")
    parser.add_argument("--eval-tokens", type=int, default=64)

    args = parser.parse_args()
    dataset_config = None if args.dataset_config in (None, "", "None", "none") else args.dataset_config

    train_hf_distributed_50m(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
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