# tinygp_train_mlx.py
# A ~9M-parameter action-aware LM in MLX + streamed remote data + ring-distributed training.
# Author: You :)
# Python 3.11+, pip install mlx requests

import math, time, json, gzip, io, random, threading, queue, requests
from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple
from .utils import hf_text_iterator, hf_qa_iterator

import mlx.core as mx
import mlx.nn as nn
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

# Stub tokenizer: replace with SentencePiece or your BPE.
# For demo, we map bytes to 256 + specials (toy but works end-to-end).
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

TOK = ByteTokenizer()  # Replace with a real tokenizer later
VOCAB = TOK.vocab_size

# ---------------------------
# Positional encoding (sinusoidal: no params)
# ---------------------------

def sinusoidal_positions(T: int, D: int) -> mx.array:
    pe = mx.zeros((T, D))
    pos = mx.arange(T).astype(mx.float32)[:, None]
    i = mx.arange(D).astype(mx.float32)[None, :]
    angle = pos / (10000 ** (2 * (i // 2) / D))
    pe = mx.where((i % 2) == 0, mx.sin(angle), mx.cos(angle))
    return pe  # (T, D)

# ---------------------------
# Tiny Transformer (~9M)
# ---------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def __call__(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each (B,T,D)

        def reshape_heads(t):
            t = mx.reshape(t, (B, T, self.h, self.dh))      # (B,T,H,dh)
            return mx.transpose(t, (0, 2, 1, 3))            # (B,H,T,dh)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # scaled dot-product
        scores = mx.matmul(q, mx.transpose(k, (0,1,3,2))) / math.sqrt(self.dh)  # (B,H,T,T)

        # causal mask
        mask = mx.tril(mx.ones((T, T)))
        mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)  # (1,1,T,T)
        scores = scores + (1.0 - mask) * (-1e9)

        attn = nn.softmax(scores, axis=-1)
        ctx = mx.matmul(attn, v)                            # (B,H,T,dh)
        ctx = mx.transpose(ctx, (0,2,1,3))                  # (B,T,H,dh)
        ctx = mx.reshape(ctx, (B, T, D))                    # (B,T,D)
        return self.proj(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

@dataclass
class TinyGPConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    max_seq: int = 512
    label_smoothing: float = 0.0

class TinyGPLM(nn.Module):
    """
    ~9M param causal LM with tied output head and optional 'ponder' refinement.
    """
    def __init__(self, cfg: TinyGPConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # ~4.1M with 16k x 256
        self.blocks = nn.Sequential(*[TransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out_bias = mx.zeros((cfg.vocab_size,))  # tie output to tok_emb weight.T + bias
        # no learned pos emb to keep params <10M
        self.register_buffer("pos_cache", sinusoidal_positions(cfg.max_seq, cfg.d_model))

    def logits(self, x_ids: mx.array) -> mx.array:
        # x_ids: (B, T)
        B, T = x_ids.shape
        tok = self.tok_emb(x_ids)                                  # (B,T,D)
        pos = self.pos_cache[:T, :]                                # (T,D)
        x = tok + pos[None, :, :]
        x = self.blocks(x)
        x = self.ln_f(x)                                           # (B,T,D)
        # tied output: (B,T,D) @ (D,V) => (B,T,V)
        W = self.tok_emb.weight                                    # (V,D)
        logits = mx.matmul(x, mx.transpose(W, (1,0))) + self.out_bias
        return logits

    def __call__(self, x_ids: mx.array, targets: Optional[mx.array] = None):
        logits = self.logits(x_ids)
        if targets is None:
            return {"logits": logits}
        loss = nn.losses.cross_entropy(logits, targets, label_smoothing=0.0, reduction="mean")
        return {"logits": logits, "loss": loss}

# ---------------------------
# Remote JSONL streamer (HTTP)
# ---------------------------

class RemoteJSONLStream:
    """
    Streams .jsonl or .jsonl.gz over HTTP(S) and yields dicts without storing files.
    Each line is a JSON object with a 'text' or 'trace' field.
    """
    def __init__(self, urls: List[str], shuffle: bool = True):
        self.urls = urls
        self.shuffle = shuffle

    def _iter_url(self, url: str) -> Iterator[dict]:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            raw = r.raw
            # gzip detect
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
        buf_ids: List[int] = []
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

def init_dist(backend="ring"):
    try:
        group = mx.distributed.init(backend=backend)
    except Exception:
        group = mx.distributed.init(backend="any")
    world = getattr(group, "size", 1)
    rank = getattr(group, "rank", 0)
    if world is None: world = int(mx.distributed.all_sum(mx.array(1)).item())
    if rank is None:  rank = 0
    return group, world, rank

def allreduce_grads(grads, world):
    if world == 1: return grads
    return nn.utils.tree_map(lambda g: mx.distributed.all_sum(g) / world, grads)

# ---------------------------
# Training
# ---------------------------

def train(
    urls: list[str] | None = None,             # old path (RemoteJSONLStream)
    hf_text: dict | None = None,               # new: {"name":..., "config":..., "split":..., "field":...}
    hf_qa: dict | None = None,                 # new: {"name":..., "config":..., "split":..., "q_field":..., "a_field":...}
    max_steps: int = 50_000,
    seq_len: int = 512,
    batch_size: int = 8,
    lr: float = 3e-4,
    wd: float = 0.1,
    backend: str = "ring",
    log_every: int = 50,
    ckpt_path: str = "tgp10m.safetensors",
):
    group, world, rank = init_dist(backend)

    cfg = TinyGPConfig(vocab_size=VOCAB, d_model=256, n_layers=6, n_heads=8, max_seq=seq_len)
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())
    opt = optim.AdamW(lr, weight_decay=wd)
    get_val_and_grad = nn.value_and_grad(model, lambda m, x, y: m(x, y)["loss"])

    # ---- choose a stream ----
    if hf_text:
        sample_iter = hf_text_iterator(
            world_size=world, rank=rank, **hf_text
        )
    elif hf_qa:
        sample_iter = hf_qa_iterator(
            world_size=world, rank=rank, **hf_qa
        )
    elif urls:
        sample_iter = RemoteJSONLStream(urls)
    else:
        raise ValueError("Provide either hf_text, hf_qa, or urls for streaming.")

    batcher = PrefetchBatches(iter(sample_iter), batch_tokens=seq_len)

    # ---- batching & training stay the same ----
    def make_batch(iterator, bs):
        X = mx.zeros((bs, seq_len), dtype=mx.int32)
        Y = mx.zeros((bs, seq_len), dtype=mx.int32)
        filled = 0
        for obj in iterator:
            text = obj.get("trace") or obj.get("text")
            if not text:
                continue
            ids = TOK.encode(text)[:seq_len]
            if len(ids) < 2:
                continue
            x = ids[:-1]; y = ids[1:]
            x += [PAD] * (seq_len - len(x))
            y += [PAD] * (seq_len - len(y))
            X[filled] = mx.array(x, dtype=mx.int32)
            Y[filled] = mx.array(y, dtype=mx.int32)
            filled += 1
            if filled == bs:
                yield X, Y
                X = mx.zeros((bs, seq_len), dtype=mx.int32)
                Y = mx.zeros((bs, seq_len), dtype=mx.int32)
                filled = 0

    step, t0 = 0, time.time()
    for X, Y in make_batch(batcher, batch_size):
        loss, grads = get_val_and_grad(model, X, Y)
        grads = allreduce_grads(grads, world)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        if step % log_every == 0 and rank == 0:
            tok_s = (batch_size * seq_len) / max(1e-9, (time.time() - t0))
            print(f"[{step}] loss={loss.item():.4f} toks/s={tok_s:.0f}")
            t0 = time.time()
        step += 1
        if step >= max_steps:
            break

    if rank == 0:
        mx.save_safetensors(ckpt_path, model.parameters())
        print("Saved", ckpt_path)

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

# Orchestrator: watches for action tokens and injects results
def answer_with_tools(model: TinyGPLM, question: str, search_fn, fetch_snippet_fn, calc_fn, max_rounds=4):
    context = f"<QUESTION>{question}</QUESTION>\n<PLAN>"
    for _ in range(max_rounds):
        out = generate(model, context + "</PLAN>\n")
        # very simple parsing; improve later
        if "<SEARCH>" in out and "</SEARCH>" in out:
            q = out.split("<SEARCH>")[1].split("</SEARCH>")[0][:256]
            urls = search_fn(q)[:3]
            snippets = [fetch_snippet_fn(u)[:800] for u in urls]
            docs = "\n".join([f"<DOC>{s}</DOC>" for s in snippets if s.strip()])
            context += f"\n<SEARCH>{q}</SEARCH>\n{docs}\n"
            continue
        if "<CALC>" in out and "</CALC>" in out:
            expr = out.split("<CALC>")[1].split("</CALC>")[0][:128]
            try:
                val = calc_fn(expr)
            except Exception:
                val = "ERR"
            context += f"\n<CALC>{expr}</CALC>\n<DOC>RESULT={val}</DOC>\n"
            continue
        if "<ANSWER>" in out and "</ANSWER>" in out:
            return out.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        # fallback: extend plan
        context += "\n" + out
    return "I could not answer."