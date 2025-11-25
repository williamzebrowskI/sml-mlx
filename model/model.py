# tinygp_train_mlx.py
# A ~50M-parameter LM in MLX with Hugging Face streaming + ring-distributed training across multiple Macs.

import math, time, json, gzip, io, os, random, threading, queue, requests, argparse
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
# Transformer backbone (~50M when configured below)
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
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 16
    max_seq: int = 1024
    label_smoothing: float = 0.0

class TinyGPLM(nn.Module):
    """
    Causal LM with tied output head (configured to ~50M params for training below).
    """
    def __init__(self, cfg: TinyGPConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # ~4.1M with 16k x 256
        self.blocks = nn.Sequential(*[TransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out_bias = mx.zeros((cfg.vocab_size,))  # tie output to tok_emb weight.T + bias
        # cache sinusoidal positions (non-trainable)
        self.pos_cache = sinusoidal_positions(cfg.max_seq, cfg.d_model)

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

def init_dist(backend: str = "ring", expected_world: int | None = None):
    """
    Initialize MLX distributed group and verify that all hosts are present.

    When expected_world is not None, we assert that the discovered world size
    matches (e.g., 4 Macs in your Thunderbolt ring).
    """
    try:
        group = mx.distributed.init(backend=backend)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MLX distributed backend '{backend}': {e}")

    # MLX may expose size/rank as methods; handle both attrs and callables.
    world_attr = getattr(group, "size", None)
    rank_attr = getattr(group, "rank", None)

    if callable(world_attr):
        world = int(world_attr())
    else:
        world = int(world_attr) if world_attr is not None else None

    if callable(rank_attr):
        rank = int(rank_attr())
    else:
        rank = int(rank_attr) if rank_attr is not None else None

    if world is None:
        world = int(mx.distributed.all_sum(mx.array(1)).item())
    if rank is None:
        rank = 0

    if expected_world is not None and world != expected_world:
        raise RuntimeError(f"Expected world size {expected_world}, but got {world}")

    if rank == 0:
        print(f"[dist] backend={backend} world={world} (expected={expected_world})")

    return group, world, rank

def allreduce_grads(grads, world):
    if world == 1: return grads
    return nn.utils.tree_map(lambda g: mx.distributed.all_sum(g) / world, grads)

# ---------------------------
# Training on Hugging Face streaming (distributed)
# ---------------------------


def _save_checkpoint(params, path: str):
    """
    Attempt to save in safetensors format; fall back to mx.save if unsupported.
    """
    try:
        mx.save_safetensors(path, params)
        return path, None
    except Exception as e:
        alt_path = path + ".npz"
        mx.save(alt_path, params)
        return alt_path, e


def train_hf_distributed_50m(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    trust_remote_code: bool = False,
    max_steps: int = 100_000,
    seq_len: int = 512,
    batch_size: int = 8,
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
    """
    Train a ~50M-parameter byte-level LM with MLX using Hugging Face's
    streaming datasets API, distributed across multiple Macs via a ring backend.

    Each rank receives a disjoint stream shard via hf_text_iterator(..., world_size, rank)
    and gradients are all-reduced across hosts.
    """
    _, world, rank = init_dist(backend=backend, expected_world=expected_world)

    cfg = TinyGPConfig(
        vocab_size=VOCAB,
        d_model=512,
        n_layers=16,
        n_heads=8,
        max_seq=seq_len,
    )
    model = TinyGPLM(cfg)
    mx.eval(model.parameters())

    opt = optim.AdamW(lr, weight_decay=wd)
    get_val_and_grad = nn.value_and_grad(model, lambda m, x, y: m(x, y)["loss"])

    sample_iter = hf_text_iterator(
        name=dataset_name,
        config=dataset_config,
        split=split,
        field=text_field,
        world_size=world,
        rank=rank,
        trust_remote_code=trust_remote_code,
    )

    def batch_iterator():
        X = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        Y = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        filled = 0
        for ex in sample_iter:
            # hf_text_iterator always yields {"text": "..."} regardless of the dataset field name
            text = ex.get("text")
            if not text:
                continue

            ids = TOK.encode(text)
            if len(ids) < 2:
                continue

            # next-token prediction: x is all but last, y is all but first
            ids = ids[: seq_len + 1]
            x_ids = ids[:-1]
            y_ids = ids[1:]

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

    step, t0 = 0, time.time()
    last_step_time = time.time()
    for X, Y in batch_iterator():
        loss, grads = get_val_and_grad(model, X, Y)
        grads = allreduce_grads(grads, world)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        now = time.time()
        step_time = now - last_step_time
        last_step_time = now

        if step % log_every == 0:
            # local tokens/sec per rank
            local_tok_s = (batch_size * seq_len) / max(1e-9, step_time)
            ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            if per_rank_logs:
                print(f"[{step}] rank={rank} loss={loss.item():.4f} ppl={ppl:.2f} local_toks/s={local_tok_s:.0f}")
            if rank == 0:
                # approximate global toks/sec over interval
                interval_tok_s = (batch_size * seq_len * world) / max(1e-9, (now - t0))
                print(f"[{step}] loss={loss.item():.4f} ppl={ppl:.2f} global_toks/s={interval_tok_s:.0f}")
                t0 = now

        # periodic eval on rank 0
        if rank == 0 and eval_every and step % eval_every == 0:
            try:
                sample = generate(model, eval_prompt, max_new_tokens=eval_tokens)
                print(f"[{step}] sample:\n{sample}\n---")
            except Exception as e:
                print(f"[{step}] eval sample failed: {e}")

        step += 1

        if rank == 0 and step > 0 and step % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_{step:06d}.safetensors")
            saved_path, err = _save_checkpoint(model.parameters(), ckpt_path)
            if err:
                print(f"[{step}] safetensors save failed ({err}); wrote fallback {saved_path}")
            else:
                print(f"[{step}] Saved checkpoint {saved_path}")

        if step >= max_steps:
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


def main():
    """
    CLI entrypoint:
      python -m model.model --dataset Skylion007/openwebtext --expected-world 4
    """
    parser = argparse.ArgumentParser(description="Distributed MLX 50M-parameter LM training on HF streaming data")
    parser.add_argument("--dataset", type=str, default="Skylion007/openwebtext", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset config name (e.g. 'plain_text' for OpenWebText)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text-field", type=str, default="text", help="Field containing raw text/code")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Allow HF dataset code execution")
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code", help="Disable HF dataset code execution")

    parser.add_argument("--max-steps", type=int, default=100_000, help="Total optimization steps")
    parser.add_argument("--seq-len", type=int, default=512, help="Context length")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")

    parser.add_argument("--backend", type=str, default="ring", help="MLX distributed backend")
    parser.add_argument("--expected-world", type=int, default=None, help="If set, assert world size equals this value")

    parser.add_argument("--log-every", type=int, default=10, help="Logging interval (steps)")
    parser.add_argument("--per-rank-logs", action="store_true", help="Print per-rank loss/tok/s each log interval")
    parser.add_argument("--save-dir", type=str, default="model/checkpoints_50m", help="Checkpoint directory (rank 0 only)")
    parser.add_argument("--save-every", type=int, default=5_000, help="Checkpoint interval (steps, rank 0 only)")
    parser.add_argument("--eval-every", type=int, default=2_000, help="Sampling interval (steps, rank 0 only)")
    parser.add_argument("--eval-prompt", type=str, default="def fib(n):", help="Prompt used for periodic sampling")
    parser.add_argument("--eval-tokens", type=int, default=64, help="Number of tokens to sample at eval")

    args = parser.parse_args()

    dataset_config = None
    if args.dataset_config not in (None, "", "None", "none"):
        dataset_config = args.dataset_config

    train_hf_distributed_50m(
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        split=args.split,
        text_field=args.text_field,
        trust_remote_code=args.trust_remote_code,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
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
