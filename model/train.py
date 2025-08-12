# model/train.py
# MLX distributed trainer (ring) with loud logging + robust HF streaming:
# - 429 Too Many Requests → exponential backoff + jitter, then resume
# - rank-staggered start to spread out Hub load
# - optional HF token login to increase Hub rate limit
# - fp16/bf16-safe (attention + CE in fp32), grad accumulation, all-reduce, clip

from __future__ import annotations
import argparse, pathlib, math, json, time, itertools, socket, os, random
from typing import Iterator, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.nn.losses as losses
from mlx.utils import tree_map

from datasets import load_dataset, DownloadConfig
import sentencepiece as spm
import numpy as np
import wandb

# for catching Hub throttling cleanly
from huggingface_hub.utils import HfHubHTTPError
try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None

from .model import OpenELM, SMLMConfig

# ───────────────────────────────────
# tiny logger
def log(rank, *msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [Rank {rank}]", *msg, flush=True)

# ───────────────────────────────────
def _barrier() -> None:
    mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32)))

def _broadcast_params(params, rank: int) -> None:
    # poor-man's broadcast: zero on non-root, sum-reduce
    for p in tree_map(lambda x: x, params):
        if not isinstance(p, mx.array):
            continue
        if rank != 0:
            p[...] = 0
        p[...] = mx.distributed.all_sum(p)

def encode_sp(example: Dict[str, Any], *, sp: spm.SentencePieceProcessor, key: str):
    ids = sp.encode(example[key], out_type=int, add_bos=True, add_eos=True)
    return {"ids": ids}

def resilient_dataset_iter(ds, rank: int, *, backoff_max: float = 60.0):
    """Iterate a streaming HF dataset and handle transient 429/HTTP hiccups."""
    backoff = 2.0
    while True:
        it = iter(ds)
        try:
            for ex in it:
                yield ex
            # exhausted (rare for streaming) → restart loop
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                sleep_for = min(backoff, backoff_max) * (1.0 + random.random())
                log(rank, f"HF 429 Too Many Requests → sleeping {sleep_for:.1f}s then retrying stream …")
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, backoff_max)
                continue
            # Retry some other transient classes, too
            if code in {500, 502, 503, 504, None}:
                sleep_for = 5.0 * (1.0 + 0.5 * random.random())
                log(rank, f"HF transient {code or 'error'} → backoff {sleep_for:.1f}s")
                time.sleep(sleep_for)
                continue
            # otherwise bubble up
            raise
        except Exception as e:
            # Generic network hiccup
            sleep_for = 5.0 * (1.0 + 0.5 * random.random())
            log(rank, f"dataset iterator error: {type(e).__name__}: {e} → retry in {sleep_for:.1f}s")
            time.sleep(sleep_for)
            continue

def sample_generator(dataset_iter: Iterator[Dict[str, Any]], ctx: int, bs: int) -> Iterator[mx.array]:
    """Packs streaming tokenized examples into fixed windows of length ctx+1."""
    window, buf = ctx + 1, []
    while True:
        for ex in dataset_iter:
            buf.extend(ex["ids"])
            while len(buf) >= window * bs:
                chunk = np.asarray(buf[: window * bs], dtype=np.int32)
                del buf[: window * bs]
                yield mx.array(chunk).reshape(bs, window)

def cosine_lr(step, *, base, warmup, total, min_lr):
    if step < warmup:
        return base * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))

def clip_global(tree, max_norm):
    flats = [l for l in tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
    total = math.sqrt(sum(float((g**2).sum()) for g in flats)) if flats else 0.0
    if total <= max_norm:
        return tree
    scale = max_norm / (total + 1e-6)
    return tree_map(lambda x: x * scale if isinstance(x, mx.array) else x, tree)

def get_args():
    p = argparse.ArgumentParser("OpenSML MLX trainer")
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--dataset", required=True, help="HF dataset id (streaming)")
    p.add_argument("--dataset-config", default=None, help="HF dataset config name")
    p.add_argument("--train-split", default="train", help="split or slice, e.g. 'train', 'train[:1%]'")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--resume")
    return p.parse_args()

# ───────────────────────────────────
def main():
    print(f"[BOOT] host={socket.gethostname()} rank_env={os.getenv('MLX_RANK')}")
    log(-1, "starting distributed init; MLX_HOSTS=", os.getenv("MLX_HOSTS"), "MLX_PORT=", os.getenv("MLX_PORT"))
    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    log(rank, f"init OK ({rank+1}/{size}) on host {socket.gethostname()}")

    args = get_args()
    cfg = SMLMConfig.from_json(args.config)

    # Optional: Hugging Face token to raise rate limits
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token and hf_login is not None:
        try:
            hf_login(token=token, add_to_git_credential=False)
            log(rank, "Hugging Face token loaded (higher Hub rate limit).")
        except Exception as e:
            log(rank, f"Hugging Face login failed (continuing anonymous): {e!r}")
    else:
        log(rank, "No HF token in env; using anonymous Hub access.")

    # device & default dtype
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    if hasattr(mx, "set_default_dtype"):
        if cfg.torch_dtype in ("float16", "fp16"):
            mx.set_default_dtype(mx.float16)
        elif cfg.torch_dtype == "bfloat16":
            mx.set_default_dtype(mx.bfloat16)
    try:
        default_dtype = mx.default_dtype()
    except Exception:
        default_dtype = "n/a"
    log(rank, f"device={'gpu' if args.device=='gpu' else 'cpu'} default_dtype={default_dtype}")

    # tokenizer + vocab sanity
    log(rank, f"loading tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_from_sp = int(sp.get_piece_size())
    if cfg.vocab_size != vocab_from_sp:
        cfg.vocab_size = vocab_from_sp
        log(rank, f"vocab_size adjusted to tokenizer size: {cfg.vocab_size}")
    pad_tok = sp.piece_to_id("<pad>")
    pad_id = pad_tok if pad_tok >= 0 else -100
    log(rank, f"tokenizer loaded. vocab={vocab_from_sp} pad_id={pad_tok}")

    # micro-batch & accumulation (can override via env)
    LOCAL_BS = int(os.getenv("LOCAL_BS", cfg.local_bs))
    ACCUM_STEPS = int(os.getenv("ACCUM_STEPS", cfg.accum_steps))
    SHUFFLE_BUF = int(os.getenv("SHUFFLE_BUFFER", 20000))
    BACKOFF_MAX = float(os.getenv("HF_BACKOFF_MAX", "60"))
    STAGGER = float(os.getenv("RANK_STAGGER_SEC", str(rank * 2.0)))  # default: 2s * rank
    log(rank, f"LOCAL_BS={LOCAL_BS} ACCUM_STEPS={ACCUM_STEPS} SHUFFLE_BUFFER={SHUFFLE_BUF} context={cfg.context_size}")
    log(rank, f"rank-stagger={STAGGER:.1f}s, backoff_max={BACKOFF_MAX:.0f}s")

    # Give each rank a small stagger to avoid synchronized bursts on the Hub
    if STAGGER > 0:
        time.sleep(STAGGER)

    # streaming dataset
    download_config = DownloadConfig(max_retries=5)
    for attempt in range(1, 6):
        try:
            log(rank, f"loading dataset {args.dataset} cfg={args.dataset_config} split={args.train_split} (streaming=True) try {attempt}/5")
            ds = load_dataset(
                args.dataset,
                args.dataset_config,
                split=args.train_split,
                streaming=True,
                download_config=download_config,
                trust_remote_code=True,
            )
            log(rank, "dataset stream acquired")
            break
        except Exception as e:
            log(rank, f"load_dataset failed: {e!r}")
            if attempt == 5:
                raise
            time.sleep(5)

    ds = ds.map(lambda ex: encode_sp(ex, sp=sp, key="text"))
    ds = ds.shard(num_shards=size, index=rank, contiguous=True)
    ds = ds.shuffle(seed=42 + rank, buffer_size=SHUFFLE_BUF)
    log(rank, "dataset mapped/sharded/shuffled; building packer")

    # offset handling for restarts
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    offset_file = out_dir / "offset.txt"
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip() or "0")
        except Exception:
            offset = 0
    else:
        offset = 0
    log(rank, f"skipping first {offset:,} global tokens (if any)")

    tokens_per_rank_mb = LOCAL_BS * (cfg.context_size + 1)
    rank_offset_tokens = offset // size
    skip_batches = rank_offset_tokens // tokens_per_rank_mb

    # Wrap the dataset with our resilient iterator to survive 429s
    ds_iter = resilient_dataset_iter(ds, rank, backoff_max=BACKOFF_MAX)

    train_it = itertools.islice(
        sample_generator(ds_iter, cfg.context_size, LOCAL_BS),
        skip_batches,
        None
    )
    log(rank, f"packer ready. skip_batches={skip_batches}")

    # model + optimizer
    model = OpenELM(cfg)
    opt = optim.AdamW(cfg.max_lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=cfg.weight_decay)

    start_step = 0
    if args.resume and rank == 0:
        try:
            model.load_weights(args.resume)
            meta = out_dir / "meta.json"
            if meta.exists():
                start_step = json.loads(meta.read_text()).get("global_step", 0)
            log(rank, f"resumed from {args.resume} at step {start_step}")
        except Exception as e:
            log(rank, f"resume failed: {e!r}")

    # broadcast params
    _barrier()
    _broadcast_params(model.parameters(), rank)
    log(rank, f"weights synced – entering compile")

    # numerically-stable loss (logits upcast to fp32)
    def loss_fn(m, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = m(x).astype(mx.float32)
        if pad_id >= 0:
            valid = (y != pad_id).astype(mx.float32)
            valid_sum = valid.sum()
        else:
            valid = mx.ones_like(y, dtype=mx.float32)
            valid_sum = float(y.size)
        ce = losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).reshape(*y.shape)
        loss = (ce * valid).sum() / (valid_sum + 1e-6)
        return loss

    value_and_grad = nn.value_and_grad(model, loss_fn)

    # warm-up compile
    log(rank, "warming up compile for value_and_grad() …")
    _ = value_and_grad(model, mx.array(np.zeros((1, 4), dtype=np.int32))); mx.eval(_)
    log(rank, "compile done; starting training loop")

    if rank == 0:
        wandb.init(
            project="fineweb-pretrain",
            config={**cfg.__dict__, "LOCAL_BS": LOCAL_BS, "ACCUM_STEPS": ACCUM_STEPS, "world_size": size},
            name=f"pretrain-{start_step:06d}",
        )

    def compute_grad_norm(tree) -> float:
        flats = [l for l in tree_map(lambda x: x, tree) if isinstance(l, mx.array)]
        return math.sqrt(sum(float((g**2).sum()) for g in flats)) if flats else 0.0

    acc_l = acc_s = 0
    accum_grads = None
    micro_step = 0
    RESTART_WARM = 10_000  # short warm restart

    # throughput accounting
    toks_per_update = size * LOCAL_BS * (cfg.context_size + 1) * ACCUM_STEPS
    last_log_t = time.time()
    log(rank, f"effective tokens/update ≈ {toks_per_update:,}")

    for global_step in range(start_step + 1, cfg.max_iterations + 1):
        # LR schedule
        if global_step < start_step + RESTART_WARM:
            opt.learning_rate = cfg.max_lr * (global_step - start_step) / max(1, RESTART_WARM)
        else:
            opt.learning_rate = cosine_lr(
                global_step, base=cfg.max_lr,
                warmup=cfg.warmup_iterations,
                total=cfg.max_iterations,
                min_lr=cfg.min_lr,
            )

        # fetch batch and compute loss+grads
        batch = next(train_it)
        loss, grads = value_and_grad(model, batch); mx.eval(loss, grads)

        # detect/skip NaN loss to keep accumulation clean
        if bool(mx.isnan(loss)) or bool(mx.isinf(loss)):
            if rank == 0:
                x, y = batch[:, :-1], batch[:, 1:]
                valid_sum = (y != pad_id).sum() if pad_id >= 0 else y.size
                log(rank, f"⚠️ NaN/Inf loss detected. valid_sum={valid_sum}")
            continue

        # scale for accumulation
        grads = tree_map(lambda g: g / ACCUM_STEPS if isinstance(g, mx.array) else g, grads)
        accum_grads = grads if accum_grads is None else tree_map(
            lambda a, g: a + g if isinstance(a, mx.array) else a, accum_grads, grads
        )
        micro_step += 1

        if micro_step == ACCUM_STEPS:
            # all-reduce grads
            global_grads = tree_map(lambda g: mx.distributed.all_sum(g), accum_grads)
            grad_norm = compute_grad_norm(global_grads)
            # clean & clip
            global_grads = tree_map(
                lambda g: mx.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4) if isinstance(g, mx.array) else g,
                global_grads
            )
            global_grads = clip_global(global_grads, cfg.grad_clip)
            # apply
            opt.update(model, global_grads); mx.eval(model.parameters())
            accum_grads = None; micro_step = 0

            acc_l += float(loss); acc_s += 1

            # logging & checkpoints
            if rank == 0:
                now = time.time()
                if global_step % 10 == 0:
                    avg_loss = acc_l / max(1, acc_s)
                    ppl = math.exp(avg_loss)
                    dt = max(1e-6, now - last_log_t)
                    updates_per_sec = 10.0 / dt
                    tokens_per_sec = updates_per_sec * toks_per_update
                    last_log_t = now
                    print(
                        f"[{time.strftime('%H:%M:%S')}] [Rank 0] "
                        f"step={global_step} loss={avg_loss:.4f} ppl={ppl:.2f} "
                        f"lr={opt.learning_rate:.2e} grad_norm={grad_norm:.3f} "
                        f"updates/s={updates_per_sec:.2f} tokens/s≈{tokens_per_sec:,.0f}",
                        flush=True
                    )
                    wandb.log({
                        "train/loss": float(avg_loss),
                        "train/perplexity": float(ppl),
                        "train/lr": float(opt.learning_rate),
                        "train/grad_norm": float(grad_norm),
                        "train/updates_per_sec": float(updates_per_sec),
                        "train/tokens_per_sec": float(tokens_per_sec),
                    }, step=int(global_step))
                    acc_l = acc_s = 0

                if global_step % 5000 == 0:
                    ckpt_path = out_dir / f"ckpt_{global_step:06d}.safetensors"
                    model.save_weights(str(ckpt_path))
                    (out_dir / "meta.json").write_text(json.dumps({"global_step": int(global_step)}))
                    processed = global_step * (size * LOCAL_BS * (cfg.context_size + 1))
                    (out_dir / "offset.txt").write_text(str(processed))
                    log(rank, f"✔ saved {ckpt_path.name} | offset={processed:,}")

    if rank == 0:
        model.save_weights(str(out_dir / "ckpt_final.safetensors"))
        final_offset = cfg.max_iterations * (size * LOCAL_BS * (cfg.context_size + 1))
        (out_dir / "offset.txt").write_text(str(final_offset))
        log(rank, "✅ Training complete")

if __name__ == "__main__":
    main()