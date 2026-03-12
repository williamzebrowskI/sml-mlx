# Distributed Streaming nanoGPT-Style Pretraining

This is a separate GPT-2 style MLX pretraining path for live Hugging Face
streaming datasets. It keeps the nanoGPT-like model and optimizer recipe, but
replaces memmap token files with a streaming token-window sampler.

What it does:
- local GPT-2 style MLX model in `distributed_nanogpt_streaming/model.py`
- live HF streaming of tokenized datasets such as `nvidia/Nemotron-ClimbMix`
- per-rank token-offset sampling across the 4-Mac Thunderbolt ring
- contiguous token-window packing into `x` / `y` batches
- rank-0-only model checkpoints by default

What is different from the memmap path:
- no `train.bin` / `val.bin` requirement
- training can start without downloading the whole corpus first
- exact resume of stream position is only available if `save_stream_state=true`
- with `shuffle_buffer>0`, resume is not exact

Default dataset:
- `nvidia/Nemotron-ClimbMix`
- source field: `tokens`

Run the real config:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanogpt_streaming/launch_train.sh
```

Run the 4-node smoke test:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanogpt_streaming/launch_smoke.sh
```

Important:
- This is still data parallel training.
- The first live batch on a cold host can take noticeably longer while the HF stream metadata resolves.
- Workers do not save checkpoint files unless you explicitly enable stream-state saves.
