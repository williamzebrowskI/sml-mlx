# Distributed nanoGPT-Style Pretraining

This is a separate MLX pretraining path that mirrors the core `nanoGPT`
workflow more closely than the repo's original transformer trainer.

What it mirrors:
- GPT-2 style architecture with learned token and position embeddings
- LayerNorm + GELU blocks
- GPT-2 style memmap token workflow using `train.bin` and `val.bin`
- `init_from` modes for `scratch`, `resume`, and `gpt2*`
- distributed data-parallel training across the 4-Mac Thunderbolt ring

What stays MLX-specific:
- MLX model/runtime instead of PyTorch
- MLX ring backend instead of PyTorch DDP
- checkpoints written only by rank 0 on `mac-1`

Files:
- `model.py`: GPT-2 style MLX model
- `train_gpt2.py`: distributed training loop
- `prepare_hf_memmap.py`: build `train.bin` and `val.bin` from HF text data
- `make_smoke_data.py`: generate a tiny synthetic token dataset

Prepare a real dataset:

```bash
/Users/williamzebrowski/sml-mlx/.venv/bin/python \
  /Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/prepare_hf_memmap.py \
  --dataset-name nvidia/Nemotron-ClimbMix \
  --train-split train \
  --token-field tokens \
  --no-append-eos \
  --output-dir /Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/data/nemotron_climbmix_gpt2
```

The default training config points at
`/Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/data/nemotron_climbmix_gpt2`,
so if you prepare a different corpus, either update `config.json` or pass
`--dataset-dir` at launch time.

Run the real training config:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/launch_train.sh
```

Run the cluster smoke test:

```bash
/Users/williamzebrowski/sml-mlx/.venv/bin/python \
  /Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/make_smoke_data.py

/Users/williamzebrowski/sml-mlx/distributed_nanogpt_pretrain/launch_smoke.sh
```

Important:
- The memmap token dataset must exist at the same absolute path on every Mac.
- Training is still data parallel, not sequence parallel.
