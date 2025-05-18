
# MLX PyCoder 200M

This repository contains a **200 M–parameter Python‐centric language model** built **from scratch** with [Apple MLX](https://github.com/ml-explore/mlx).  
It is designed to train on a single Apple‑silicon laptop/desktop (tested on an **M4 Pro 24 GB**) in ≈7–10 days.

## Repository layout
```
mlx-pycoder-200m/
├── data/               # raw + cleaned code
│   └── scripts/        # data download & cleaning helpers
├── tokenizer/          # SentencePiece training + vocab files
├── model/              # config, architecture, train/eval loops
├── notebooks/          # interactive eval / inspection
├── Makefile            # one‑command workflow
├── requirements.txt
└── README.md
```

## Quick start
```bash
# 1. create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. acquire + dedupe dataset (≈32 GB download → 4 B tokens)
make data

# 3. train tokenizer (50 K unigram)
make tokenizer

# 4. run training (8 B tokens, bf16, checkpoints every 1 B)
make train

# 5. evaluation (HumanEval, MBPP)
make eval
```

> **Memory note:** the default `train.py` uses `batch_size=4`, `seq_len=2 048`, `grad_accum=4` which fits in **≈18 GB** unified memory (bf16 activations).  
> Adjust `--grad_accum` and `--seq_len` if you hit OOM on a 24 GB device.

----

## Hardware & software
* Apple M‑series Mac (M3 Pro / M4 Pro with ≥24 GB RAM recommended)
* macOS 14.4+
* Python 3.12+
* `mlx` ≥ 0.3, `mlx-lm` ≥ 0.5

Happy hacking!
