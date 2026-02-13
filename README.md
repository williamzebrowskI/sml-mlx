# SML-MLX Tiny LM

Compact (~25M parameter) transformer for Apple MLX with SentencePiece-only tokenization, HF streaming data, safetensors checkpoints, and ring-based multi-host support.

## What’s here
- `model/model.py`: 12-layer Transformer (d_model=384, 6 heads) with RoPE, RMSNorm + SiLU MLP, KV-cache generation helpers, and safetensors save/load + resume logic. Contains the training loop driven by HF streaming datasets.
- `model/utils.py`: streaming dataset iterators for plain text and simple QA traces (sharded per rank).
- `tokenizer/fineweb_spm/`: SentencePiece model/vocab expected by the code; PAD/BOS/EOS ids come from the tokenizer.
- `data/`: example artifacts (encoded text/bin, OpenWebText shard, QA prompt-completion splits).
- `Makefile`: convenience targets; `hosts.json` is an example multi-host inventory.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
Python 3.11+ is required; dependencies are listed in `pyproject.toml` (MLX, mlx-lm, transformers, datasets, sentencepiece, etc.).

## Notes
- Training and generation entrypoints live in `model/model.py`; it expects `--spm-model` pointing to your SentencePiece file.
- PAD is used as the loss mask; gradient clipping is disabled when running multi-host to avoid Metal GPU timeouts.
