#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
mkdir -p \
  "$ROOT/distributed_nanogpt_streaming/checkpoints/sft_d14_qa_v1" \
  "$ROOT/distributed_nanogpt_streaming/.hf_cache_sft/datasets"
exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$ROOT/distributed_nanogpt_streaming/hosts.json" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/distributed_nanochat_gpt2_sft/train_sft_gpt2.py" \
  --config "$ROOT/distributed_nanogpt_streaming/config_sft_d14_qa_v1.json" \
  "$@"
