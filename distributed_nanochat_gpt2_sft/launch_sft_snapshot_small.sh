#!/bin/zsh
set -euo pipefail

ROOT="/Users/williamzebrowski/sml-mlx"
exec "$ROOT/.venv/bin/mlx.launch" \
  --hostfile "$ROOT/distributed_nanochat_gpt2_sft/hosts.json" \
  --backend ring \
  --python "$ROOT/.venv/bin/python" \
  -- "$ROOT/distributed_nanochat_gpt2_sft/train_sft_gpt2.py" \
  --config "$ROOT/distributed_nanochat_gpt2_sft/config_snapshot_small.json" \
  "$@"
