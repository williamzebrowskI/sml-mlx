# Distributed Nanochat SFT

This is a `nanochat`-inspired supervised fine-tuning path for the current
4-Mac Thunderbolt ring cluster.

What it mimics from `nanochat`:
- chat/instruction SFT instead of plain LM pretraining
- assistant-only loss masking
- streaming HF conversational datasets
- distributed data-parallel training

What it does not implement yet:
- `nanochat`'s PyTorch task stack
- RL / GRPO style post-training
- sequence/model parallelism

Current recipe:
- `nanochat_like`: weighted mixture of `allenai/tulu-v2-sft-mixture` and
  `HuggingFaceH4/ultrachat_200k`

Training mode:
- full model replica on every rank
- local forward/backward on each Mac
- all-reduce gradients across the 4 Macs
- rank 0 applies optimizer updates and writes checkpoints

Launch:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanochat_sft/launch_sft.sh
```

Quick smoke run:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanochat_sft/launch_sft.sh \
  --max-steps 2 \
  --log-every 1 \
  --save-every 0
```
