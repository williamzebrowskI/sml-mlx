# Distributed nanochat-style GPT-2 SFT

This is a separate MLX-only supervised fine-tuning path that mirrors the `nanochat` SFT recipe more closely than the older local SFT trainer.

It keeps the current 4-Mac Thunderbolt ring data-parallel setup:

- full GPT-2 model replica on each Mac
- different packed SFT batches on each rank
- gradient all-reduce across the group
- rank-0-only checkpoint writing

It is aligned to the GPT-2-tokenized pretrain model from:

- `/Users/williamzebrowski/sml-mlx/distributed_nanogpt_streaming/checkpoints/run_climbmix_shuffled_v3/final.safetensors`

The default training mixture follows the upstream `nanochat` direction:

- `HuggingFaceTB/smol-smoltalk`
- `cais/mmlu` auxiliary train
- `openai/gsm8k` main train
- Karpathy identity conversations
- synthetic spelling tasks

Validation mixture:

- `smol-smoltalk` test
- `mmlu` test
- `gsm8k` test

Run:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/launch_sft.sh
```

Post-SFT eval:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/launch_chat_eval.sh
```

Useful overrides:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/launch_sft.sh \
  --max-iters 1000 \
  --save-dir /Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/checkpoints/sft_smoke
```
