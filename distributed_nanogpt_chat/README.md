# nanochat-style pretrained checkpoint UI

This serves a local `nanochat`-style web UI for the final GPT-2 checkpoint:

- checkpoint: `/Users/williamzebrowski/sml-mlx/distributed_nanogpt_streaming/checkpoints/run_climbmix_shuffled_v3/final.safetensors`
- tokenizer: `gpt2`

Run:

```bash
/Users/williamzebrowski/sml-mlx/distributed_nanogpt_chat/launch_chat.sh
```

Then open:

```text
http://127.0.0.1:8000
```

Notes:

- This is a pretrained checkpoint, not an instruction-tuned assistant. The UI is chat-shaped, but the responses are still plain completions.
- The server uses tuned defaults that worked better than plain greedy decoding:
  - `temperature=0.8`
  - `top_k=40`
  - `repetition_penalty=1.2`
  - `no_repeat_ngram_size=3`
- Slash commands in the UI:
  - `/temperature`
  - `/topk`
  - `/maxtokens`
  - `/clear`
  - `/help`
