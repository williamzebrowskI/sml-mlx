# file: pp_local_demo.py
import argparse, mlx.core as mx
from mlx_lm import load, stream_generate

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="openai/gpt-oss-20b")
parser.add_argument("--prompt", default="Say hi and tell me a fun fact.")
parser.add_argument("--max-tokens", type=int, default=128)
args = parser.parse_args()

group = mx.distributed.init(backend="ring")  # or "any" / "mpi"
rank  = group.rank()

# Simple rank-0 print helper
def rprint(*a, **k):
    if rank == 0:
        print(*a, **k, flush=True)

# Load model/tokenizer (simple path; for big models use the example that shards downloads)
model, tok = load(args.model)

# Enable pipeline partitioning across ranks if the model supports it
if hasattr(model, "model") and hasattr(model.model, "pipeline"):
    model.model.pipeline(group)
else:
    rprint("This model does not support pipeline(); falling back to single-rank behavior.")

# Make a chat prompt
messages = [{"role": "user", "content": args.prompt}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True)

# Stream tokens; only rank 0 prints
for resp in stream_generate(model, tok, prompt, max_tokens=args.max_tokens):
    rprint(getattr(resp, "text", str(resp)), end="")
rprint()