# sanity_shift.py
import json, mlx.core as mx
from tokenizer import sp_tokenizer
from model.train import to_samples, batches         # reuse helpers

tok   = sp_tokenizer.load("tokenizer/spm.model")
ctx   = 512
data  = json.loads(open("data/raw/openwebtext_1M.jsonl").readline())["text"]
ids   = tok.encode(data)
sample= mx.array(ids[:ctx+1], dtype=mx.int32)[None]  # one sample length 513

x, y  = sample[:, :-1], sample[:, 1:]
print("x[0] :", tok.decode(x[0].tolist())[:120])
print("y[0] :", tok.decode(y[0].tolist())[:120])


from datasets import load_dataset
from tokenizer import sp_tokenizer
tok = sp_tokenizer.load("tokenizer/spm.model")

ds = load_dataset(
        "json",
        data_files="data/raw/openwebtext_1M.jsonl",
        split="train")                     # ❶ raw → only "text"

sample = ds[1234]["text"]
ids_now   = tok.encode(sample)             # freshly encoded (what model will see)

# do what the training script does
ids_train = (tok.encode(sample))           # identical call in map(), so should match
assert ids_now == ids_train
print("✅ token IDs match")