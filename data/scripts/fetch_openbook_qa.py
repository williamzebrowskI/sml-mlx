import random, json, pathlib
from datasets import load_dataset

# --- load ---
train = load_dataset("allenai/openbookqa", split="train")
valid = load_dataset("allenai/openbookqa", split="validation")

# --- helper: convert MC row -> (question, answer) pair ---
def mc_to_pair(row):
    q = row["question_stem"].strip()
    # answerKey is a letter; map to index in choices["label"]
    idx  = row["choices"]["label"].index(row["answerKey"])
    ans  = row["choices"]["text"][idx].strip()
    # turn the MC stem into a declarative fill-in-the-blank
    qfmt = q if q.endswith(("?", "is", "are", "was", "were")) else q + " "
    return {"question": qfmt, "answer": ans}

# --- sample 4 000 train rows (shuffle for variety) ---
random.seed(42)
idxs = random.sample(range(len(train)), 4000)   # 4 000 random indices
pairs_train = [mc_to_pair(train[i]) for i in idxs]
pairs_val   = [mc_to_pair(r) for r in valid]

# --- write JSONL ---
pathlib.Path("data").mkdir(exist_ok=True)
with open("data/qa_train.jsonl", "w") as f:
    for ex in pairs_train:
        f.write(json.dumps(ex) + "\n")

with open("data/qa_valid.jsonl", "w") as f:
    for ex in pairs_val:
        f.write(json.dumps(ex) + "\n")

print("wrote", len(pairs_train), "train &", len(pairs_val), "valid rows")