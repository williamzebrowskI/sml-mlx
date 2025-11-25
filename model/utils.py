# ---- HF streaming helpers ----
from datasets import load_dataset

def hf_text_iterator(
    name: str,
    split: str = "train",
    config: str | None = None,
    field: str = "text",
    world_size: int = 1,
    rank: int = 0,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
    trust_remote_code: bool = False,
):
    """
    Streams lines from a Hugging Face dataset WITHOUT downloading the whole set.
    Each yielded item is {'text': "..."} for Phase A pretraining.

    Example:
      name="c4", config="en", split="train", field="text"
      name="wikimedia/wikipedia", config="20231101.en", split="train", field="text"
    """
    ds = load_dataset(
        name,
        config,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )
    # Ensure each machine sees a disjoint portion of the stream
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
    # Local reservoir shuffle (constant-memory)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for ex in ds:
        txt = ex.get(field)
        if not txt:
            continue
        yield {"text": txt}

def hf_qa_iterator(
    name: str,
    split: str,
    config: str | None,
    q_field: str,
    a_field: str | list[str],
    world_size: int = 1,
    rank: int = 0,
    shuffle_buffer: int = 2048,
    seed: int = 123,
    trust_remote_code: bool = False,
):
    """
    Streams QA examples for Phase B (tool-augmented supervision).
    Yields {'trace': '<QUESTION>...</QUESTION> ... <ANSWER>...'} minimal traces.
    You can later expand this to inject <SEARCH>/<DOC> steps.
    """
    ds = load_dataset(
        name,
        config,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    def normalize_ans(ans):
        if isinstance(a_field, list):
            for k in a_field:
                if k in ans:
                    return ans[k]
        return ans

    for ex in ds:
        q = ex.get(q_field)
        ans_raw = ex.get(a_field if isinstance(a_field, str) else (a_field[0] if a_field else "answers"))
        if not q or ans_raw is None:
            continue
        # Many QA sets store answers as dict/list; pick the first string
        if isinstance(ans_raw, dict):
            vals = list(ans_raw.values())
            ans = vals[0][0] if vals and isinstance(vals[0], list) and vals[0] else str(vals[0])
        elif isinstance(ans_raw, list):
            ans = ans_raw[0] if ans_raw else ""
        else:
            ans = str(ans_raw)

        trace = f"<QUESTION>{q}</QUESTION>\n<PLAN>brief</PLAN>\n<ANSWER>{ans}</ANSWER>"
        yield {"trace": trace}
