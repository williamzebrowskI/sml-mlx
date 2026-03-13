#!/usr/bin/env python3
"""nanochat-style SFT task sources for the GPT-2 MLX path."""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset


IDENTITY_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000
GSM_TOOL_RE = re.compile(r"(<<[^>]+>>)")
MMLU_LETTERS = ("A", "B", "C", "D")


def ensure_download(url: str, path: str) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return str(target)


def render_mc(question: str, letters: tuple[str, ...], choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with the correct letter.")
    return "\n".join(lines)


def _identity_conversation_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(row, list):
        return {"messages": row}
    if isinstance(row.get("messages"), list):
        return {"messages": row["messages"]}
    if isinstance(row.get("conversation"), list):
        return {"messages": row["conversation"]}
    if isinstance(row.get("conversations"), list):
        return {"messages": row["conversations"]}
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    answer = row.get("response") or row.get("completion") or row.get("output") or row.get("answer")
    if isinstance(prompt, str) and isinstance(answer, str):
        return {
            "messages": [
                {"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": answer.strip()},
            ]
        }
    return None


def _gsm8k_conversation(row: Dict[str, Any]) -> Dict[str, Any]:
    parts: List[Dict[str, str]] = []
    for chunk in GSM_TOOL_RE.split(str(row["answer"])):
        if not chunk:
            continue
        if chunk.startswith("<<") and chunk.endswith(">>"):
            inner = chunk[2:-2]
            if "=" in inner:
                expr, result = inner.rsplit("=", 1)
            else:
                expr, result = inner, ""
            parts.append({"type": "python", "text": expr})
            parts.append({"type": "python_output", "text": result})
        else:
            parts.append({"type": "text", "text": chunk})
    return {
        "messages": [
            {"role": "user", "content": str(row["question"]).strip()},
            {"role": "assistant", "content": parts},
        ]
    }


def _mmlu_conversation(row: Dict[str, Any]) -> Dict[str, Any]:
    if "train" in row and isinstance(row["train"], dict):
        row = row["train"]
    question = str(row["question"]).strip()
    choices = [str(choice).strip() for choice in row["choices"]]
    answer = int(row["answer"])
    return {
        "messages": [
            {"role": "user", "content": render_mc(question, MMLU_LETTERS, choices)},
            {"role": "assistant", "content": MMLU_LETTERS[answer]},
        ]
    }


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    weight: int
    split: str = "train"
    dataset_name: str = ""
    config_name: str = ""
    subset: str = ""
    path: str = ""
    url: str = ""
    size: int = 0
    shuffle_buffer: int = 0
    repeat: int = 1


class BaseCursor:
    def next_conversation(self) -> Dict[str, Any]:
        raise NotImplementedError


class HFMapCursor(BaseCursor):
    def __init__(
        self,
        spec: SourceSpec,
        *,
        rank: int,
        world: int,
        seed: int,
        trust_remote_code: bool,
    ):
        self.spec = spec
        self.rank = rank
        self.world = world
        self.seed = seed
        self.trust_remote_code = trust_remote_code
        self.epoch = 0
        self.index = 0
        self.base_ds = self._build_dataset()
        self.ds = None
        self._reset()

    def _dataset_args(self) -> tuple[str, Optional[str], str]:
        if self.spec.kind == "smoltalk":
            return "HuggingFaceTB/smol-smoltalk", None, self.spec.split
        if self.spec.kind == "mmlu":
            subset = self.spec.subset or "auxiliary_train"
            return "cais/mmlu", subset, self.spec.split
        if self.spec.kind == "gsm8k":
            subset = self.spec.subset or "main"
            return "openai/gsm8k", subset, self.spec.split
        raise ValueError(f"Unsupported HF source kind: {self.spec.kind}")

    def _normalize(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.spec.kind == "smoltalk":
            return {"messages": row["messages"]}
        if self.spec.kind == "mmlu":
            return _mmlu_conversation(row)
        if self.spec.kind == "gsm8k":
            return _gsm8k_conversation(row)
        raise ValueError(f"Unsupported HF source kind: {self.spec.kind}")

    def _build_dataset(self):
        name, config, split = self._dataset_args()
        ds = load_dataset(
            name,
            config,
            split=split,
            trust_remote_code=self.trust_remote_code,
        )
        if self.world > 1:
            ds = ds.shard(num_shards=self.world, index=self.rank, contiguous=False)
        return ds

    def _reset(self) -> None:
        self.ds = self.base_ds
        if self.spec.split == "train":
            self.ds = self.ds.shuffle(seed=self.seed + self.epoch)
        self.index = 0

    def next_conversation(self) -> Dict[str, Any]:
        if self.index >= len(self.ds):
            self.epoch += 1
            self._reset()
        row = self.ds[self.index]
        self.index += 1
        return self._normalize(row)


class JSONLinesCursor(BaseCursor):
    def __init__(self, spec: SourceSpec, *, rank: int, world: int, seed: int):
        self.spec = spec
        self.rank = rank
        self.world = world
        self.seed = seed
        self.epoch = 0
        self.items = self._load_items()
        self.order: List[int] = []
        self.pos = 0
        self._reset()

    def _load_items(self) -> List[Dict[str, Any]]:
        path = ensure_download(self.spec.url or IDENTITY_URL, self.spec.path)
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                conv = _identity_conversation_from_row(row)
                if conv is not None:
                    items.append(conv)
        if not items:
            raise RuntimeError(f"No usable conversations found in {path}")
        return items

    def _reset(self) -> None:
        rng = random.Random(self.seed + self.epoch)
        self.order = list(range(len(self.items)))
        rng.shuffle(self.order)
        self.pos = 0

    def next_conversation(self) -> Dict[str, Any]:
        while True:
            if self.pos >= len(self.order):
                self.epoch += 1
                self._reset()
            idx = self.order[self.pos]
            self.pos += 1
            if self.world > 1 and (self.pos - 1) % self.world != self.rank:
                continue
            return self.items[idx]


class SimpleSpellingCursor(BaseCursor):
    def __init__(self, spec: SourceSpec, *, split: str):
        self.spec = spec
        self.split = split
        self.words = _load_word_list()
        self.index = 0

    def next_conversation(self) -> Dict[str, Any]:
        seed = self.index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + self.index
        self.index += 1
        rng = random.Random(seed)
        word = rng.choice(self.words)
        letters = ",".join(list(word))
        return {
            "messages": [
                {"role": "user", "content": f"Spell the word: {word}"},
                {"role": "assistant", "content": f"{word}:{letters}"},
            ]
        }


class SpellingBeeCursor(BaseCursor):
    templates = (
        "How many {letter} are in the word {word}",
        "How many times does {letter} appear in {word}",
        "Count the number of {letter} in {word}",
        "In the word {word}, how many {letter} are there",
        "How many {letter}s are in {word}",
        "What is the frequency of {letter} in {word}",
    )

    def __init__(self, spec: SourceSpec, *, split: str):
        self.spec = spec
        self.split = split
        self.words = _load_word_list()
        self.index = 0

    def next_conversation(self) -> Dict[str, Any]:
        seed = self.index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + self.index
        self.index += 1
        rng = random.Random(seed)
        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice("abcdefghijklmnopqrstuvwxyz")
        count = word.count(letter)
        template = rng.choice(self.templates)
        if rng.random() < 0.3:
            template = template.lower()
        user = template.format(letter=letter, word=word)
        if rng.random() < 0.5:
            user += "?"
        parts = [
            {
                "type": "text",
                "text": (
                    f"We need to count the number of '{letter}' characters in '{word}'.\n\n"
                    f"Spelling it out: {word}:{','.join(list(word))}\n\n"
                    f"My final answer is:\n\n#### {count}"
                ),
            }
        ]
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": parts},
            ]
        }


_WORD_LIST_CACHE: Optional[List[str]] = None


def _load_word_list() -> List[str]:
    global _WORD_LIST_CACHE
    if _WORD_LIST_CACHE is not None:
        return _WORD_LIST_CACHE
    path = ensure_download(
        WORD_LIST_URL,
        "/Users/williamzebrowski/sml-mlx/distributed_nanochat_gpt2_sft/data/words_alpha.txt",
    )
    with open(path, "r", encoding="utf-8") as f:
        _WORD_LIST_CACHE = [line.strip() for line in f if line.strip()]
    return _WORD_LIST_CACHE


def parse_source_specs(value: Any, *, default_shuffle_buffer: int) -> List[SourceSpec]:
    if isinstance(value, str):
        raw = json.loads(value) if value.strip() else []
    else:
        raw = value or []
    specs: List[SourceSpec] = []
    for item in raw:
        data = dict(item)
        data.setdefault("shuffle_buffer", default_shuffle_buffer)
        data.setdefault("repeat", 1)
        specs.append(SourceSpec(**data))
    return specs


def build_cursor(
    spec: SourceSpec,
    *,
    rank: int,
    world: int,
    seed: int,
    trust_remote_code: bool,
) -> BaseCursor:
    if spec.kind in {"smoltalk", "mmlu", "gsm8k"}:
        return HFMapCursor(
            spec,
            rank=rank,
            world=world,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )
    if spec.kind == "identity_jsonl":
        return JSONLinesCursor(spec, rank=rank, world=world, seed=seed)
    if spec.kind == "simple_spelling":
        return SimpleSpellingCursor(spec, split=spec.split)
    if spec.kind == "spelling_bee":
        return SpellingBeeCursor(spec, split=spec.split)
    raise ValueError(f"Unsupported source kind: {spec.kind}")
