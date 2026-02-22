#!/usr/bin/env python3
"""Hugging Face streaming data pipeline with resumable cursor state.

For exact-ish resume, set shuffle_buffer=0 for each source.
When shuffle_buffer>0, datasets warns that shuffle buffer content is not stored.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sentencepiece as spm
from datasets import load_dataset

import mlx.core as mx


@dataclass
class HFSourceConfig:
    name: str
    split: str = "train"
    config: Optional[str] = None
    text_field: str = "text"
    weight: int = 1
    shuffle_buffer: int = 0
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HFSourceConfig":
        return cls(
            name=d["name"],
            split=d.get("split", "train"),
            config=d.get("config", None),
            text_field=d.get("text_field", "text"),
            weight=int(d.get("weight", 1)),
            shuffle_buffer=int(d.get("shuffle_buffer", 0)),
            trust_remote_code=bool(d.get("trust_remote_code", False)),
        )


def parse_source_configs(value: Any) -> list[HFSourceConfig]:
    """Parse sources from list/dict/json-string/json-file path."""
    if value is None:
        return []

    payload = value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        p = Path(s)
        if p.exists():
            with open(p, "r") as f:
                payload = json.load(f)
        else:
            payload = json.loads(s)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Sources must be a list/dict/json-string/json-file.")

    out = [HFSourceConfig.from_dict(x) for x in payload]
    if not out:
        raise ValueError("Sources list is empty.")
    for src in out:
        if src.weight <= 0:
            raise ValueError(f"Source {src.name} has invalid weight={src.weight}")
    return out


class _SourceCursor:
    def __init__(self, cfg: HFSourceConfig, world_size: int, rank: int, base_seed: int):
        self.cfg = cfg
        self.world_size = world_size
        self.rank = rank
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.buffer: list[int] = []

        self.ds = None
        self.it = None
        self._reset_dataset()

    def _build_dataset(self):
        ds = load_dataset(
            self.cfg.name,
            self.cfg.config,
            split=self.cfg.split,
            streaming=True,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        if self.cfg.shuffle_buffer > 0:
            ds = ds.shuffle(
                seed=self.base_seed + self.epoch,
                buffer_size=self.cfg.shuffle_buffer,
            )
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self.epoch)
        return ds

    def _reset_dataset(self):
        self.ds = self._build_dataset()
        self.it = iter(self.ds)

    def _next_text(self) -> str:
        while True:
            try:
                ex = next(self.it)
            except StopIteration:
                self.epoch += 1
                self._reset_dataset()
                continue
            text = ex.get(self.cfg.text_field, "")
            if text:
                return text

    def state_dict(self) -> dict[str, Any]:
        ds_state = None
        if hasattr(self.ds, "state_dict"):
            ds_state = self.ds.state_dict()
        return {
            "cfg": asdict(self.cfg),
            "epoch": self.epoch,
            "buffer": self.buffer,
            "dataset_state": ds_state,
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.epoch = int(state.get("epoch", 0))
        self.buffer = [int(t) for t in state.get("buffer", [])]
        self._reset_dataset()
        ds_state = state.get("dataset_state")
        if ds_state is not None and hasattr(self.ds, "load_state_dict"):
            self.ds.load_state_dict(ds_state)
            self.it = iter(self.ds)


class HFStreamingBatcher:
    """Token batcher for one or more HF streaming sources."""

    def __init__(
        self,
        sources: list[HFSourceConfig],
        spm_model: str,
        world_size: int,
        rank: int,
        seed: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        if not sources:
            raise ValueError("No streaming sources configured.")

        self.sources = [
            _SourceCursor(cfg=s, world_size=world_size, rank=rank, base_seed=seed + i * 1_000_003)
            for i, s in enumerate(sources)
        ]
        self.schedule = []
        for i, s in enumerate(sources):
            self.schedule.extend([i] * int(s.weight))
        if not self.schedule:
            raise ValueError("Source weights created an empty schedule.")

        self.schedule_pos = 0
        self.total_tokens_emitted = 0
        self.total_batches_emitted = 0

        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def _encode_text(self, text: str) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if not ids:
            return ids
        if self.add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if self.add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def _next_source_index(self) -> int:
        idx = self.schedule[self.schedule_pos % len(self.schedule)]
        self.schedule_pos += 1
        return idx

    def _ensure_tokens(self, src_idx: int, needed: int):
        src = self.sources[src_idx]
        while len(src.buffer) < needed:
            text = src._next_text()
            tok = self._encode_text(text)
            if len(tok) < 2:
                continue
            src.buffer.extend(tok)

    def sample_batch(self, batch_size: int, seq_len: int):
        need = seq_len + 1
        x = np.empty((batch_size, seq_len), dtype=np.int32)
        y = np.empty((batch_size, seq_len), dtype=np.int32)

        for i in range(batch_size):
            src_idx = self._next_source_index()
            src = self.sources[src_idx]
            self._ensure_tokens(src_idx, need)
            chunk = src.buffer[:need]
            # Consume seq_len so adjacent chunks are contiguous in token-space.
            del src.buffer[:seq_len]
            x[i] = np.asarray(chunk[:-1], dtype=np.int32)
            y[i] = np.asarray(chunk[1:], dtype=np.int32)

        self.total_tokens_emitted += int(batch_size * seq_len)
        self.total_batches_emitted += int(batch_size)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "schedule": self.schedule,
            "schedule_pos": self.schedule_pos,
            "total_tokens_emitted": self.total_tokens_emitted,
            "total_batches_emitted": self.total_batches_emitted,
            "sources": [s.state_dict() for s in self.sources],
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.schedule_pos = int(state.get("schedule_pos", 0))
        self.total_tokens_emitted = int(state.get("total_tokens_emitted", 0))
        self.total_batches_emitted = int(state.get("total_batches_emitted", 0))
        src_states = state.get("sources", [])
        if len(src_states) != len(self.sources):
            raise ValueError(
                f"Source state length mismatch: file={len(src_states)} runtime={len(self.sources)}"
            )
        for src, src_state in zip(self.sources, src_states):
            src.load_state_dict(src_state)


class StreamingDatasetAdapter:
    """Adapter matching the trainer's sample_batch interface."""

    def __init__(self, batcher: HFStreamingBatcher):
        self.batcher = batcher

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        seed: int,
        step: int,
        rank: int,
        stream: int = 0,
    ):
        del seed, step, rank, stream
        return self.batcher.sample_batch(batch_size=batch_size, seq_len=seq_len)


def data_state_path(ckpt_path: str, rank: int) -> str:
    return f"{ckpt_path}.rank{rank}.data_state.json.gz"


def save_data_state(path: str, state: dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def load_data_state(path: str) -> Optional[dict[str, Any]]:
    if not Path(path).exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
