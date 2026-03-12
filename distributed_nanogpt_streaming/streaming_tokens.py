#!/usr/bin/env python3
"""Streaming token-window pipeline for nanoGPT-style MLX pretraining."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

import mlx.core as mx


@dataclass
class HFTokenSourceConfig:
    name: str
    split: str = "train"
    config: Optional[str] = None
    token_field: str = "tokens"
    weight: int = 1
    shuffle_buffer: int = 0
    trust_remote_code: bool = False
    append_eos: bool = False
    eos_token_id: int = 50256

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HFTokenSourceConfig":
        return cls(
            name=d["name"],
            split=d.get("split", "train"),
            config=d.get("config"),
            token_field=d.get("token_field", "tokens"),
            weight=int(d.get("weight", 1)),
            shuffle_buffer=int(d.get("shuffle_buffer", 0)),
            trust_remote_code=bool(d.get("trust_remote_code", False)),
            append_eos=bool(d.get("append_eos", False)),
            eos_token_id=int(d.get("eos_token_id", 50256)),
        )


def parse_token_source_configs(value: Any) -> list[HFTokenSourceConfig]:
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
        raise ValueError("Token sources must be a list/dict/json-string/json-file.")

    out = [HFTokenSourceConfig.from_dict(x) for x in payload]
    if not out:
        raise ValueError("Token source list is empty.")
    for src in out:
        if src.weight <= 0:
            raise ValueError(f"Source {src.name} has invalid weight={src.weight}")
    return out


class _TokenSourceCursor:
    def __init__(self, cfg: HFTokenSourceConfig, world_size: int, rank: int, base_seed: int):
        self.cfg = cfg
        self.world_size = world_size
        self.rank = rank
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.example_index = 0
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

    def _next_tokens(self) -> list[int]:
        while True:
            try:
                ex = next(self.it)
            except StopIteration:
                self.epoch += 1
                self._reset_dataset()
                continue

            idx = self.example_index
            self.example_index += 1

            tokens = ex.get(self.cfg.token_field)
            if not isinstance(tokens, list) or len(tokens) < 2:
                continue

            out = [int(tok) for tok in tokens]
            if self.cfg.append_eos and self.cfg.eos_token_id >= 0:
                out.append(int(self.cfg.eos_token_id))
            if len(out) >= 2:
                return out

    def state_dict(self) -> dict[str, Any]:
        ds_state = None
        if hasattr(self.ds, "state_dict"):
            ds_state = self.ds.state_dict()
        return {
            "cfg": asdict(self.cfg),
            "epoch": self.epoch,
            "example_index": self.example_index,
            "buffer": self.buffer,
            "dataset_state": ds_state,
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.epoch = int(state.get("epoch", 0))
        self.example_index = int(state.get("example_index", 0))
        self.buffer = [int(t) for t in state.get("buffer", [])]
        self._reset_dataset()
        ds_state = state.get("dataset_state")
        if ds_state is not None and hasattr(self.ds, "load_state_dict"):
            self.ds.load_state_dict(ds_state)
            self.it = iter(self.ds)


class HFTokenStreamingBatcher:
    def __init__(
        self,
        sources: list[HFTokenSourceConfig],
        world_size: int,
        rank: int,
        seed: int,
    ):
        if not sources:
            raise ValueError("No token streaming sources configured.")

        self.sources = [
            _TokenSourceCursor(cfg=s, world_size=world_size, rank=rank, base_seed=seed + i * 1_000_003)
            for i, s in enumerate(sources)
        ]
        self.schedule = []
        for i, s in enumerate(sources):
            self.schedule.extend([i] * int(s.weight))
        if not self.schedule:
            raise ValueError("Source weights created an empty schedule.")

        self.rank = rank
        self.schedule_pos = 0
        self.total_tokens_emitted = 0
        self.total_batches_emitted = 0
        self.source_initial_skips: list[int] = []

    def _next_source_index(self) -> int:
        idx = self.schedule[self.schedule_pos % len(self.schedule)]
        self.schedule_pos += 1
        return idx

    def _ensure_tokens(self, src_idx: int, needed: int):
        src = self.sources[src_idx]
        while len(src.buffer) < needed:
            src.buffer.extend(src._next_tokens())

    def sample_batch(self, batch_size: int, seq_len: int):
        need = seq_len + 1
        x = np.empty((batch_size, seq_len), dtype=np.int32)
        y = np.empty((batch_size, seq_len), dtype=np.int32)

        for i in range(batch_size):
            src_idx = self._next_source_index()
            src = self.sources[src_idx]
            while len(self.source_initial_skips) <= src_idx:
                self.source_initial_skips.append(-1)
            if self.source_initial_skips[src_idx] < 0:
                self.source_initial_skips[src_idx] = int(self.rank * seq_len)
            initial_skip = self.source_initial_skips[src_idx]
            if initial_skip > 0:
                self._ensure_tokens(src_idx, need + initial_skip)
                del src.buffer[:initial_skip]
                self.source_initial_skips[src_idx] = 0
            self._ensure_tokens(src_idx, need)
            chunk = src.buffer[:need]
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
            "source_initial_skips": self.source_initial_skips,
            "sources": [s.state_dict() for s in self.sources],
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.schedule_pos = int(state.get("schedule_pos", 0))
        self.total_tokens_emitted = int(state.get("total_tokens_emitted", 0))
        self.total_batches_emitted = int(state.get("total_batches_emitted", 0))
        self.source_initial_skips = [int(x) for x in state.get("source_initial_skips", [])]
        src_states = state.get("sources", [])
        if len(src_states) != len(self.sources):
            raise ValueError(
                f"Source state length mismatch: file={len(src_states)} runtime={len(self.sources)}"
            )
        for src, src_state in zip(self.sources, src_states):
            src.load_state_dict(src_state)


class StreamingTokenDatasetAdapter:
    def __init__(self, batcher: HFTokenStreamingBatcher):
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
