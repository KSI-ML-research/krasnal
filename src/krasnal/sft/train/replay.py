from __future__ import annotations

import random
from pathlib import Path

import torch

from krasnal.dataset import ChessDataset


def resolve_shard_paths(shards_dir: Path) -> list[Path]:
    """Return all parquet shards inside a shard directory."""
    if not shards_dir.exists():
        raise FileNotFoundError(f"CoT shards directory not found at {shards_dir}")
    paths = sorted(path for path in shards_dir.glob("*.parquet") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No parquet shards found in {shards_dir}")
    return paths


def split_shard_paths(
    shard_paths: list[Path],
    *,
    eval_fraction: float,
) -> tuple[list[Path], list[Path]]:
    """Split shard paths into deterministic train/eval subsets."""
    if not shard_paths:
        raise ValueError("shard_paths cannot be empty")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")

    if len(shard_paths) == 1:
        return shard_paths, shard_paths

    eval_count = max(1, round(len(shard_paths) * eval_fraction))
    eval_count = min(eval_count, len(shard_paths) - 1)
    train_paths = shard_paths[:-eval_count]
    eval_paths = shard_paths[-eval_count:]
    return train_paths, eval_paths


class RandomTokenSource:
    """Sample token sequences uniformly from parquet-backed datasets."""

    def __init__(self, paths: Path | list[Path], *, seed: int = 42) -> None:
        self.dataset = ChessDataset(paths)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def sample_sequences(self, batch_size: int) -> list[torch.Tensor]:
        """Return uniformly sampled token sequences."""
        if batch_size <= 0:
            return []
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        indices = [self.rng.randrange(len(self.dataset)) for _ in range(batch_size)]
        return [self.dataset[idx] for idx in indices]


class CotReplaySource(RandomTokenSource):
    """Read previously saved CoT rows from a shard directory."""

    def __init__(self, shards_dir: Path, *, seed: int = 42) -> None:
        self.shard_paths = resolve_shard_paths(shards_dir)
        super().__init__(self.shard_paths, seed=seed)
        self.order = list(range(len(self.dataset)))
        self.rng.shuffle(self.order)
        self.index = 0

    @property
    def total_rows(self) -> int:
        """Return the total number of replay rows."""
        return len(self.dataset)

    def remaining_rows(self) -> int:
        """Return the number of replay rows left in the current pass."""
        return len(self.dataset) - self.index

    def sample_sequences(self, batch_size: int) -> list[torch.Tensor]:
        """Return the next replay rows without replacement."""
        if batch_size <= 0 or self.remaining_rows() == 0:
            return []
        stop = min(self.index + batch_size, len(self.dataset))
        indices = self.order[self.index : stop]
        self.index = stop
        return [self.dataset[idx] for idx in indices]
