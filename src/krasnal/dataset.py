import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.supervised_target_mask import LOSS_IGNORE_INDEX, apply_supervised_loss_mask
from krasnal.time_conditioning import shift_clock_rows_for_training
from krasnal.tokens import PAD_ID


def resolve_hf_datasets_cache_dir() -> str:
    """Return the datasets cache directory, preferring repo-local storage."""
    cache_dir = os.environ.get("KRASNAL_DATASETS_CACHE_DIR")
    if cache_dir:
        return cache_dir

    hf_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if hf_cache_dir:
        return hf_cache_dir

    return str(Path(".hf_cache/datasets"))


class ChessDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    A PyTorch Dataset for loading chess games stored in a Parquet file.

    By using the Hugging Face `datasets` library (which is backed by Apache Arrow),
    the dataset is memory-mapped from disk. This allows for fast, zero-copy access
    to the data without loading the entire dataset and its Python lists into RAM,
    effectively solving memory usage issues during multi-process training.
    """

    def __init__(self, parquet_path: Path | list[Path], include_elo: bool = True):
        paths = (
            [str(path) for path in parquet_path]
            if isinstance(parquet_path, list)
            else str(parquet_path)
        )
        cache_dir = Path(resolve_hf_datasets_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = HFDataset.from_parquet(paths, cache_dir=str(cache_dir))
        self.has_clock_columns = {
            "active_clock_ids",
            "opponent_clock_ids",
        }.issubset(self.dataset.column_names)
        columns = ["token_ids"]
        if self.has_clock_columns:
            columns.extend(["active_clock_ids", "opponent_clock_ids"])
        self.dataset.set_format(type="torch", columns=columns)

        self.include_elo = include_elo
        from krasnal.tokens import ELO_TOKENS

        self.elo_tensor = torch.tensor(list(ELO_TOKENS.values()), dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        tokens = row["token_ids"].to(torch.long)
        if tokens.min() < 0:
            raise ValueError(f"Invalid negative tokens found at index {idx}: {tokens[tokens < 0]}")

        if self.has_clock_columns:
            active_clocks = row["active_clock_ids"].to(torch.long)
            opponent_clocks = row["opponent_clock_ids"].to(torch.long)
        else:
            active_clocks = torch.full_like(tokens, CLOCK_IGNORE_ID)
            opponent_clocks = torch.full_like(tokens, CLOCK_IGNORE_ID)

        if not self.include_elo:
            mask = ~torch.isin(tokens, self.elo_tensor)
            tokens = tokens[mask]
            active_clocks = active_clocks[mask]
            opponent_clocks = opponent_clocks[mask]

        if not (tokens.size(0) == active_clocks.size(0) == opponent_clocks.size(0)):
            raise ValueError(f"Clock/token length mismatch at index {idx}")

        return tokens, active_clocks, opponent_clocks


class PackedChessDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Fixed-length packed training windows with per-token segment and position metadata."""

    def __init__(self, parquet_path: Path | list[Path]):
        paths = (
            [str(path) for path in parquet_path]
            if isinstance(parquet_path, list)
            else str(parquet_path)
        )
        cache_dir = Path(resolve_hf_datasets_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = HFDataset.from_parquet(paths, cache_dir=str(cache_dir))
        required = {
            "token_ids",
            "active_clock_ids",
            "opponent_clock_ids",
            "segment_ids",
            "position_ids",
        }
        missing = required - set(self.dataset.column_names)
        if missing:
            raise ValueError(f"Packed dataset missing columns: {sorted(missing)}")
        self.dataset.set_format(
            type="torch",
            columns=[
                "token_ids",
                "active_clock_ids",
                "opponent_clock_ids",
                "segment_ids",
                "position_ids",
            ],
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        tokens = row["token_ids"].to(torch.long)
        active_clocks = row["active_clock_ids"].to(torch.long)
        opponent_clocks = row["opponent_clock_ids"].to(torch.long)
        segment_ids = row["segment_ids"].to(torch.long)
        position_ids = row["position_ids"].to(torch.long)
        n = tokens.size(0)
        if not (
            active_clocks.size(0)
            == opponent_clocks.size(0)
            == segment_ids.size(0)
            == position_ids.size(0)
            == n
        ):
            raise ValueError(f"Packed row length mismatch at index {idx}")
        return tokens, active_clocks, opponent_clocks, segment_ids, position_ids


def _mask_pad_targets(y: torch.Tensor) -> torch.Tensor:
    out = y.clone()
    out[out == PAD_ID] = LOSS_IGNORE_INDEX
    return out


def _mask_cross_segment_targets(y: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    out = y.clone()
    cross = segment_ids[:, :-1] != segment_ids[:, 1:]
    out[cross] = LOSS_IGNORE_INDEX
    return out


def _get_bucket_size(seq_len: int, bucket_sizes: tuple[int, ...]) -> int:
    """Return the smallest bucket size that is >= seq_len."""
    for b in bucket_sizes:
        if seq_len <= b:
            return b
    return seq_len


class CollateFn:
    """Picklable collate callable for DataLoader worker processes."""

    def __init__(self, bucket_sizes: tuple[int, ...] = ()) -> None:
        self.bucket_sizes = tuple(int(b) for b in bucket_sizes if int(b) > 0)

    def __call__(self, batch):
        """
        Pad sequences and optionally bucket to stable lengths for eval batches.

        The model receives x=padded[:, :-1], y=padded[:, 1:], so we pad to
        (bucket_size + 1) to keep model sequence length exactly bucket_size.
        """
        if isinstance(batch[0], tuple):
            token_batch, active_clock_batch, opponent_clock_batch = zip(*batch, strict=True)
        else:
            token_batch = batch
            active_clock_batch = [
                torch.full_like(tokens, CLOCK_IGNORE_ID) for tokens in token_batch
            ]
            opponent_clock_batch = [
                torch.full_like(tokens, CLOCK_IGNORE_ID) for tokens in token_batch
            ]

        padded = pad_sequence(token_batch, batch_first=True, padding_value=PAD_ID)
        active_padded = pad_sequence(
            active_clock_batch,
            batch_first=True,
            padding_value=CLOCK_IGNORE_ID,
        )
        opponent_padded = pad_sequence(
            opponent_clock_batch,
            batch_first=True,
            padding_value=CLOCK_IGNORE_ID,
        )

        seq_len = padded.size(1) - 1
        if seq_len > 0 and self.bucket_sizes:
            target_len = _get_bucket_size(seq_len, self.bucket_sizes)
            target_total_len = target_len + 1
            if padded.size(1) < target_total_len:
                pad_size = target_total_len - padded.size(1)
                padded = F.pad(padded, (0, pad_size), value=PAD_ID)
                active_padded = F.pad(active_padded, (0, pad_size), value=CLOCK_IGNORE_ID)
                opponent_padded = F.pad(opponent_padded, (0, pad_size), value=CLOCK_IGNORE_ID)

        x = padded[:, :-1]
        active_x, opponent_x = shift_clock_rows_for_training(active_padded, opponent_padded)
        y = apply_supervised_loss_mask(padded[:, 1:])
        y = _mask_pad_targets(y)
        return x, active_x, opponent_x, y


class PackedCollateFn:
    """Collate fixed-length packed windows for training."""

    def __call__(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        tokens, active, opponent, segments, positions = zip(*batch, strict=True)
        padded = torch.stack(tokens)
        active_padded = torch.stack(active)
        opponent_padded = torch.stack(opponent)
        segment_padded = torch.stack(segments)
        position_padded = torch.stack(positions)

        x = padded[:, :-1]
        segment_x = segment_padded[:, :-1]
        position_x = position_padded[:, :-1]
        active_x, opponent_x = shift_clock_rows_for_training(active_padded, opponent_padded)
        y = apply_supervised_loss_mask(padded[:, 1:])
        y = _mask_pad_targets(y)
        y = _mask_cross_segment_targets(y, segment_padded)
        return x, active_x, opponent_x, y, segment_x, position_x


def make_collate_fn(bucket_sizes: tuple[int, ...] = ()) -> Callable:
    """Build a collate function configured with optional bucket sizes (eval)."""
    return CollateFn(bucket_sizes)


def make_packed_collate_fn() -> Callable:
    """Build the collate function for packed training windows."""
    return PackedCollateFn()
