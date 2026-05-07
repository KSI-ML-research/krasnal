import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from krasnal.supervised_target_mask import apply_supervised_loss_mask
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


class ChessDataset(Dataset[torch.Tensor]):
    """
    A PyTorch Dataset for loading chess games stored in a Parquet file.

    By using the Hugging Face `datasets` library (which is backed by Apache Arrow),
    the dataset is memory-mapped from disk. This allows for fast, zero-copy access
    to the data without loading the entire dataset and its Python lists into RAM,
    effectively solving memory usage issues during multi-process training.
    """

    def __init__(self, parquet_path: Path | list[Path]):
        paths = (
            [str(path) for path in parquet_path]
            if isinstance(parquet_path, list)
            else str(parquet_path)
        )
        cache_dir = Path(resolve_hf_datasets_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = HFDataset.from_parquet(paths, cache_dir=str(cache_dir))
        self.dataset.set_format(type="torch", columns=["token_ids"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        tokens = self.dataset[idx]["token_ids"].to(torch.long)
        if tokens.min() < 0:
            raise ValueError(f"Invalid negative tokens found at index {idx}: {tokens[tokens < 0]}")
        return tokens


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
        Pad sequences and bucket to stable lengths for torch.compile friendliness.

        The model receives x=padded[:, :-1], y=padded[:, 1:], so we pad to
        (bucket_size + 1) to keep model sequence length exactly bucket_size.
        """
        padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)

        seq_len = padded.size(1) - 1
        if seq_len > 0 and self.bucket_sizes:
            target_len = _get_bucket_size(seq_len, self.bucket_sizes)
            target_total_len = target_len + 1
            if padded.size(1) < target_total_len:
                pad_size = target_total_len - padded.size(1)
                padded = F.pad(padded, (0, pad_size), value=PAD_ID)

        x = padded[:, :-1]
        y = apply_supervised_loss_mask(padded[:, 1:])
        return x, y


def make_collate_fn(bucket_sizes: tuple[int, ...] = ()) -> Callable:
    """Build a collate function configured with explicit bucket sizes."""
    return CollateFn(bucket_sizes)
