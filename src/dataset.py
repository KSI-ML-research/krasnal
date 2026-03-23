import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import TrainConfig
from tokenizer import PAD_ID

HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", "/tmp/krasnal-hf-datasets")
PADDING_BUCKET_SIZES = TrainConfig.padding_bucket_sizes


def _get_bucket_size(seq_len: int) -> int:
    """Return the smallest bucket size that is >= seq_len."""
    for b in PADDING_BUCKET_SIZES:
        if seq_len <= b:
            return b
    return seq_len


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
        self.dataset = HFDataset.from_parquet(paths, cache_dir=HF_DATASETS_CACHE)
        self.dataset.set_format(type="torch", columns=["token_ids"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]["token_ids"].to(torch.long)


class MixedChessDataset(Dataset[torch.Tensor]):
    """Sample a fixed-ratio mixture of CoT and normal-play rows."""

    def __init__(self, cot_path: Path, normal_path: Path, *, cot_ratio: float, seed: int = 42):
        if not 0.0 <= cot_ratio <= 1.0:
            raise ValueError("cot_ratio must be between 0 and 1")

        self.cot_dataset = HFDataset.from_parquet(str(cot_path), cache_dir=HF_DATASETS_CACHE)
        self.normal_dataset = HFDataset.from_parquet(str(normal_path), cache_dir=HF_DATASETS_CACHE)
        self.cot_dataset.set_format(type="torch", columns=["token_ids"])
        self.normal_dataset.set_format(type="torch", columns=["token_ids"])

        cot_len = len(self.cot_dataset)
        normal_len = len(self.normal_dataset)
        if cot_len == 0 and normal_len == 0:
            raise ValueError("Both datasets are empty")
        if cot_len == 0:
            cot_count = 0
            normal_count = normal_len
        elif normal_len == 0 or cot_ratio == 1.0:
            cot_count = cot_len
            normal_count = 0
        elif cot_ratio == 0.0:
            cot_count = 0
            normal_count = normal_len
        else:
            normal_target = round(cot_len * (1.0 - cot_ratio) / cot_ratio)
            if normal_target <= normal_len:
                cot_count = cot_len
                normal_count = normal_target
            else:
                normal_count = normal_len
                cot_count = max(1, min(cot_len, round(normal_len * cot_ratio / (1.0 - cot_ratio))))

        rng = random.Random(seed)
        cot_indices = list(range(cot_len))
        normal_indices = list(range(normal_len))
        rng.shuffle(cot_indices)
        rng.shuffle(normal_indices)
        self.sources = [("cot", idx) for idx in cot_indices[:cot_count]]
        self.sources.extend(("normal", idx) for idx in normal_indices[:normal_count])
        self.cot_count = cot_count
        self.normal_count = normal_count

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> torch.Tensor:
        source, source_idx = self.sources[idx]
        dataset = self.cot_dataset if source == "cot" else self.normal_dataset
        return dataset[source_idx]["token_ids"].to(torch.long)


def collate_fn(batch):
    """
    Pads sequences and applies Bucket Padding (Sequence Bucketing) to stabilize torch.compile().

    Dynamo/torch.compile recompiles the graph on every unique input shape. By padding
    batch sequences to predefined bucket sizes (multiples of 64), we minimize recompilations
    and optimize hardware utilization for Flash Attention and Tensor Cores.

    The sequence is padded to (bucket_size + 1) to ensure that after splitting into
    inputs [:, :-1] and targets [:, 1:], both tensors match the optimal bucket size exactly.
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)

    # The actual length for the model input/output is padded.size(1) - 1
    seq_len = padded.size(1) - 1
    if seq_len > 0:
        # Find the optimal bucket size
        target_len = _get_bucket_size(seq_len)
        target_total_len = target_len + 1

        if padded.size(1) < target_total_len:
            pad_size = target_total_len - padded.size(1)
            padded = F.pad(padded, (0, pad_size), value=PAD_ID)

    return padded[:, :-1], padded[:, 1:]
