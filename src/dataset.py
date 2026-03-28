from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import TrainConfig
from tokenizer import PAD_ID

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

    def __init__(self, parquet_path: Path):
        self.dataset = HFDataset.from_parquet(str(parquet_path))
        self.dataset.set_format(type="torch", columns=["token_ids"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # HF datasets upcasts UInt16 from Parquet, casting explicitly for nn.Embedding just in case
        return self.dataset[idx]["token_ids"].to(torch.long)


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
