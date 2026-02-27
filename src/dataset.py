import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset
from config import PAD_ID


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
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    return padded[:, :-1], padded[:, 1:]
