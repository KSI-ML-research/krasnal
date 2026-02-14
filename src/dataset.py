import torch
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ChessDataset(Dataset):
    def __init__(self, parquet_path: Path):
        df = pl.read_parquet(parquet_path)
        self.games = df["token_ids"].to_list()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx: int):
        return torch.tensor(self.games[idx], dtype=torch.long)


def collate_fn(batch):
    padded = pad_sequence(batch, batch_first=True, padding_value=2)
    return padded[:, :-1], padded[:, 1:]

