from __future__ import annotations

import random
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence

from dataset import collate_fn
from tokenizer import PAD_ID


class RLPhase1DataSource:
    """Sample RL prompts and supervised batches from the pretrain parquet dataset."""

    def __init__(self, parquet_path: Path, *, max_prompt_tokens: int) -> None:
        self.dataset = HFDataset.from_parquet(str(parquet_path))
        self.dataset.set_format(type="torch", columns=["token_ids"])
        self.max_prompt_tokens = max_prompt_tokens

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_sequences(self, batch_size: int) -> list[torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self.dataset) == 0:
            raise ValueError("RL dataset is empty")
        indices = [random.randrange(len(self.dataset)) for _ in range(batch_size)]
        return [self.dataset[idx]["token_ids"].to(torch.long) for idx in indices]

    def sample_prompt_batch(
        self,
        batch_size: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = self._sample_sequences(batch_size)
        prompts: list[torch.Tensor] = []
        prompt_lengths: list[int] = []

        for seq in sequences:
            usable_len = max(1, min(int(seq.numel()) - 1, self.max_prompt_tokens))
            prefix_len = random.randint(1, usable_len)
            prompt = seq[:prefix_len]
            prompts.append(prompt)
            prompt_lengths.append(int(prompt.numel()))

        padded = pad_sequence(prompts, batch_first=True, padding_value=PAD_ID)
        return padded.to(device), torch.tensor(prompt_lengths, dtype=torch.long, device=device)

    def sample_supervised_batch(
        self, batch_size: int, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = self._sample_sequences(batch_size)
        x, y = collate_fn(sequences)
        return x.to(device), y.to(device)
