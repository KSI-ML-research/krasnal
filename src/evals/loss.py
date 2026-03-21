from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..dataset import ChessDataset, collate_fn
from ..tokenizer import PAD_ID


@torch.inference_mode()
def evaluate_unseen_loss(
    model: torch.nn.Module,
    dataset: ChessDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, int, int]:
    """Return (mean_loss, perplexity, sequences, predicted_tokens) on unseen data."""
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model.eval()
    total_loss_weighted = 0.0
    total_tokens = 0
    total_sequences = 0

    for x, y in tqdm(loader, desc="Evaluating unseen loss"):
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y, ignore_index=PAD_ID)
        if loss is None:
            continue

        valid_tokens = int((y != PAD_ID).sum().item())
        if valid_tokens == 0:
            continue

        total_loss_weighted += float(loss.item()) * valid_tokens
        total_tokens += valid_tokens
        total_sequences += x.size(0)

    if total_tokens == 0:
        return float("nan"), float("nan"), total_sequences, total_tokens

    mean_loss = total_loss_weighted / total_tokens
    perplexity = float(torch.exp(torch.tensor(mean_loss)).item())
    return mean_loss, perplexity, total_sequences, total_tokens
