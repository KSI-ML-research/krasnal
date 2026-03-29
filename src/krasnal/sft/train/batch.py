from __future__ import annotations

import torch
import torch.nn.functional as F

from krasnal.tokens import PAD_ID


def compute_batch_sizes(batch_size: int, cot_ratio: float) -> tuple[int, int]:
    """
    Compute the number of CoT and normal-play samples in a batch.

    Args:
        batch_size: The total number of samples in the batch.
        cot_ratio: The ratio of CoT samples in the batch.

    Returns:
        A tuple of (CoT samples, normal-play samples).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not 0.0 <= cot_ratio <= 1.0:
        raise ValueError("cot_ratio must be between 0 and 1")
    if cot_ratio == 0.0:
        return 0, batch_size
    if cot_ratio == 1.0:
        return batch_size, 0

    cot_samples = max(1, min(batch_size - 1, round(batch_size * cot_ratio)))
    normal_samples = batch_size - cot_samples
    return cot_samples, normal_samples


def compute_split_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_ids: torch.Tensor,
) -> tuple[float | None, float | None]:
    """
    Compute CoT and normal losses from one mixed forward pass.

    Args:
        logits: The model's logits.
        targets: The target token IDs.
        source_ids: The source IDs (1 for CoT, 0 for normal-play).

    Returns:
        A tuple of (CoT loss, normal-play loss).
    """
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(targets)
    valid_mask = targets != PAD_ID
    per_example_den = valid_mask.sum(dim=1)
    per_example_num = (token_losses * valid_mask).sum(dim=1)
    per_example_loss = per_example_num / per_example_den.clamp_min(1)

    cot_mask = (source_ids == 1) & (per_example_den > 0)
    normal_mask = (source_ids == 0) & (per_example_den > 0)
    cot_loss = float(per_example_loss[cot_mask].mean().item()) if cot_mask.any() else None
    normal_loss = float(per_example_loss[normal_mask].mean().item()) if normal_mask.any() else None
    return cot_loss, normal_loss
