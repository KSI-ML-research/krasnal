"""Supervised CE targets: which next-token positions never contribute to loss."""

from typing import Final

import torch

from krasnal.tokens import (
    CONDITIONING_METADATA_TARGET_MASK_IDS,
    IS_CHECK_ID,
    PIECE_TYPE_MOVED_ID,
    WHATS_ON_PROMPT_TOKEN_IDS,
)

LOSS_IGNORE_INDEX = -100

IGNORE_IN_SUPERVISED_TARGET: Final[frozenset[int]] = frozenset(
    {
        IS_CHECK_ID,
        PIECE_TYPE_MOVED_ID,
        *WHATS_ON_PROMPT_TOKEN_IDS,
        *CONDITIONING_METADATA_TARGET_MASK_IDS,
    }
)


def apply_supervised_loss_mask(
    y: torch.Tensor,
    ignore_index: int = LOSS_IGNORE_INDEX,
) -> torch.Tensor:
    """Return a copy of ``y`` with supervised-loss-ignore applied to prompt-like token positions."""
    out = y.clone()
    ids = torch.tensor(
        sorted(IGNORE_IN_SUPERVISED_TARGET),
        dtype=out.dtype,
        device=out.device,
    )
    mask = torch.isin(out, ids)
    out[mask] = ignore_index
    return out
