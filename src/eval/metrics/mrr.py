from typing import Any

import torch

from .context import EvalContext


class MRRMetric:
    """Compute Mean Reciprocal Rank (MRR).

    Measures the average reciprocal rank of the actual move in the model's
    probability distribution. A rank of 1 means the move is top-predicted.
    """

    name = "mrr"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        actual_token = ctx.actual_token
        if actual_token is None:
            return {"mrr": 0.0}

        probs = ctx.probs
        if probs.numel() == 0:
            return {"mrr": 0.0}

        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == actual_token).nonzero(as_tuple=True)[0]

        if rank.numel() == 0:
            return {"mrr": 0.0}

        reciprocal_rank = 1.0 / (rank.item() + 1)
        return {"mrr": reciprocal_rank}
