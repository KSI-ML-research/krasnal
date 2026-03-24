from typing import Any

import torch

from .context import EvalContext


class Top1LegalMetric:
    """Compute top-1 legal move accuracy.

    Measures whether the model's highest-probability token is a legal move.
    """

    name = "top1_legal"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        top1_token = torch.argmax(ctx.probs).item()
        is_legal = 1.0 if top1_token in ctx.legal_ids else 0.0
        return {"top1_legal": is_legal}
