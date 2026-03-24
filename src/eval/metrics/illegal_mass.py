from typing import Any

import torch

from .context import EvalContext


class IllegalMassMetric:
    """Compute total probability mass on illegal moves.

    Sums the model's probability distribution over all illegal move tokens.
    """

    name = "illegal_mass"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        vocab_size = ctx.probs.shape[0]
        illegal_mask = torch.ones(vocab_size, dtype=torch.bool, device=ctx.probs.device)
        illegal_mask[ctx.legal_ids] = False
        illegal_mass = float(ctx.probs[illegal_mask].sum().item())
        return {"illegal_mass": illegal_mass}
