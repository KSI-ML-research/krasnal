from typing import Any

import torch

from .context import EvalContext


class Top1LegalWhenGivesCheckMetric:
    """Compute legal move rate when move gives check.

    Measures whether the model's highest-probability token is a legal move,
    but only for positions where the move gives check to the opponent.
    """

    name = "top1_legal_when_gives_check"

    def __init__(self):
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.gives_check:
            top1_token = torch.argmax(ctx.probs).item()
            is_legal = 1.0 if top1_token in ctx.legal_ids else 0.0
            self.buffer.append(is_legal)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {"top1_legal_when_gives_check": 0.0}
        return {"top1_legal_when_gives_check": sum(self.buffer) / len(self.buffer)}


class AccWhenGivesCheckMetric:
    """Compute accuracy when move gives check.

    Measures whether the model's highest-probability token matches
    the actual (ground truth) move, but only for positions where
    the move gives check to the opponent.
    """

    name = "acc_when_gives_check"

    def __init__(self):
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.gives_check and ctx.actual_token is not None:
            top1_token = torch.argmax(ctx.probs).item()
            is_correct = 1.0 if top1_token == ctx.actual_token else 0.0
            self.buffer.append(is_correct)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {"acc_when_gives_check": 0.0}
        return {"acc_when_gives_check": sum(self.buffer) / len(self.buffer)}


class IllegalMassWhenGivesCheckMetric:
    """Compute illegal mass when move gives check.

    Sums the model's probability distribution over all illegal move tokens,
    but only for positions where the move gives check to the opponent.
    """

    name = "illegal_mass_when_gives_check"

    def __init__(self):
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.gives_check:
            vocab_size = ctx.probs.shape[0]
            illegal_mask = torch.ones(vocab_size, dtype=torch.bool, device=ctx.probs.device)
            illegal_mask[ctx.legal_ids] = False
            illegal_mass = float(ctx.probs[illegal_mask].sum().item())
            self.buffer.append(illegal_mass)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {"illegal_mass_when_gives_check": 0.0}
        return {"illegal_mass_when_gives_check": sum(self.buffer) / len(self.buffer)}
