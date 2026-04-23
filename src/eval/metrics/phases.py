from typing import Any

import torch

from .context import EvalContext

PHASES = ["opening", "middlegame", "endgame"]


class Top1LegalPhaseMetric:
    """Compute top-1 legal move accuracy by game phase.

    Phases: opening (0-20 plies), middlegame (20-80 plies), endgame (80+ plies)
    """

    def __init__(self, phase: str):
        self.phase = phase
        self.name = f"top1_legal_{phase}"
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.phase == self.phase and ctx.legal_ids:
            # Use argmax over all tokens - consistency with Top1LegalMetric
            # This measures: is the top token (from entire vocab) a legal move?
            top1_token = torch.argmax(ctx.probs).item()
            is_legal = 1.0 if top1_token in ctx.legal_ids else 0.0
            self.buffer.append(is_legal)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {self.name: 0.0}
        return {self.name: sum(self.buffer) / len(self.buffer)}


class AccPhaseMetric:
    """Compute top-1 accuracy by game phase.

    Measures whether the model's highest-probability token matches
    the actual (ground truth) move, broken down by game phase.
    Phases: opening (0-20 plies), middlegame (20-80 plies), endgame (80+ plies)
    """

    def __init__(self, phase: str):
        self.phase = phase
        self.name = f"acc_{phase}"
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.phase == self.phase and ctx.actual_token is not None:
            top1_token = torch.argmax(ctx.probs).item()
            is_correct = 1.0 if top1_token == ctx.actual_token else 0.0
            self.buffer.append(is_correct)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {self.name: 0.0}
        return {self.name: sum(self.buffer) / len(self.buffer)}


class IllegalMassPhaseMetric:
    """Compute illegal mass by game phase.

    Sums the model's probability distribution over all illegal move tokens,
    broken down by game phase.
    Phases: opening (0-20 plies), middlegame (20-80 plies), endgame (80+ plies)
    """

    def __init__(self, phase: str):
        self.phase = phase
        self.name = f"illegal_mass_{phase}"
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.phase == self.phase:
            vocab_size = ctx.probs.shape[0]
            illegal_mask = torch.ones(vocab_size, dtype=torch.bool, device=ctx.probs.device)
            illegal_mask[ctx.legal_ids] = False
            illegal_mass = float(ctx.probs[illegal_mask].sum().item())
            self.buffer.append(illegal_mass)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {self.name: 0.0}
        return {self.name: sum(self.buffer) / len(self.buffer)}
