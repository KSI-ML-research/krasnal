from typing import Any

import torch

from .context import EvalContext

PHASES = ["opening", "middlegame", "endgame"]


class Top1LegalPhaseMetric:
    """Compute top-1 legal move accuracy by game phase.

    Phases: opening (0-20 plies), middlegame (20-80 plies), endgame (80+ plies)
    """

    name = "top1_legal"

    def __init__(self):
        self.buffers: dict[str, list[float]] = {phase: [] for phase in PHASES}

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.phase and ctx.phase in self.buffers:
            top1_token = torch.argmax(ctx.probs).item()
            is_legal = 1.0 if top1_token in ctx.legal_ids else 0.0
            self.buffers[ctx.phase].append(is_legal)
        return {}

    def finalize(self) -> dict[str, Any]:
        result = {}
        for phase, values in self.buffers.items():
            result[f"top1_legal_{phase}"] = sum(values) / len(values) if values else 0.0
        return result


class AccPhaseMetric:
    """Compute top-1 accuracy by game phase.

    Measures whether the model's highest-probability token matches
    the actual (ground truth) move, broken down by game phase.
    Phases: opening (0-20 plies), middlegame (20-80 plies), endgame (80+ plies)
    """

    name = "acc"

    def __init__(self):
        self.buffers: dict[str, list[float]] = {phase: [] for phase in PHASES}

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.phase and ctx.phase in self.buffers and ctx.actual_token is not None:
            top1_token = torch.argmax(ctx.probs).item()
            is_correct = 1.0 if top1_token == ctx.actual_token else 0.0
            self.buffers[ctx.phase].append(is_correct)
        return {}

    def finalize(self) -> dict[str, Any]:
        result = {}
        for phase, values in self.buffers.items():
            result[f"acc_{phase}"] = sum(values) / len(values) if values else 0.0
        return result


class IllegalMassPhaseMetric:
    """Compute illegal mass by game phase.

    Sums the model's probability distribution over all illegal move tokens,
    broken down by game phase.
    Phases: opening (0-20 plies), middlegame (20-80 plies), endgame (80+ plies)
    """

    name = "illegal_mass"

    def __init__(self):
        self.buffers: dict[str, list[float]] = {phase: [] for phase in PHASES}

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.phase and ctx.phase in self.buffers:
            vocab_size = ctx.probs.shape[0]
            illegal_mask = torch.ones(vocab_size, dtype=torch.bool, device=ctx.probs.device)
            illegal_mask[ctx.legal_ids] = False
            illegal_mass = float(ctx.probs[illegal_mask].sum().item())
            self.buffers[ctx.phase].append(illegal_mass)
        return {}

    def finalize(self) -> dict[str, Any]:
        result = {}
        for phase, values in self.buffers.items():
            result[f"illegal_mass_{phase}"] = sum(values) / len(values) if values else 0.0
        return result
