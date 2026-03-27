from typing import Any

import torch

from .context import EvalContext

PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


class PerPieceLegalMetric:
    """Compute top-1 legal accuracy per piece type.

    Measures whether the model's highest-probability move is legal,
    broken down by the piece type that was moved (ground truth).
    """

    name = "target_piece_legal"

    def __init__(self):
        self.buffers: dict[int, list[float]] = {p: [] for p in PIECE_NAMES}

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.piece_type and ctx.piece_type in self.buffers:
            top1_token = torch.argmax(ctx.probs).item()
            is_legal = 1.0 if top1_token in ctx.legal_ids else 0.0
            self.buffers[ctx.piece_type].append(is_legal)
        return {}

    def finalize(self) -> dict[str, Any]:
        result = {}
        for ptype, values in self.buffers.items():
            name = PIECE_NAMES.get(ptype, f"piece_{ptype}")
            result[f"target_{name}_legal"] = sum(values) / len(values) if values else 0.0
        return result


class PerPieceAccuracyMetric:
    """Compute top-1 accuracy per piece type.

    Measures whether the model's highest-probability move matches
    the actual (ground truth) move, broken down by piece type.
    """

    name = "target_piece_acc"

    def __init__(self):
        self.buffers: dict[int, list[float]] = {p: [] for p in PIECE_NAMES}

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.piece_type and ctx.piece_type in self.buffers and ctx.actual_token is not None:
            top1_token = torch.argmax(ctx.probs).item()
            is_correct = 1.0 if top1_token == ctx.actual_token else 0.0
            self.buffers[ctx.piece_type].append(is_correct)
        return {}

    def finalize(self) -> dict[str, Any]:
        result = {}
        for ptype, values in self.buffers.items():
            name = PIECE_NAMES.get(ptype, f"piece_{ptype}")
            result[f"target_{name}_acc"] = sum(values) / len(values) if values else 0.0
        return result
