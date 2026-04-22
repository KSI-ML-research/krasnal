from abc import ABC, abstractmethod
from typing import Any

import torch

from .base import Metric
from .context import EvalContext


class CoreMetric(Metric, ABC):
    """Base for core metrics that compute a value from EvalContext."""

    name: str

    @abstractmethod
    def compute_value(self, ctx: EvalContext) -> float | None:
        """Returns None if metric doesn't apply to this context."""

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        value = self.compute_value(ctx)
        if value is None:
            return {}
        return {self.name: value}


class Top1LegalCore(CoreMetric):
    """Check if model's top-1 prediction is a legal move."""

    name = "top1_legal"

    def compute_value(self, ctx: EvalContext) -> float | None:
        if ctx.probs is None or ctx.legal_ids is None:
            return None
        top1 = torch.argmax(ctx.probs).item()
        return 1.0 if top1 in ctx.legal_ids else 0.0


class AccuracyCore(CoreMetric):
    """Check if top-1 matches actual move."""

    name = "acc"

    def compute_value(self, ctx: EvalContext) -> float | None:
        if ctx.probs is None or ctx.actual_token is None:
            return None
        top1 = torch.argmax(ctx.probs).item()
        return 1.0 if top1 == ctx.actual_token else 0.0


class IllegalMassCore(CoreMetric):
    """Sum probability mass on illegal moves."""

    name = "illegal_mass"

    def compute_value(self, ctx: EvalContext) -> float | None:
        if ctx.probs is None or ctx.legal_ids is None:
            return None
        vocab_size = ctx.probs.shape[0]
        illegal_mask = torch.ones(vocab_size, dtype=torch.bool, device=ctx.probs.device)
        illegal_mask[ctx.legal_ids] = False
        return float(ctx.probs[illegal_mask].sum().item())


class MRRCore(CoreMetric):
    """Mean Reciprocal Rank of actual token in probability distribution."""

    name = "mrr"

    def compute_value(self, ctx: EvalContext) -> float | None:
        if ctx.probs is None or ctx.actual_token is None:
            return None
        sorted_idx = torch.argsort(ctx.probs, descending=True)
        rank = (sorted_idx == ctx.actual_token).nonzero()
        if rank.numel() == 0:
            return 0.0
        return 1.0 / (rank.item() + 1)
