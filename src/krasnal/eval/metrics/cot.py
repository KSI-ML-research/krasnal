from typing import Any

import torch

from .base import Metric
from .context import EvalContext


class CotFormatValidMetric(Metric):
    """Measure whether the generated CoT sequence is structurally valid."""

    @property
    def name(self) -> str:
        return "cot_format_valid"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        return {"cot_format_valid": 1.0 if ctx.cot_format_valid else 0.0}


class CotPostThinkTop1Metric(Metric):
    """Measure whether the post-think top-1 token matches the target move."""

    @property
    def name(self) -> str:
        return "cot_post_think_top1"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        probs = ctx.cot_post_think_probs
        actual_token = ctx.cot_post_think_actual_token
        if probs is None or actual_token is None or probs.numel() == 0:
            return {"cot_post_think_top1": 0.0}
        top1_token = int(torch.argmax(probs).item())
        return {"cot_post_think_top1": 1.0 if top1_token == actual_token else 0.0}


class CotPostThinkMRRMetric(Metric):
    """Measure the reciprocal rank of the target move after the think block."""

    @property
    def name(self) -> str:
        return "cot_post_think_mrr"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        probs = ctx.cot_post_think_probs
        actual_token = ctx.cot_post_think_actual_token
        if probs is None or actual_token is None or probs.numel() == 0:
            return {"cot_post_think_mrr": 0.0}
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == actual_token).nonzero(as_tuple=True)[0]
        if rank.numel() == 0:
            return {"cot_post_think_mrr": 0.0}
        return {"cot_post_think_mrr": 1.0 / (rank.item() + 1)}


class CotPostThinkTop1LegalMetric(Metric):
    """Measure whether the post-think top-1 token is legal in the position."""

    @property
    def name(self) -> str:
        return "cot_post_think_top1_legal"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        probs = ctx.cot_post_think_probs
        legal_ids = ctx.cot_post_think_legal_ids
        if probs is None or not legal_ids:
            return {"cot_post_think_top1_legal": 0.0}
        top1_token = int(torch.argmax(probs).item())
        return {"cot_post_think_top1_legal": 1.0 if top1_token in legal_ids else 0.0}


class CotThinkTokenRecallMetric(Metric):
    """Measure set recall of generated think tokens against target PV tokens."""

    @property
    def name(self) -> str:
        return "cot_think_token_recall"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        target_tokens = ctx.target_think_tokens or []
        generated_tokens = ctx.generated_think_tokens or []
        if not target_tokens:
            return {"cot_think_token_recall": 0.0}
        target_set = set(target_tokens)
        generated_set = set(generated_tokens)
        hits = len(target_set & generated_set)
        return {"cot_think_token_recall": hits / len(target_set)}
