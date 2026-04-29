from typing import Any

from loguru import logger

from ..stockfish import StockfishClient
from .base import Metric
from .context import EvalContext


class StockfishTop1AgreementMetric(Metric):
    """Compute agreement between the model's top-1 legal move and Stockfish."""

    @property
    def name(self) -> str:
        return "stockfish_top1"

    def __init__(self, stockfish: StockfishClient | None = None, sample_size: int = 100):
        self.stockfish = stockfish
        self.sample_size = sample_size
        self._contexts: list[EvalContext] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        self._contexts.append(ctx)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.stockfish:
            logger.warning("Stockfish top-1: no stockfish client provided")
            return {"stockfish_top1": 0.0}

        import random

        contexts = self._contexts
        if len(contexts) > self.sample_size:
            contexts = random.sample(contexts, self.sample_size)

        matches = 0
        count = 0
        skipped = 0
        for ctx in contexts:
            if ctx.fen is None or ctx.top1_move_uci is None:
                skipped += 1
                continue

            bestmove = self.stockfish.get_best_move(ctx.fen)
            if bestmove is None:
                skipped += 1
                continue

            matches += 1 if bestmove == ctx.top1_move_uci else 0
            count += 1

        if skipped > 0:
            logger.warning("Stockfish top-1: computed={}, skipped={}", count, skipped)

        self._contexts = []
        return {"stockfish_top1": matches / count if count > 0 else 0.0}
