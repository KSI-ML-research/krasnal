from typing import Any

from loguru import logger

from ..stockfish import StockfishClient
from .acpl import ACPLMetric
from .base import Metric
from .context import EvalContext


class BlunderRateMetric(Metric):
    """Compute the fraction of moves losing at least `threshold_cp` centipawns."""

    @property
    def name(self) -> str:
        return "blunder_rate"

    def __init__(
        self,
        stockfish: StockfishClient | None = None,
        sample_size: int = 100,
        threshold_cp: float = 100.0,
    ):
        self.stockfish = stockfish
        self.sample_size = sample_size
        self.threshold_cp = threshold_cp
        self._contexts: list[EvalContext] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        self._contexts.append(ctx)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.stockfish:
            logger.warning("Blunder rate: no stockfish client provided")
            return {"blunder_rate": 0.0}

        import random

        contexts = self._contexts
        if len(contexts) > self.sample_size:
            contexts = random.sample(contexts, self.sample_size)

        blunders = 0
        count = 0
        skipped = 0

        for ctx in contexts:
            if ctx.fen is None or ctx.top1_fen is None:
                skipped += 1
                continue

            cp_before = self.stockfish.get_eval(ctx.fen)
            cp_after = self.stockfish.get_eval(ctx.top1_fen)

            if cp_before is None or cp_after is None:
                skipped += 1
                continue

            cp_before_norm, cp_after_norm = ACPLMetric._normalize_to_mover_perspective(
                cp_before=cp_before,
                cp_after=cp_after,
                fen_before=ctx.fen,
                fen_after=ctx.top1_fen,
            )
            if cp_before_norm - cp_after_norm >= self.threshold_cp:
                blunders += 1
            count += 1

        if skipped > 0:
            logger.warning("Blunder rate: computed={}, skipped={}", count, skipped)

        self._contexts = []
        return {"blunder_rate": blunders / count if count > 0 else 0.0}
