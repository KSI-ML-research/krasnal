import logging
from typing import Any

from ..stockfish import StockfishClient
from .context import EvalContext

logger = logging.getLogger(__name__)


class ACPLMetric:
    """Compute Average Centipawn Loss (ACPL).

    Measures the average difference between Stockfish evaluation before a move
    and after the model's top-1 legal move.
    Lower is better - 0 means model always plays its top-1 legal move.
    """

    name = "acpl"

    def __init__(self, stockfish: StockfishClient | None = None, sample_size: int = 100):
        self.stockfish = stockfish
        self.sample_size = sample_size
        self._contexts: list[EvalContext] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        self._contexts.append(ctx)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.stockfish:
            logger.warning("ACPL: no stockfish client provided")
            return {"acpl": 0.0}

        if len(self._contexts) < 2:
            logger.warning(f"ACPL: not enough contexts ({len(self._contexts)})")
            return {"acpl": 0.0}

        import random

        indices = list(range(len(self._contexts)))
        if len(indices) > self.sample_size:
            indices = random.sample(indices, self.sample_size)

        total_loss = 0.0
        count = 0
        skipped_illegal = 0
        skipped_stockfish = 0

        for i in indices:
            ctx = self._contexts[i]

            if ctx.fen is None or ctx.top1_fen is None:
                skipped_illegal += 1
                continue

            cp_before = self.stockfish.get_eval(ctx.fen)
            if cp_before is None:
                skipped_stockfish += 1
                continue

            cp_after = self.stockfish.get_eval(ctx.top1_fen)
            if cp_after is None:
                skipped_stockfish += 1
                continue

            total_loss += cp_after - cp_before
            count += 1

        if skipped_illegal > 0 or skipped_stockfish > 0:
            logger.warning(
                f"ACPL: computed={count}, skipped_illegal={skipped_illegal}, "
                f"skipped_stockfish={skipped_stockfish}"
            )

        result = {
            "acpl": total_loss / count if count > 0 else 0.0,
        }

        self._contexts = []

        return result
