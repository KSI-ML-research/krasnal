from typing import Any

from loguru import logger

from ..stockfish import StockfishClient
from .base import Metric
from .context import EvalContext


class ACPLMetric(Metric):
    """Compute Average Centipawn Loss (ACPL).

    Measures the average difference between Stockfish evaluation before a move
    and after the model's top-1 legal move.
    Lower is better - 0 means model always plays its top-1 legal move.
    """

    @property
    def name(self) -> str:
        return "acpl"

    def __init__(self, stockfish: StockfishClient | None = None, sample_size: int = 100):
        self.stockfish = stockfish
        self.sample_size = sample_size
        self._contexts: list[EvalContext] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        self._contexts.append(ctx)
        return {}

    @staticmethod
    def _side_to_move_from_fen(fen: str) -> str:
        parts = fen.split()
        if len(parts) < 2 or parts[1] not in {"w", "b"}:
            raise ValueError(f"Invalid FEN: {fen}")
        return parts[1]

    @classmethod
    def _normalize_to_mover_perspective(
        cls,
        *,
        cp_before: float,
        cp_after: float,
        fen_before: str,
        fen_after: str,
    ) -> tuple[float, float]:
        """Express before/after evals from the same player's perspective.

        StockfishClient scores are treated as relative to the side to move.
        Before the move, that is the mover; after the move, that is the opponent.
        """
        mover_side = cls._side_to_move_from_fen(fen_before)
        before_side = cls._side_to_move_from_fen(fen_before)
        after_side = cls._side_to_move_from_fen(fen_after)

        normalized_before = cp_before if before_side == mover_side else -cp_before
        normalized_after = cp_after if after_side == mover_side else -cp_after
        return normalized_before, normalized_after

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
            cp_after = self.stockfish.get_eval(ctx.top1_fen)

            if cp_before is None or cp_after is None:
                skipped_stockfish += 1
                continue

            cp_before_norm, cp_after_norm = self._normalize_to_mover_perspective(
                cp_before=cp_before,
                cp_after=cp_after,
                fen_before=ctx.fen,
                fen_after=ctx.top1_fen,
            )

            total_loss += cp_before_norm - cp_after_norm
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
