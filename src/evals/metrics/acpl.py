from __future__ import annotations

import logging
from typing import Any

import chess
import chess.engine

logger = logging.getLogger(__name__)


class ACPLMetric:
    """Average Centipawn Loss using Stockfish engine."""

    name = "acpl"

    def __init__(self, stockfish_path: str = "stockfish", time_limit: float = 0.05):
        self.stockfish_path = stockfish_path
        self.time_limit = time_limit

    def compute(
        self,
        session: Any,
        board: chess.Board,
        legal_ids: set[int],
        engine: chess.engine.SimpleEngine | None,
        generator: Any | None,
        tokenizer: Any = None,
        sampler: Any = None,
    ) -> dict[str, Any]:
        if not engine or not legal_ids or not generator or not tokenizer or not sampler:
            return {"acpl": None}

        best_uci = self._get_best_move(session, board, generator, tokenizer, sampler)
        if not best_uci:
            return {"acpl": None}

        return {"acpl": self._compute_acpl(engine, board, best_uci)}

    def _get_best_move(
        self, session: Any, board: chess.Board, generator: Any, tokenizer: Any, sampler: Any
    ) -> str | None:
        return generator.generate_move(session, board, tokenizer, sampler, temperature=0.0)

    def _compute_acpl(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        model_move_uci: str,
    ) -> float:
        try:
            limit = chess.engine.Limit(time=self.time_limit)
            info = engine.analyse(board, limit=limit)
            best_score = (
                info["score"].pov(board.turn).score(mate_score=10000)
                if info and "score" in info
                else 0
            )

            if info.get("pv") and model_move_uci == info["pv"][0].uci():
                return 0

            model_move = chess.Move.from_uci(model_move_uci)
            if model_move in board.legal_moves:
                board_copy = board.copy()
                board_copy.push(model_move)
                score_after = (
                    engine.analyse(board_copy, limit=limit)["score"]
                    .pov(board.turn)
                    .score(mate_score=10000)
                )
                return max(best_score - score_after, 0)
        except Exception as e:
            logger.debug(f"Error computing ACPL: {e}")
        return 0
