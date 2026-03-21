from __future__ import annotations

import logging
import random
from typing import Any

import chess
import chess.engine
import polars as pl
import torch
from tqdm.auto import tqdm

from ..dataset import ChessDataset
from ..inference import (
    DefaultSampler,
    InferenceSession,
    SimpleGenerator,
    get_legal_token_ids,
)
from ..tokenizer import DRAW_ID, SOS_ID, SPECIAL_TOKENS, WIN_BLACK_ID, WIN_WHITE_ID
from .metrics import METRIC_REGISTRY

logger = logging.getLogger(__name__)


class ChessEvaluator:
    """Evaluates chess model on legal move metrics."""

    DEFAULT_METRICS = ["top1_legal", "illegal_mass", "acpl"]

    def __init__(
        self,
        metrics: list[str] | None = None,
        stockfish_path: str = "stockfish",
        stockfish_time: float = 0.05,
        cot: bool = False,
        cot_max_tokens: int = 128,
    ):
        self.requested_metrics = metrics or self.DEFAULT_METRICS
        self.stockfish_path = stockfish_path
        self.stockfish_time = stockfish_time
        self.cot = cot
        self.cot_max_tokens = cot_max_tokens
        self.metrics = self._init_metrics()

    def _init_metrics(self) -> dict[str, Any]:
        metrics = {}
        if "acpl" in self.requested_metrics:
            from .metrics.acpl import ACPLMetric

            metrics["acpl"] = ACPLMetric(self.stockfish_path, self.stockfish_time)
        if "top1_legal" in self.requested_metrics:
            metrics["top1_legal"] = METRIC_REGISTRY["top1_legal"]()
        if "illegal_mass" in self.requested_metrics:
            metrics["illegal_mass"] = METRIC_REGISTRY["illegal_mass"]()
        if self.cot and "cot_acpl" in self.requested_metrics:
            from .metrics.cot_acpl import CoTACPLMetric

            metrics["cot_acpl"] = CoTACPLMetric(
                self.stockfish_path, self.stockfish_time, self.cot_max_tokens
            )
        return metrics

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        dataset: ChessDataset,
        num_games: int,
        device: torch.device,
    ) -> pl.DataFrame:
        special_ids = set(SPECIAL_TOKENS)
        block_size = model.config.block_size

        engine = None
        if "acpl" in self.requested_metrics or "cot_acpl" in self.requested_metrics:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                logger.error(f"Failed to start Stockfish at '{self.stockfish_path}': {e}")
                raise SystemExit(1) from None

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_games]

        results = []
        session = InferenceSession(model, device)
        sampler = DefaultSampler()
        simple_gen = SimpleGenerator()

        for idx in tqdm(indices, desc="Evaluating sampled games"):
            token_ids = dataset[idx].tolist()
            outcome_token = (
                token_ids[0]
                if token_ids and token_ids[0] in {WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID, SOS_ID}
                else SOS_ID
            )

            moves = [t for t in token_ids if t not in special_ids]
            if len(moves) < 1:
                continue

            if len(moves) + 1 > block_size:
                logger.error(
                    f"Game {idx} length ({len(moves) + 1}) exceeds {block_size=}. Skipping."
                )
                continue

            session.reset(outcome_token)
            board = chess.Board()

            for i, move_token in tqdm(
                enumerate(moves),
                total=len(moves),
                desc=f"Game {idx} moves",
                leave=False,
            ):
                legal_ids = get_legal_token_ids(board, tokenizer)
                if not legal_ids:
                    break

                result = {"move_num": i + 1}

                for metric in self.metrics.values():
                    result.update(
                        metric.compute(
                            session, board, legal_ids, engine, simple_gen, tokenizer, sampler
                        )
                    )

                results.append(result)

                uci_move = tokenizer.id_to_move.get(move_token)
                if not uci_move:
                    logger.warning(
                        "Stopping game %s at move %s: unknown token_id=%s",
                        idx,
                        i + 1,
                        move_token,
                    )
                    break
                try:
                    board.push_uci(uci_move)
                except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                    logger.warning(
                        "Stopping game %s at move %s: invalid replay move=%s token_id=%s fen=%s",
                        idx,
                        i + 1,
                        uci_move,
                        move_token,
                        board.fen(),
                    )
                    break
                session.feed(move_token)

        if engine:
            engine.quit()

        return pl.DataFrame(results) if results else pl.DataFrame()
