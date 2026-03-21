from __future__ import annotations

from typing import Any, Protocol

import chess
import chess.engine

from ..dataset import ChessDataset
from ..tokenizer import Tokenizer


class BaseMetric(Protocol):
    """Protocol for evaluation metrics."""

    name: str

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
        """Compute metric for a single position. Returns dict of results."""


class BaseEvaluator(Protocol):
    """Protocol for evaluators that run metrics on datasets."""

    def evaluate(
        self,
        model: Any,
        tokenizer: Tokenizer,
        dataset: ChessDataset,
        metrics: list[str] | None,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation on dataset with specified metrics."""
        ...
