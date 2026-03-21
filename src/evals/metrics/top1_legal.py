from __future__ import annotations

from typing import Any

import chess
import chess.engine
import torch


class Top1LegalMetric:
    """Check if the model's top-1 prediction is a legal move."""

    name = "top1_legal"

    def compute(
        self,
        session: Any,
        _board: chess.Board,
        legal_ids: set[int],
        _engine: chess.engine.SimpleEngine | None,
        _generator: Any | None,
        _tokenizer: Any = None,
        _sampler: Any = None,
    ) -> dict[str, Any]:
        probs = session.get_probs()
        top1_id = int(torch.argmax(probs).item())
        return {"top1_legal": top1_id in legal_ids}
