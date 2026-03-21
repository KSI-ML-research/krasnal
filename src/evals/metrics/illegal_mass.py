from __future__ import annotations

from typing import Any

import chess
import chess.engine


class IllegalMassMetric:
    """Compute probability mass assigned to illegal moves."""

    name = "illegal_mass"

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
        legal_mass = float(probs[list(legal_ids)].sum().item()) if legal_ids else 0.0
        illegal_mass = 1.0 - legal_mass
        return {"illegal_mass": illegal_mass}
