"""Training-frequency baseline for ``what_is_on`` (square, ply) → argmax label."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import bulletchess

from krasnal.tokens import COLORED_PIECE_TOKENS, EMPTY_ID

_WHAT_IS_ON_LABEL_IDS: tuple[int, ...] = (EMPTY_ID, *sorted(COLORED_PIECE_TOKENS.values()))
_SQUARES: tuple[tuple[str, bulletchess.Square], ...] = tuple(
    (f"{file}{rank}", bulletchess.Square.from_str(f"{file}{rank}"))
    for file in "abcdefgh"
    for rank in range(1, 9)
)


def _label_id_for_piece(piece) -> int:
    if piece is None:
        return EMPTY_ID
    color_str = "w" if str(piece.color) == "White" else "b"
    piece_str = str(piece.piece_type).lower()
    return COLORED_PIECE_TOKENS[f"<{color_str}:{piece_str}>"]


class WhatIsOnBaselineCounts:
    """Empirical counts from training positions (after each full move, same ply as eval)."""

    def __init__(
        self,
        by_sq_ply: dict[tuple[str, int], dict[int, int]],
        by_sq: dict[str, dict[int, int]],
    ) -> None:
        self.by_sq_ply = by_sq_ply
        self.by_sq = by_sq

    def predict(self, sq: str, ply: int) -> int:
        row = self.by_sq_ply.get((sq, ply))
        if row is not None and sum(row.values()) > 0:
            return max(_WHAT_IS_ON_LABEL_IDS, key=lambda tid: (row.get(tid, 0), -tid))
        row_sq = self.by_sq.get(sq, {})
        return max(_WHAT_IS_ON_LABEL_IDS, key=lambda tid: (row_sq.get(tid, 0), -tid))

    def to_json_obj(self) -> dict[str, Any]:
        by_sq_ply_s: dict[str, dict[str, int]] = {}
        for (sq, ply), c in self.by_sq_ply.items():
            by_sq_ply_s[f"{sq}:{ply}"] = {str(k): v for k, v in c.items()}
        by_sq_s = {sq: {str(k): v for k, v in c.items()} for sq, c in self.by_sq.items()}
        return {"by_sq_ply": by_sq_ply_s, "by_sq": by_sq_s}

    @classmethod
    def from_json_obj(cls, data: dict[str, Any]) -> WhatIsOnBaselineCounts:
        by_sq_ply: dict[tuple[str, int], dict[int, int]] = {}
        for key, counts in data["by_sq_ply"].items():
            sq, _, ply_s = key.partition(":")
            by_sq_ply[(sq, int(ply_s))] = {int(tid): n for tid, n in counts.items()}
        by_sq: dict[str, dict[int, int]] = {
            sq: {int(tid): n for tid, n in c.items()} for sq, c in data["by_sq"].items()
        }
        return cls(by_sq_ply, by_sq)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_obj(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> WhatIsOnBaselineCounts:
        return cls.from_json_obj(json.loads(path.read_text(encoding="utf-8")))


class WhatIsOnBaselineAccumulator:
    """Incrementally accumulates empirical ``what_is_on`` baseline counts."""

    def __init__(self) -> None:
        self.by_sq_ply: dict[tuple[str, int], defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.by_sq: dict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))

    def update_board(self, board: bulletchess.Board, ply: int) -> None:
        for sq, bullet_sq in _SQUARES:
            lid = _label_id_for_piece(board[bullet_sq])
            self.by_sq_ply[(sq, ply)][lid] += 1
            self.by_sq[sq][lid] += 1

    def to_counts(self) -> WhatIsOnBaselineCounts:
        return WhatIsOnBaselineCounts(
            {k: dict(v) for k, v in self.by_sq_ply.items()},
            {k: dict(v) for k, v in self.by_sq.items()},
        )
