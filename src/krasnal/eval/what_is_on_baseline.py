"""Training-frequency baseline for ``what_is_on`` (square, ply) → argmax label."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import bulletchess

from krasnal.eval.metrics.context import EvalContext
from krasnal.tokens import COLORED_PIECE_TOKENS, EMPTY_ID

_WHAT_IS_ON_LABEL_IDS: tuple[int, ...] = (EMPTY_ID, *sorted(COLORED_PIECE_TOKENS.values()))


def _label_id_for_square(board: bulletchess.Board, sq_str: str) -> int:
    piece = board[bulletchess.Square.from_str(sq_str)]
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


def accumulate_from_eval_contexts(contexts: list[EvalContext]) -> WhatIsOnBaselineCounts:
    """Build counts from replayed contexts (``post_move_fen``, ``what_is_on_ply``)."""
    by_sq_ply_dd: dict[tuple[str, int], defaultdict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    by_sq_dd: dict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))

    files = "abcdefgh"
    for ctx in contexts:
        if ctx.post_move_fen is None or ctx.what_is_on_ply is None:
            continue
        board = bulletchess.Board.from_fen(ctx.post_move_fen)
        ply = int(ctx.what_is_on_ply)
        for fi in range(8):
            for ri in range(8):
                sq = f"{files[fi]}{ri + 1}"
                lid = _label_id_for_square(board, sq)
                by_sq_ply_dd[(sq, ply)][lid] += 1
                by_sq_dd[sq][lid] += 1

    return WhatIsOnBaselineCounts(
        {k: dict(v) for k, v in by_sq_ply_dd.items()},
        {k: dict(v) for k, v in by_sq_dd.items()},
    )


def macro_f1_multiclass(y_true: list[int], y_pred: list[int], *, labels: tuple[int, ...]) -> float:
    """Unweighted mean of per-class F1.

    Ignores classes with no true instances and no predictions.
    """
    if not y_true:
        return 0.0
    total = 0.0
    valid_labels = 0
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p != c)

        if tp + fp + fn == 0:
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        total += f1
        valid_labels += 1

    return total / valid_labels if valid_labels > 0 else 0.0
