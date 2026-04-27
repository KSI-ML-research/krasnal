from pathlib import Path

import pytest
import torch

from krasnal.eval.puzzles import (
    DEFAULT_PUZZLE_BUCKETS,
    estimate_pseudo_elo,
    evaluate_model_on_puzzles,
)


class _DummyModel:
    pass


class _DummySession:
    def __init__(self, _model, _device, game):
        self.game = game

    def get_legal_probs(self):
        probs = torch.zeros(100, dtype=torch.float32)
        token = 42
        probs[token] = 1.0
        return probs


def test_estimate_pseudo_elo_monotonicity():
    easy = [(1000, 1), (1000, 1), (1100, 1), (1200, 0)]
    hard = [(1600, 1), (1700, 1), (1800, 1), (1900, 0)]

    easy_elo = estimate_pseudo_elo(easy)
    hard_elo = estimate_pseudo_elo(hard)

    assert hard_elo > easy_elo


def test_estimate_pseudo_elo_empty_is_zero():
    assert estimate_pseudo_elo([]) == 0.0


def test_bucket_coverage_definition():
    names = [bucket.name for bucket in DEFAULT_PUZZLE_BUCKETS]
    assert names == [
        "1000_1200",
        "1200_1400",
        "1400_1600",
        "1600_1800",
        "1800_plus",
    ]


def test_evaluate_model_on_puzzles_bucketed_metrics(monkeypatch):
    from krasnal.eval import puzzles as puzzles_mod

    monkeypatch.setattr(puzzles_mod, "InferenceSession", _DummySession)
    monkeypatch.setattr(puzzles_mod, "legal_token_ids", lambda _board: [42])
    monkeypatch.setattr(puzzles_mod, "to_uci", lambda _token: "e2e4")

    class _FakeMove:
        def __init__(self, uci):
            self._uci = uci

        def uci(self):
            return self._uci

    class _FakeBoard:
        turn = "White"

        @staticmethod
        def from_fen(_fen):
            return _FakeBoard()

        @staticmethod
        def legal_moves():
            return [_FakeMove("e2e4"), _FakeMove("d2d4")]

    monkeypatch.setattr(puzzles_mod.bulletchess, "Board", _FakeBoard)

    puzzles = [
        {"fen": "f 0", "solution": "e2e4", "rating": 1100},
        {"fen": "f 1", "solution": "d2d4", "rating": 1300},
        {"fen": "f 2", "solution": "e2e4", "rating": 1500},
        {"fen": "f 3", "solution": "e2e4", "rating": 1700},
        {"fen": "f 4", "solution": "e2e4", "rating": 1850},
    ]

    metrics = evaluate_model_on_puzzles(
        model=_DummyModel(),
        device=torch.device("cpu"),
        puzzles=puzzles,
    )

    assert metrics["overall/evaluated"] == 5
    assert metrics["overall/exact_match"] == pytest.approx(0.8)
    assert metrics["bucket/1000_1200/evaluated"] == 1
    assert metrics["bucket/1200_1400/evaluated"] == 1
    assert metrics["bucket/1400_1600/evaluated"] == 1
    assert metrics["bucket/1600_1800/evaluated"] == 1
    assert metrics["bucket/1800_plus/evaluated"] == 1


def test_eval_puzzles_script_importable():
    script_path = Path("scripts/evals/eval_puzzles.py")
    assert script_path.exists()
