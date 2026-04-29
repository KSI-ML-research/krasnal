import torch

from krasnal.eval.puzzles import (
    DEFAULT_PUZZLE_BUCKETS,
    PuzzleEvalResult,
    estimate_pseudo_elo,
    evaluate_model_on_puzzles,
)
from krasnal.tokens import MOVE_TO_ID


class _DummyModel:
    pass


def test_estimate_pseudo_elo_empty_is_zero():
    assert estimate_pseudo_elo([]) == 0.0


def test_estimate_pseudo_elo_rises_with_stronger_results():
    easy = [(1000, 1), (1100, 1), (1200, 0)]
    hard = [(1600, 1), (1700, 1), (1800, 0)]

    assert estimate_pseudo_elo(hard) > estimate_pseudo_elo(easy)


def test_puzzle_eval_result_filters_metrics():
    result = PuzzleEvalResult(
        overall={
            "overall/exact_match": 0.5,
            "overall/mrr": 0.75,
            "overall/pseudo_elo": 1400,
            "overall/total": 2,
            "overall/source_total": 10,
        },
        buckets={
            "1000_1200": {
                "bucket/1000_1200/exact_match": 1.0,
                "bucket/1000_1200/mrr": 1.0,
                "bucket/1000_1200/pseudo_elo": 1500,
                "bucket/1000_1200/total": 1,
            }
        },
    )

    metrics = result.to_metrics()
    assert metrics == {
        "puzzle/overall/exact_match": 0.5,
        "puzzle/overall/pseudo_elo": 1400,
    }

    metrics = result.to_metrics(log_mrr=True, log_bucket_metrics=True, log_diagnostics=True)
    assert metrics["puzzle/overall/mrr"] == 0.75
    assert metrics["puzzle/overall/total"] == 2
    assert metrics["puzzle/overall/source_total"] == 10
    assert metrics["puzzle/bucket/1000_1200/mrr"] == 1.0
    assert metrics["puzzle/bucket/1000_1200/total"] == 1


def test_evaluate_model_on_puzzles_computes_exact_match_and_mrr(monkeypatch):
    from krasnal.eval import puzzles as puzzles_mod

    class _FakeMove:
        def __init__(self, uci: str):
            self._uci = uci

        def uci(self) -> str:
            return self._uci

    class _FakeBoard:
        turn = "White"

        @staticmethod
        def from_fen(_fen: str):
            return _FakeBoard()

        @staticmethod
        def legal_moves():
            return [_FakeMove("e2e4"), _FakeMove("d2d4")]

    class _DummySession:
        def __init__(self, _model, _device, game):
            self.game = game

        def get_legal_probs(self):
            top = MOVE_TO_ID["w:e2e4"]
            second = MOVE_TO_ID["w:d2d4"]
            probs = torch.zeros(max(top, second) + 1, dtype=torch.float32)
            probs[top] = 1.0
            probs[second] = 0.5
            return probs

    monkeypatch.setattr(puzzles_mod, "InferenceSession", _DummySession)
    monkeypatch.setattr(
        puzzles_mod,
        "legal_token_ids",
        lambda _board: [MOVE_TO_ID["w:e2e4"], MOVE_TO_ID["w:d2d4"]],
    )
    monkeypatch.setattr(puzzles_mod.bulletchess, "Board", _FakeBoard)

    result = evaluate_model_on_puzzles(
        model=_DummyModel(),
        device=torch.device("cpu"),
        puzzles=[
            {"fen": "fen-1", "solution": "e2e4", "rating": 1100},
            {"fen": "fen-2", "solution": "d2d4", "rating": 1300},
        ],
    )

    assert result.overall["overall/evaluated"] == 2
    assert result.overall["overall/exact_match"] == 0.5
    assert result.overall["overall/mrr"] == 0.75
    assert result.buckets["1000_1200"]["bucket/1000_1200/evaluated"] == 1
    assert result.buckets["1200_1400"]["bucket/1200_1400/evaluated"] == 1
    assert result.to_metrics(log_bucket_metrics=True)["puzzle/bucket/1000_1200/exact_match"] == 1.0


def test_default_puzzle_buckets_cover_expected_ranges():
    assert [bucket.name for bucket in DEFAULT_PUZZLE_BUCKETS] == [
        "1000_1200",
        "1200_1400",
        "1400_1600",
        "1600_1800",
        "1800_plus",
    ]
