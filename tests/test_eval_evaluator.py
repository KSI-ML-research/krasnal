from types import SimpleNamespace

import bulletchess
import torch

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.eval.evaluator import (
    ChessEvaluator,
    _is_legal_token_in_position,
    _MoveMetricAccumulator,
    _update_context_sample,
)
from krasnal.eval.metrics import EvalContext
from krasnal.eval.qa_probes import build_what_is_on_heatmap, compute_binary_f1_metrics
from krasnal.tokens import (
    ELO_1500_1599_ID,
    ELO_1600_1699_ID,
    GAME_END_ID,
    GAME_START_ID,
    get_vocab_size,
    move_token_id_for_turn,
)


def test_compute_binary_f1_metrics_returns_expected_values():
    result = compute_binary_f1_metrics(tp=3, fp=1, fn=2)

    assert result["qa/is_check/precision"] == 0.75
    assert result["qa/is_check/recall"] == 0.6
    assert result["qa/is_check/f1"] == 2 * 0.75 * 0.6 / (0.75 + 0.6)


def test_build_what_is_on_heatmap_uses_all_squares():
    square_accs = {f"{file}{rank}": float(rank) for rank in range(1, 9) for file in "abcdefgh"}

    heatmap = build_what_is_on_heatmap(square_accs)

    assert heatmap is not None


def test_evaluate_resets_metrics_between_runs(monkeypatch):
    evaluator = ChessEvaluator(metrics=["acc"])

    dataset = [SimpleNamespace(tolist=lambda: [GAME_START_ID, GAME_END_ID])]
    model = SimpleNamespace(config=SimpleNamespace(block_size=128))

    def fake_evaluate_indices(**_kwargs):
        return {"acc": 0.0}

    monkeypatch.setattr(evaluator, "_evaluate_indices", fake_evaluate_indices)

    result = evaluator.evaluate(model=model, dataset=dataset, num_games=1, device=None)

    assert result == {"acc": 0.0}


def test_evaluate_num_games_zero_uses_all_games(monkeypatch):
    evaluator = ChessEvaluator(metrics=["acc"])
    dataset = [object(), object(), object()]
    model = SimpleNamespace(config=SimpleNamespace(block_size=128))

    def fake_evaluate_indices(**kwargs):
        assert kwargs["indices"] == [0, 1, 2]
        return {"acc": 1.0}

    monkeypatch.setattr("krasnal.eval.evaluator.random.shuffle", lambda _indices: None)
    monkeypatch.setattr(evaluator, "_evaluate_indices", fake_evaluate_indices)

    result = evaluator.evaluate(model=model, dataset=dataset, num_games=0, device=None)

    assert result == {"acc": 1.0}


def test_move_metric_accumulator_computes_requested_metrics():
    board = bulletchess.Board()
    legal_move = bulletchess.Move.from_uci("e2e4")
    legal_piece = board[legal_move.origin]
    legal_token = move_token_id_for_turn("e2e4", board.turn, legal_piece.piece_type)
    black_board = board.copy()
    black_board.apply(legal_move)
    black_move = bulletchess.Move.from_uci("e7e5")
    black_piece = black_board[black_move.origin]
    illegal_for_start_position = move_token_id_for_turn(
        "e7e5",
        black_board.turn,
        black_piece.piece_type,
    )
    assert legal_token is not None
    assert illegal_for_start_position is not None

    metrics = [
        "acc",
        "acc_opening",
        "acc_middlegame",
        "acc_endgame",
        "acc_when_gives_check",
        "acc_when_in_check",
        "acc_elo_1500_1599",
        "acc_elo_1600_1699",
        "top1_legal",
    ]
    contexts = [
        EvalContext(
            fen=board.fen(),
            actual_token=legal_token,
            in_check=False,
            phase="opening",
            player_elo_token=ELO_1500_1599_ID,
            gives_check=True,
            active_clock_seconds=90,
        ),
        EvalContext(
            fen=board.fen(),
            actual_token=legal_token,
            in_check=True,
            phase="middlegame",
            player_elo_token=ELO_1600_1699_ID,
            gives_check=False,
            active_clock_seconds=20,
        ),
        EvalContext(
            fen=board.fen(),
            actual_token=legal_token,
            in_check=False,
            phase="endgame",
            player_elo_token=ELO_1500_1599_ID,
            gives_check=True,
            active_clock_seconds=CLOCK_IGNORE_ID,
        ),
    ]

    accumulator = _MoveMetricAccumulator(metrics)
    logits = torch.zeros((3, get_vocab_size()))
    logits[0, legal_token] = 1.0
    logits[1, illegal_for_start_position] = 1.0
    logits[2, legal_token] = 1.0
    accumulator.update(contexts, logits)
    result = accumulator.finalize()

    assert result == {
        "acc": 2 / 3,
        "acc_opening": 1.0,
        "acc_middlegame": 0.0,
        "acc_endgame": 1.0,
        "acc_when_gives_check": 1.0,
        "acc_when_in_check": 0.0,
        "acc/acc_elo_1500_1599": 1.0,
        "acc/acc_elo_1600_1699": 0.0,
        "top1_legal": 2 / 3,
    }


def test_fen_based_top1_legal_matches_board_legal_moves():
    board = bulletchess.Board()
    legal_move = bulletchess.Move.from_uci("e2e4")
    legal_piece = board[legal_move.origin]
    legal_token = move_token_id_for_turn("e2e4", board.turn, legal_piece.piece_type)
    black_board = board.copy()
    black_board.apply(legal_move)
    black_move = bulletchess.Move.from_uci("e7e5")
    black_piece = black_board[black_move.origin]
    illegal_token = move_token_id_for_turn("e7e5", black_board.turn, black_piece.piece_type)
    ctx = EvalContext(fen=board.fen())

    assert legal_token is not None
    assert illegal_token is not None
    assert _is_legal_token_in_position(ctx, legal_token)
    assert not _is_legal_token_in_position(ctx, illegal_token)


def test_update_context_sample_caps_sample_deterministically():
    import random

    contexts = [EvalContext(actual_token=i) for i in range(20)]
    sample: list[EvalContext] = []
    seen = _update_context_sample(
        sample=sample,
        seen=0,
        contexts=contexts,
        limit=5,
        rng=random.Random(0),
    )

    assert seen == 20
    assert len(sample) == 5
    assert [ctx.actual_token for ctx in sample] == [8, 1, 2, 5, 9]
