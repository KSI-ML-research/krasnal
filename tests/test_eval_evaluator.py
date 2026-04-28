from types import SimpleNamespace

from krasnal.eval.evaluator import ChessEvaluator
from krasnal.tokens import (
    BLACK_PREFIX,
    GAME_END_ID,
    GAME_START_ID,
    MOVE_TO_ID,
    THINK_END_ID,
    THINK_START_ID,
    WHITE_PREFIX,
    WHITE_WON_ID,
)


def test_is_valid_cot_sequence_accepts_valid_sequence():
    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    c7c5 = MOVE_TO_ID[BLACK_PREFIX + "c7c5"]

    token_ids = [
        GAME_START_ID,
        WHITE_WON_ID,
        e2e4,
        THINK_START_ID,
        c7c5,
        THINK_END_ID,
        c7c5,
        GAME_END_ID,
    ]

    assert ChessEvaluator._is_valid_cot_sequence(token_ids) is True


def test_is_valid_cot_sequence_accepts_multiple_think_blocks():
    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    c7c5 = MOVE_TO_ID[BLACK_PREFIX + "c7c5"]
    e7e5 = MOVE_TO_ID[WHITE_PREFIX + "e7e5"]
    g1f3 = MOVE_TO_ID[BLACK_PREFIX + "g1f3"]

    token_ids = [
        GAME_START_ID,
        WHITE_WON_ID,
        e2e4,
        THINK_START_ID,
        c7c5,
        THINK_END_ID,
        c7c5,
        THINK_START_ID,
        e7e5,
        THINK_END_ID,
        g1f3,
        GAME_END_ID,
    ]

    assert ChessEvaluator._is_valid_cot_sequence(token_ids) is True


def test_is_valid_cot_sequence_rejects_missing_think():
    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    c7c5 = MOVE_TO_ID[BLACK_PREFIX + "c7c5"]

    token_ids = [
        GAME_START_ID,
        WHITE_WON_ID,
        e2e4,
        c7c5,
        GAME_END_ID,
    ]

    assert ChessEvaluator._is_valid_cot_sequence(token_ids) is False


def test_is_valid_cot_sequence_rejects_unclosed_think():
    token_ids = [
        GAME_START_ID,
        WHITE_WON_ID,
        THINK_START_ID,
        MOVE_TO_ID[WHITE_PREFIX + "e2e4"],
        GAME_END_ID,
    ]

    assert ChessEvaluator._is_valid_cot_sequence(token_ids) is False


def test_extract_generated_think_tokens_extracts_between_think_markers():
    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    c7c5 = MOVE_TO_ID[BLACK_PREFIX + "c7c5"]

    tokens = [
        GAME_START_ID,
        THINK_START_ID,
        e2e4,
        c7c5,
        THINK_END_ID,
        e2e4,
    ]

    result = ChessEvaluator._extract_generated_think_tokens(tokens)
    assert result == [e2e4, c7c5]


def test_extract_generated_think_tokens_returns_empty_when_no_think():
    tokens = [GAME_START_ID, MOVE_TO_ID[WHITE_PREFIX + "e2e4"], GAME_END_ID]
    result = ChessEvaluator._extract_generated_think_tokens(tokens)
    assert result == []


def test_extract_generated_think_tokens_handles_multiple_think_blocks():
    e2e4 = MOVE_TO_ID[WHITE_PREFIX + "e2e4"]
    c7c5 = MOVE_TO_ID[BLACK_PREFIX + "c7c5"]

    tokens = [
        THINK_START_ID,
        e2e4,
        THINK_END_ID,
        THINK_START_ID,
        c7c5,
        THINK_END_ID,
    ]

    result = ChessEvaluator._extract_generated_think_tokens(tokens)
    assert result == [e2e4, c7c5]  # both think blocks combined


def test_compute_binary_f1_metrics_returns_expected_values():
    result = ChessEvaluator._compute_binary_f1_metrics(tp=3, fp=1, fn=2)

    assert result["qa_check_precision"] == 0.75
    assert result["qa_check_recall"] == 0.6
    assert result["qa_check_f1"] == 2 * 0.75 * 0.6 / (0.75 + 0.6)


def test_piece_probe_metrics_can_skip_piece_f1_breakdown():
    evaluator = ChessEvaluator(
        metrics=["piece_acc"],
        enable_piece_probe_metrics=True,
        enable_piece_f1_breakdown_metrics=False,
    )
    model = SimpleNamespace(config=SimpleNamespace(block_size=128))

    result = evaluator._evaluate_piece_probe([], model=model, device=None)

    assert result == {"piece_acc": 0.0, "qa_piece_f1": 0.0}
    assert not any(key.startswith("piece_f1_") for key in result)
