from types import SimpleNamespace

from krasnal.eval.evaluator import ChessEvaluator
from krasnal.eval.metrics import EvalContext
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

    assert result["qa/check/precision"] == 0.75
    assert result["qa/check/recall"] == 0.6
    assert result["qa/check/f1"] == 2 * 0.75 * 0.6 / (0.75 + 0.6)


def test_build_what_is_on_heatmap_uses_all_squares():
    square_f1s = {f"{file}{rank}": float(rank) for rank in range(1, 9) for file in "abcdefgh"}

    heatmap = ChessEvaluator._build_what_is_on_heatmap(square_f1s)

    assert heatmap is not None


def test_piece_probe_metrics_can_skip_piece_f1_per_piece():
    evaluator = ChessEvaluator(
        metrics=["piece_acc"],
        qa_config={
            "piece_type_moved": {
                "enabled": True,
                "f1_per_piece": False,
            }
        },
    )
    model = SimpleNamespace(config=SimpleNamespace(block_size=128))

    result = evaluator._evaluate_piece_probe([], model=model, device=None)

    assert result == {"qa/piece/acc": 0.0, "qa/piece/f1": 0.0}
    assert not any(key.startswith("qa/piece/f1_per_piece/") for key in result)


def test_evaluate_resets_stateful_metrics_between_runs(monkeypatch):
    evaluator = ChessEvaluator(metrics=["acc_opening"])
    evaluator.metrics["acc_opening"].buffer.append(1.0)

    dataset = [SimpleNamespace(tolist=lambda: [GAME_START_ID, GAME_END_ID])]
    model = SimpleNamespace(config=SimpleNamespace(block_size=128))

    def fake_parse_game_tokens(_token_ids):
        return SimpleNamespace(move_tokens=[], initial_context=[])

    def fake_get_moves_only(_token_ids):
        return [MOVE_TO_ID[WHITE_PREFIX + "e2e4"]]

    def fake_replay_games(_games, _block_size):
        return [EvalContext(sequence=[])]

    def fake_infer_and_aggregate(contexts, _model, _device):
        assert contexts
        assert evaluator.metrics["acc_opening"].buffer == []
        return {"game/acc_opening": 0.0}

    monkeypatch.setattr("krasnal.eval.evaluator.parse_game_tokens", fake_parse_game_tokens)
    monkeypatch.setattr("krasnal.eval.evaluator.get_moves_only", fake_get_moves_only)
    monkeypatch.setattr("krasnal.eval.evaluator.replay_games", fake_replay_games)
    monkeypatch.setattr(evaluator, "_infer_and_aggregate", fake_infer_and_aggregate)

    result = evaluator.evaluate(model=model, dataset=dataset, num_games=1, device=None)

    assert result == {"game/acc_opening": 0.0}
