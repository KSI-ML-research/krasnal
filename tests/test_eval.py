import pytest

from krasnal.eval.metrics.acpl import ACPLMetric
from krasnal.eval.metrics.blunder_rate import BlunderRateMetric
from krasnal.eval.metrics.context import EvalContext
from krasnal.eval.metrics.stockfish_top1 import StockfishTop1AgreementMetric
from krasnal.eval.parsers import GameTokens, parse_game_tokens
from krasnal.eval.replayer import replay_game_tokens
from krasnal.eval.stockfish import StockfishAnalysis, StockfishClient
from krasnal.tokens import (
    BLACK_PREFIX,
    DRAW_ID,
    ELO_1500_1599_ID,
    ELO_2000_2099_ID,
    MOVE_TO_ID,
    TC_RAPID_INC_ID,
    WHITE_PREFIX,
)


def test_acpl_normalizes_post_move_eval_to_mover_perspective():
    class DummyStockfish:
        def __init__(self, scores):
            self.scores = scores

        def get_eval(self, fen: str) -> float | None:
            return self.scores.get(fen)

    fen_before = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    fen_after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    stockfish = DummyStockfish(
        {
            fen_before: 50.0,
            fen_after: -20.0,
        }
    )

    metric = ACPLMetric(stockfish=stockfish, sample_size=2)
    for _ in range(2):
        metric.compute(EvalContext(fen=fen_before, top1_fen=fen_after))

    result = metric.finalize()

    assert result["acpl"] == 30.0


def test_stockfish_client_parses_go_output_bestmove_and_cp():
    client = StockfishClient(depth=10)

    analysis = client._parse_go_output(
        "\n".join(
            [
                "info depth 10 score cp 42 pv e2e4 e7e5",
                "bestmove e2e4 ponder e7e5",
            ]
        )
    )

    assert analysis == StockfishAnalysis(bestmove="e2e4", score_cp=42.0)


def test_stockfish_client_parses_terminal_position_without_bestmove():
    client = StockfishClient(depth=10)

    analysis = client._parse_go_output(
        "\n".join(
            [
                "info depth 0 score mate 0",
                "bestmove (none)",
            ]
        )
    )

    assert analysis == StockfishAnalysis(bestmove=None, score_cp=-1000.0)


def test_stockfish_top1_metric_compares_to_stockfish_bestmove():
    class DummyStockfish:
        def get_best_move(self, fen: str) -> str | None:
            assert fen == "fen-1"
            return "e2e4"

    metric = StockfishTop1AgreementMetric(stockfish=DummyStockfish(), sample_size=1)
    metric.compute(EvalContext(fen="fen-1", top1_move_uci="e2e4"))

    assert metric.finalize()["stockfish_top1"] == 1.0


def test_stockfish_top1_metric_skips_terminal_positions():
    class DummyStockfish:
        def get_best_move(self, _fen: str) -> str | None:
            return None

    metric = StockfishTop1AgreementMetric(stockfish=DummyStockfish(), sample_size=1)
    metric.compute(EvalContext(fen="fen-1", top1_move_uci="e2e4"))

    assert metric.finalize()["stockfish_top1"] == 0.0


def test_blunder_rate_metric_counts_large_eval_drop():
    fen_before = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    fen_after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    class DummyStockfish:
        def get_eval(self, fen: str) -> float:
            return {
                fen_before: 120.0,
                fen_after: 30.0,
            }[fen]

    metric = BlunderRateMetric(stockfish=DummyStockfish(), sample_size=1, threshold_cp=100.0)
    metric.compute(
        EvalContext(
            fen=fen_before,
            top1_fen=fen_after,
        )
    )

    assert metric.finalize()["blunder_rate"] == 1.0


def test_replay_game_tokens_sets_player_elo_by_side_to_move():
    game_tokens = GameTokens(
        outcome_token=DRAW_ID,
        time_control_token=None,
        white_elo_token=ELO_1500_1599_ID,
        black_elo_token=ELO_2000_2099_ID,
        move_tokens=[
            MOVE_TO_ID[WHITE_PREFIX + "e2e4"],
            MOVE_TO_ID[BLACK_PREFIX + "e7e5"],
        ],
    )

    contexts = replay_game_tokens(game_tokens)

    assert [ctx.player_elo_token for ctx in contexts] == [
        ELO_1500_1599_ID,
        ELO_2000_2099_ID,
    ]


def test_parse_game_tokens_preserves_time_control_in_initial_context():
    parsed = parse_game_tokens(
        [
            0,
            TC_RAPID_INC_ID,
            DRAW_ID,
            ELO_1500_1599_ID,
            ELO_2000_2099_ID,
            500,
            1,
        ]
    )

    assert parsed is not None
    assert parsed.time_control_token == TC_RAPID_INC_ID
    assert parsed.initial_context == [
        0,
        TC_RAPID_INC_ID,
        DRAW_ID,
        ELO_1500_1599_ID,
        ELO_2000_2099_ID,
    ]


def test_acpl_propagates_stockfish_errors():
    class FailingStockfish:
        def get_eval(self, _fen: str) -> float:
            raise RuntimeError("stockfish crashed")

    metric = ACPLMetric(stockfish=FailingStockfish(), sample_size=2)
    metric.compute(EvalContext(fen="fen-1", top1_fen="fen-2"))
    metric.compute(EvalContext(fen="fen-3", top1_fen="fen-4"))

    with pytest.raises(RuntimeError, match="stockfish crashed"):
        metric.finalize()
