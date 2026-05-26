import json

import polars as pl
import pytest

import krasnal.preprocess.tokenize as tokenize_module
from krasnal.preprocess import (
    InvalidClockDataError,
    PreprocessConfig,
    build_move_vocab_from_corpus,
)
from krasnal.preprocess.stats import compute_token_mix_stats
from krasnal.preprocess.tokenize import (
    _build_game_tokens,
    _compute_check_qa_probs,
)
from krasnal.tokens import (
    ELO_1500_1599_ID,
    GAME_END_ID,
    GAME_START_ID,
    TC_BLITZ_INC_ID,
    TC_BLITZ_NO_INC_ID,
    TC_TOKENS,
    WHITE_WON_ID,
)


def test_compute_check_qa_probs_balances_yes_no_average():
    p_yes, p_no = _compute_check_qa_probs(check_count=30, no_check_count=70, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.21428571428571427


def test_compute_check_qa_probs_handles_no_non_check_positions():
    p_yes, p_no = _compute_check_qa_probs(check_count=10, no_check_count=0, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.0


def test_build_move_vocab_from_corpus_writes_sorted_manifest_vocab(tmp_path):
    corpus_path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "uci_moves": ["g1f3 e7e5", "e2e4"],
            "piece_moved": [["n", "p"], ["p"]],
        }
    ).write_parquet(corpus_path)
    output_path = tmp_path / "move_vocab.json"

    build_move_vocab_from_corpus(
        [corpus_path],
        piece_aware_moves=True,
        side_prefixed_moves=True,
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text())
    move_vocab = {
        token: token_id for token, token_id in payload["vocab"].items() if not token.startswith("<")
    }

    assert payload["manifest"]["piece_aware_moves"] is True
    assert payload["manifest"]["side_prefixed_moves"] is True
    assert payload["manifest"]["vocab_size"] == len(payload["vocab"])
    assert list(move_vocab) == ["b:pawn:e7e5", "w:knight:g1f3", "w:pawn:e2e4"]


def test_build_move_vocab_from_corpus_fails_on_malformed_piece_moved(tmp_path):
    corpus_path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "uci_moves": ["e2e4 e7e5"],
            "piece_moved": [["p"]],
        }
    ).write_parquet(corpus_path)

    with pytest.raises(ValueError, match="piece_moved length"):
        build_move_vocab_from_corpus(
            [corpus_path],
            piece_aware_moves=False,
            side_prefixed_moves=True,
            output_path=tmp_path / "move_vocab.json",
        )


def test_build_game_tokens_adds_time_control_after_game_start(monkeypatch):
    monkeypatch.setattr(tokenize_module, "move_token_id_for_ply", lambda *_args: 500)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        include_check_qa=False,
        check_qa_prob=0.0,
    )
    tokens, active_clocks, opponent_clocks = _build_game_tokens(
        uci_moves="e2e4",
        is_check=[False],
        piece_moved=["p"],
        result="1-0",
        white_rating=1500,
        black_rating=1500,
        time_initial=180,
        time_increment=2,
        cfg=cfg,
        p_no=0.0,
        clocks_white=[170],
        clocks_black=[],
    )

    assert tokens == [
        GAME_START_ID,
        TC_BLITZ_INC_ID,
        WHITE_WON_ID,
        ELO_1500_1599_ID,
        ELO_1500_1599_ID,
        500,
        GAME_END_ID,
    ]
    assert active_clocks[0] == 180
    assert opponent_clocks[0] == 180
    assert active_clocks[-2] == 170
    assert opponent_clocks[-2] == 180


def test_build_game_tokens_skips_time_control_when_disabled(monkeypatch):
    monkeypatch.setattr(tokenize_module, "move_token_id_for_ply", lambda *_args: 500)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=False,
        include_check_qa=False,
        check_qa_prob=0.0,
    )
    tokens, active_clocks, opponent_clocks = _build_game_tokens(
        uci_moves="e2e4",
        is_check=[False],
        piece_moved=["p"],
        result="1-0",
        white_rating=1500,
        black_rating=1500,
        time_initial=180,
        time_increment=2,
        cfg=cfg,
        p_no=0.0,
        clocks_white=[170],
        clocks_black=[],
    )

    assert tokens == [
        GAME_START_ID,
        WHITE_WON_ID,
        ELO_1500_1599_ID,
        ELO_1500_1599_ID,
        500,
        GAME_END_ID,
    ]
    assert active_clocks[0] == 180
    assert active_clocks[-2] == 170
    assert opponent_clocks[-2] == 180


def test_build_game_tokens_raises_on_missing_clocks(monkeypatch):
    monkeypatch.setattr(tokenize_module, "move_token_id_for_ply", lambda *_args: 500)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        include_check_qa=False,
        check_qa_prob=0.0,
    )
    with pytest.raises(InvalidClockDataError, match="missing"):
        _build_game_tokens(
            uci_moves="e2e4",
            is_check=[False],
            piece_moved=["p"],
            result="1-0",
            white_rating=1500,
            black_rating=1500,
            time_initial=180,
            time_increment=0,
            cfg=cfg,
            p_no=0.0,
            clocks_white=None,
            clocks_black=None,
        )


def test_build_game_tokens_raises_on_mismatched_clock_lengths(monkeypatch):
    monkeypatch.setattr(tokenize_module, "move_token_id_for_ply", lambda *_args: 500)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        include_check_qa=False,
        check_qa_prob=0.0,
    )
    with pytest.raises(InvalidClockDataError, match="clock lengths"):
        _build_game_tokens(
            uci_moves="e2e4 e7e5 g1f3",
            is_check=[False, False, False],
            piece_moved=["p", "p", "n"],
            result="1-0",
            white_rating=1500,
            black_rating=1500,
            time_initial=180,
            time_increment=0,
            cfg=cfg,
            p_no=0.0,
            clocks_white=[170],
            clocks_black=[165],
        )


def test_token_mix_stats_counts_time_control_buckets():
    stats = compute_token_mix_stats(
        pl.DataFrame(
            {
                "token_ids": [
                    [GAME_START_ID, TC_BLITZ_NO_INC_ID, WHITE_WON_ID, 500, GAME_END_ID],
                    [GAME_START_ID, TC_BLITZ_INC_ID, WHITE_WON_ID, 501, GAME_END_ID],
                ]
            }
        ).lazy()
    )

    assert stats["tc_count"] == 2
    assert stats["tc_<tc_blitz_no_inc>_count"] == 1
    assert stats["tc_<tc_blitz_inc>_count"] == 1
    assert sum(stats[f"tc_{bucket}_count"] for bucket in TC_TOKENS) == 2
    assert stats["game_start_count"] == 2
    assert stats["game_end_count"] == 2
    assert stats["total_tokens"] == 10
    assert stats["uci_move_count"] == 2
