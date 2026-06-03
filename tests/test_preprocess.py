from collections import Counter

import polars as pl
import pytest

from krasnal.preprocess import InvalidClockDataError, PreprocessConfig
from krasnal.preprocess.stats import token_mix_from_raw_sums, token_mix_raw_from_counts
from krasnal.preprocess.tokenize import (
    _build_game_tokens,
    _compute_check_qa_probs,
    _sample_bool_with_prefix,
    _tokenize_batch,
)
from krasnal.tokens import (
    ELO_1500_1599_ID,
    GAME_END_ID,
    GAME_START_ID,
    MOVE_TO_ID,
    OPP_MATERIAL_TOKENS,
    TC_BLITZ_INC_ID,
    TC_BLITZ_NO_INC_ID,
    TC_TOKENS,
    WHITE_WON_ID,
)


def _install_test_move(monkeypatch, move_key: str = "w:e2e4", token_id: int = 500) -> None:
    monkeypatch.setitem(MOVE_TO_ID, move_key, token_id)


def test_compute_check_qa_probs_balances_yes_no_average():
    p_yes, p_no = _compute_check_qa_probs(check_count=30, no_check_count=70, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.21428571428571427


def test_compute_check_qa_probs_handles_no_non_check_positions():
    p_yes, p_no = _compute_check_qa_probs(check_count=10, no_check_count=0, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.0


def test_sample_bool_with_prefix_deterministic():
    prefix = b"42|e2e4 e7e5|"
    kwargs = dict(prefix=prefix, ply=5, probability=0.5)
    assert _sample_bool_with_prefix(**kwargs) == _sample_bool_with_prefix(**kwargs)


def test_sample_bool_with_prefix_always_false_at_prob_zero():
    assert _sample_bool_with_prefix(b"x|", ply=0, probability=0.0) is False


def test_sample_bool_with_prefix_always_true_at_prob_one():
    assert _sample_bool_with_prefix(b"x|", ply=0, probability=1.0) is True


def test_sample_bool_with_prefix_different_prefixes_differ():
    a = _sample_bool_with_prefix(b"1|e2e4|", ply=0, probability=0.5)
    b = _sample_bool_with_prefix(b"2|e2e4|", ply=0, probability=0.5)
    assert a != b


def test_build_game_tokens_adds_time_control_after_game_start(monkeypatch):
    _install_test_move(monkeypatch)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        opponent_material_enabled=True,
        outcome_conditioning_enabled=True,
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
        opponent_material=[39],
    )

    assert tokens == [
        GAME_START_ID,
        TC_BLITZ_INC_ID,
        WHITE_WON_ID,
        ELO_1500_1599_ID,
        ELO_1500_1599_ID,
        500,
        OPP_MATERIAL_TOKENS["<opp_mat_39>"],
        GAME_END_ID,
    ]
    assert active_clocks[0] == 180
    assert opponent_clocks[0] == 180
    assert active_clocks[-2] == 170
    assert opponent_clocks[-2] == 180


def test_build_game_tokens_skips_time_control_when_disabled(monkeypatch):
    _install_test_move(monkeypatch)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=False,
        opponent_material_enabled=True,
        outcome_conditioning_enabled=True,
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
        opponent_material=[39],
    )

    assert tokens == [
        GAME_START_ID,
        WHITE_WON_ID,
        ELO_1500_1599_ID,
        ELO_1500_1599_ID,
        500,
        OPP_MATERIAL_TOKENS["<opp_mat_39>"],
        GAME_END_ID,
    ]
    assert active_clocks[0] == 180
    assert active_clocks[-2] == 170
    assert opponent_clocks[-2] == 180


def test_build_game_tokens_skips_elo_when_disabled(monkeypatch):
    _install_test_move(monkeypatch)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        include_elo=False,
        time_control_enabled=True,
        opponent_material_enabled=True,
        outcome_conditioning_enabled=True,
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
        opponent_material=[39],
    )

    assert tokens == [
        GAME_START_ID,
        TC_BLITZ_INC_ID,
        WHITE_WON_ID,
        500,
        OPP_MATERIAL_TOKENS["<opp_mat_39>"],
        GAME_END_ID,
    ]
    assert ELO_1500_1599_ID not in tokens
    assert len(tokens) == len(active_clocks) == len(opponent_clocks)


def test_build_game_tokens_skips_opponent_material_when_disabled(monkeypatch):
    _install_test_move(monkeypatch)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        opponent_material_enabled=False,
        time_control_enabled=True,
        outcome_conditioning_enabled=True,
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
    assert OPP_MATERIAL_TOKENS["<opp_mat_39>"] not in tokens
    assert len(tokens) == len(active_clocks) == len(opponent_clocks)


def test_build_game_tokens_raises_on_missing_clocks():
    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        opponent_material_enabled=False,
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


def test_build_game_tokens_raises_on_mismatched_clock_lengths():
    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        opponent_material_enabled=False,
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


def test_tokenize_batch_counts_invalid_clock_skips(monkeypatch):
    _install_test_move(monkeypatch)

    batch = pl.DataFrame(
        {
            "uci_moves": ["e2e4", "e2e4 e7e5"],
            "is_check": [[False], [False, False]],
            "piece_moved": [["p"], ["p", "p"]],
            "opponent_material": [[39], [39, 39]],
            "result": ["1-0", "0-1"],
            "white_rating": [1500, 1500],
            "black_rating": [1500, 1500],
            "clocks_white": [[170], [170]],
            "clocks_black": [[], []],
            "time_initial": [180, 180],
            "time_increment": [0, 0],
        }
    )

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        opponent_material_enabled=True,
        include_check_qa=False,
        check_qa_prob=0.0,
    )

    (token_rows, active_rows, opponent_rows), invalid_clock_skips = _tokenize_batch(
        batch,
        cfg,
        p_no=0.0,
    )

    assert invalid_clock_skips == 1
    assert len(token_rows) == 1
    assert len(active_rows) == 1
    assert len(opponent_rows) == 1


def test_token_mix_stats_counts_time_control_buckets():
    token_lists = [
        [
            GAME_START_ID,
            TC_BLITZ_NO_INC_ID,
            WHITE_WON_ID,
            500,
            OPP_MATERIAL_TOKENS["<opp_mat_39>"],
            GAME_END_ID,
        ],
        [
            GAME_START_ID,
            TC_BLITZ_INC_ID,
            WHITE_WON_ID,
            501,
            OPP_MATERIAL_TOKENS["<opp_mat_38>"],
            GAME_END_ID,
        ],
    ]
    id_counts: Counter[int] = Counter()
    for tokens in token_lists:
        id_counts.update(tokens)
    stats = token_mix_from_raw_sums(token_mix_raw_from_counts(dict(id_counts)))

    assert stats["tc_count"] == 2
    assert stats["tc_<tc_blitz_no_inc>_count"] == 1
    assert stats["tc_<tc_blitz_inc>_count"] == 1
    assert sum(stats[f"tc_{bucket}_count"] for bucket in TC_TOKENS) == 2
    assert stats["game_start_count"] == 2
    assert stats["game_end_count"] == 2
    assert stats["total_tokens"] == 12
    assert stats["uci_move_count"] == 2
    assert stats["material_count"] == 2
    assert stats["opp_mat_39_count"] == 1
    assert stats["opp_mat_38_count"] == 1
