import json

import bulletchess
import polars as pl
import pytest

from krasnal.preprocess import (
    InvalidClockDataError,
    PreprocessConfig,
    build_move_vocab_from_corpus,
)
from krasnal.preprocess.stats import compute_token_mix_stats
from krasnal.preprocess.tokenize import (
    _build_game_tokens,
    _compute_check_qa_probs,
    _sample_bool_with_prefix,
    _tokenize_batch,
)
from krasnal.sampling import sample_bool
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
    opponent_material_token_id,
)


def _install_test_move(monkeypatch, move_key: str = "w:e2e4", token_id: int = 500) -> None:
    monkeypatch.setitem(MOVE_TO_ID, move_key, token_id)


def _install_test_moves(monkeypatch, moves: list[str], start_token_id: int = 500) -> None:
    for ply, move in enumerate(moves):
        prefix = "w:" if ply % 2 == 0 else "b:"
        monkeypatch.setitem(MOVE_TO_ID, f"{prefix}{move}", start_token_id + ply)


def test_compute_check_qa_probs_balances_yes_no_average():
    p_yes, p_no = _compute_check_qa_probs(check_count=30, no_check_count=70, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.21428571428571427


def test_compute_check_qa_probs_handles_no_non_check_positions():
    p_yes, p_no = _compute_check_qa_probs(check_count=10, no_check_count=0, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.0


def test_sample_bool_with_prefix_matches_shared_sampler():
    game_key = "e2e4 e7e5 g1f3"
    seed = 123
    prefix = f"{seed}|{game_key}|".encode()

    for probability in [0.0, 0.1, 0.5, 1.0]:
        assert [_sample_bool_with_prefix(prefix, ply, probability) for ply in range(20)] == [
            sample_bool(seed=seed, game_key=game_key, ply=ply, probability=probability)
            for ply in range(20)
        ]


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
    _install_test_move(monkeypatch)

    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
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


def test_build_game_tokens_incremental_material_matches_board_scan(monkeypatch):
    cfg = PreprocessConfig(
        seed=1,
        block_size=1024,
        time_control_enabled=True,
        outcome_conditioning_enabled=True,
        include_check_qa=False,
        check_qa_prob=0.0,
    )
    sequences = [
        ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"],
        ["h2h4", "a7a5", "h4h5", "a5a4", "h5h6", "a4a3", "h6g7", "a3b2", "g7h8q"],
    ]

    for moves in sequences:
        _install_test_moves(monkeypatch, moves)
        tokens, _active_clocks, _opponent_clocks = _build_game_tokens(
            uci_moves=" ".join(moves),
            is_check=[False] * len(moves),
            piece_moved=["p"] * len(moves),
            result="1-0",
            white_rating=1500,
            black_rating=1500,
            time_initial=180,
            time_increment=2,
            cfg=cfg,
            p_no=0.0,
            clocks_white=[170] * ((len(moves) + 1) // 2),
            clocks_black=[165] * (len(moves) // 2),
        )

        board = bulletchess.Board()
        expected_material_tokens = []
        for move in moves:
            board.apply(bulletchess.Move.from_uci(move))
            expected_material_tokens.append(opponent_material_token_id(board))

        prefix_len = 5
        actual_material_tokens = tokens[prefix_len + 1 : -1 : 2]
        assert actual_material_tokens == expected_material_tokens


def test_build_game_tokens_raises_on_missing_clocks():
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


def test_build_game_tokens_raises_on_mismatched_clock_lengths():
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


def test_tokenize_batch_counts_invalid_clock_skips(monkeypatch):
    _install_test_move(monkeypatch)

    batch = pl.DataFrame(
        {
            "uci_moves": ["e2e4", "e2e4 e7e5"],
            "is_check": [[False], [False, False]],
            "piece_moved": [["p"], ["p", "p"]],
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
    stats = compute_token_mix_stats(
        pl.DataFrame(
            {
                "token_ids": [
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
            }
        ).lazy()
    )

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
