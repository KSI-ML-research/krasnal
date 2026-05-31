import json

import bulletchess
import pytest

from krasnal.sampling import whats_on_square_index
from krasnal.tokens import (
    BLACK_WON_ID,
    COLORED_PIECE_TOKENS,
    DRAW_ID,
    ELO_1000_1099_ID,
    ELO_1500_1599_ID,
    ELO_2000_2099_ID,
    ELO_2100_2199_ID,
    ELO_ABOVE_2200_ID,
    ELO_BELOW_1000_ID,
    ELO_TOKENS,
    EMPTY_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    MAX_SIDE_MATERIAL,
    MOVE_TO_ID,
    NO_CHECK_ID,
    OPP_MATERIAL_TOKENS,
    PAD_ID,
    PIECE_MATERIAL_VALUES,
    SPECIAL_TOKENS,
    TC_BLITZ_INC_ID,
    TC_BLITZ_NO_INC_ID,
    TC_CLASSICAL_ID,
    TC_RAPID_INC_ID,
    TC_RAPID_NO_INC_ID,
    TC_TOKENS,
    TC_UNKNOWN_ID,
    WHATS_ON_SQUARE,
    WHITE_WON_ID,
    YES_CHECK_ID,
    build_move_key,
    get_elo_bucket,
    get_move_clock_pairs,
    get_moves_only,
    get_time_control_bucket,
    load_move_vocab,
    make_move_vocab_artifact,
    move_key_for_ply,
    normalize_history_uci_moves,
    opponent_material_points,
    square_index_to_str,
    whats_on_answer_token_id,
    whats_on_probe_labels,
    whats_on_prompt_token_id,
)


def test_special_tokens_exist():
    assert GAME_START_ID in SPECIAL_TOKENS.values()
    assert GAME_END_ID in SPECIAL_TOKENS.values()
    assert PAD_ID in SPECIAL_TOKENS.values()
    assert WHITE_WON_ID in SPECIAL_TOKENS.values()
    assert BLACK_WON_ID in SPECIAL_TOKENS.values()
    assert DRAW_ID in SPECIAL_TOKENS.values()

    assert IS_CHECK_ID in SPECIAL_TOKENS.values()
    assert YES_CHECK_ID in SPECIAL_TOKENS.values()
    assert NO_CHECK_ID in SPECIAL_TOKENS.values()


def test_special_tokens_in_vocab():
    assert all(tok_str in MOVE_TO_ID for tok_str in SPECIAL_TOKENS)


def test_elo_tokens_exist():
    assert ELO_BELOW_1000_ID in ELO_TOKENS.values()
    assert ELO_1000_1099_ID in ELO_TOKENS.values()
    assert ELO_1500_1599_ID in ELO_TOKENS.values()
    assert ELO_2000_2099_ID in ELO_TOKENS.values()
    assert ELO_ABOVE_2200_ID in ELO_TOKENS.values()


def test_elo_tokens_in_vocab():
    assert all(tok_str in MOVE_TO_ID for tok_str in ELO_TOKENS)


def test_time_control_tokens_exist():
    assert TC_BLITZ_NO_INC_ID in TC_TOKENS.values()
    assert TC_BLITZ_INC_ID in TC_TOKENS.values()
    assert TC_RAPID_NO_INC_ID in TC_TOKENS.values()
    assert TC_RAPID_INC_ID in TC_TOKENS.values()
    assert TC_CLASSICAL_ID in TC_TOKENS.values()
    assert TC_UNKNOWN_ID in TC_TOKENS.values()


def test_time_control_tokens_in_vocab():
    assert all(tok_str in MOVE_TO_ID for tok_str in TC_TOKENS)


def test_opponent_material_tokens_cover_computed_maximum():
    assert PIECE_MATERIAL_VALUES["queen"] == 9
    assert MAX_SIDE_MATERIAL == 103
    assert len(OPP_MATERIAL_TOKENS) == 104
    assert OPP_MATERIAL_TOKENS["<opp_mat_0>"] in SPECIAL_TOKENS.values()
    assert OPP_MATERIAL_TOKENS["<opp_mat_103>"] in SPECIAL_TOKENS.values()


def test_opponent_material_points_for_starting_side_to_move():
    assert opponent_material_points(bulletchess.Board()) == 39


def test_elo_bucket_function():
    assert get_elo_bucket(999) == ELO_BELOW_1000_ID
    assert get_elo_bucket(1000) == ELO_1000_1099_ID
    assert get_elo_bucket(1099) == ELO_1000_1099_ID
    assert get_elo_bucket(1500) == ELO_1500_1599_ID
    assert get_elo_bucket(1599) == ELO_1500_1599_ID
    assert get_elo_bucket(2000) == ELO_2000_2099_ID
    assert get_elo_bucket(2099) == ELO_2000_2099_ID
    assert get_elo_bucket(2199) == ELO_2100_2199_ID
    assert get_elo_bucket(2200) == ELO_ABOVE_2200_ID
    assert get_elo_bucket(2500) == ELO_ABOVE_2200_ID
    assert get_elo_bucket(3000) == ELO_ABOVE_2200_ID


def test_time_control_bucket_function():
    assert get_time_control_bucket(None, 0) == TC_UNKNOWN_ID
    assert get_time_control_bucket(300, 0) == TC_BLITZ_NO_INC_ID
    assert get_time_control_bucket(180, 3) == TC_BLITZ_INC_ID
    assert get_time_control_bucket(600, 0) == TC_RAPID_NO_INC_ID
    assert get_time_control_bucket(600, 5) == TC_RAPID_INC_ID
    assert get_time_control_bucket(900, 15) == TC_CLASSICAL_ID


def test_get_moves_only_basic():
    assert get_moves_only([GAME_START_ID, WHITE_WON_ID, 5000, 5001, 5002, GAME_END_ID]) == [
        5000,
        5001,
        5002,
    ]


def test_get_moves_only_with_elo():
    assert get_moves_only(
        [GAME_START_ID, WHITE_WON_ID, ELO_2000_2099_ID, ELO_1500_1599_ID, 5000, 5001, GAME_END_ID]
    ) == [5000, 5001]


def test_get_moves_only_all_special_tokens():
    assert get_moves_only(
        [
            GAME_START_ID,
            WHITE_WON_ID,
            ELO_BELOW_1000_ID,
            ELO_ABOVE_2200_ID,
            5000,
            5001,
            GAME_END_ID,
        ]
    ) == [5000, 5001]


def test_side_prefixed_moves_toggle_changes_move_keys():
    assert move_key_for_ply("e2e4", 0, side_prefixed_moves=True) == "w:e2e4"
    assert move_key_for_ply("e7e5", 1, side_prefixed_moves=True) == "b:e7e5"
    assert move_key_for_ply("e2e4", 0, side_prefixed_moves=False) == "e2e4"
    assert move_key_for_ply("e7e5", 1, side_prefixed_moves=False) == "e7e5"


def test_move_vocab_generation_is_deterministic_and_sorted():
    move_keys = ["w:e2e4", "b:e7e5", "w:a2a3", "b:e7e5"]

    first = make_move_vocab_artifact(
        move_keys,
        piece_aware_moves=False,
        side_prefixed_moves=True,
        generation_timestamp="test",
    )
    second = make_move_vocab_artifact(
        reversed(move_keys),
        piece_aware_moves=False,
        side_prefixed_moves=True,
        generation_timestamp="test",
    )

    assert first["vocab"] == second["vocab"]
    move_vocab = {
        token: token_id for token, token_id in first["vocab"].items() if token not in SPECIAL_TOKENS
    }
    assert list(move_vocab) == ["b:e7e5", "w:a2a3", "w:e2e4"]
    assert list(move_vocab.values()) == list(
        range(max(SPECIAL_TOKENS.values()) + 1, max(SPECIAL_TOKENS.values()) + 4)
    )


def test_piece_aware_moves_and_uci_move_keys_keep_promotion_suffix():
    assert (
        build_move_key(
            "e7e8q",
            "pawn",
            ply=0,
            piece_aware_moves=False,
            side_prefixed_moves=True,
        )
        == "w:e7e8q"
    )
    assert (
        build_move_key(
            "g1f3",
            "n",
            ply=1,
            piece_aware_moves=True,
            side_prefixed_moves=True,
        )
        == "b:knight:g1f3"
    )
    assert (
        build_move_key(
            "e7e8q",
            "pawn",
            ply=0,
            piece_aware_moves=True,
            side_prefixed_moves=False,
        )
        == "pawn:e7e8q"
    )


def test_side_prefix_combinations():
    assert (
        build_move_key(
            "e2e4",
            "pawn",
            ply=0,
            piece_aware_moves=False,
            side_prefixed_moves=True,
        )
        == "w:e2e4"
    )
    assert (
        build_move_key(
            "e2e4",
            "pawn",
            ply=0,
            piece_aware_moves=False,
            side_prefixed_moves=False,
        )
        == "e2e4"
    )
    assert (
        build_move_key(
            "e2e4",
            "pawn",
            ply=0,
            piece_aware_moves=True,
            side_prefixed_moves=True,
        )
        == "w:pawn:e2e4"
    )
    assert (
        build_move_key(
            "e2e4",
            "pawn",
            ply=0,
            piece_aware_moves=True,
            side_prefixed_moves=False,
        )
        == "pawn:e2e4"
    )


def test_get_move_clock_pairs_aligns_with_get_moves_only():
    from krasnal.config import CLOCK_IGNORE_ID

    m1, m2 = 9000, 9001
    token_ids = [GAME_START_ID, DRAW_ID, m1, IS_CHECK_ID, YES_CHECK_ID, m2, GAME_END_ID]
    active = [CLOCK_IGNORE_ID] * len(token_ids)
    opp = [CLOCK_IGNORE_ID] * len(token_ids)
    active[2] = 50
    opp[2] = 300
    active[5] = 12
    opp[5] = 400

    assert get_moves_only(token_ids) == [m1, m2]
    pairs = get_move_clock_pairs(token_ids, active, opp)
    assert pairs == [(50, 300), (12, 400)]


def test_normalize_history_uci_move():
    from krasnal.tokens import normalize_history_uci_move

    assert normalize_history_uci_move("e2e4") == "e2e4"
    assert normalize_history_uci_move("w:pawn:e2e4") == "e2e4"
    assert normalize_history_uci_move("pawn:e2e4") == "e2e4"


def test_normalize_history_uci_moves_filters_blank_entries():
    assert normalize_history_uci_moves(" e2e4   w:pawn:e7e5  ") == ["e2e4", "e7e5"]


def test_move_vocab_manifest_mismatch_fails(tmp_path):
    path = tmp_path / "move_vocab.json"
    artifact = make_move_vocab_artifact(
        ["w:e2e4"],
        piece_aware_moves=False,
        side_prefixed_moves=True,
        generation_timestamp="test",
    )
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="manifest does not match runtime config"):
        load_move_vocab(path, piece_aware_moves=True, side_prefixed_moves=True)


_WHATS_ON_PROBE_KWARGS = dict(
    game_key="e2e4 e7e5",
    ply=1,
    seed=123,
)
_WHATS_ON_PROBE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


def test_square_index_to_str_corners():
    assert square_index_to_str(0) == "a1"
    assert square_index_to_str(7) == "h1"
    assert square_index_to_str(56) == "a8"
    assert square_index_to_str(63) == "h8"


def test_whats_on_prompt_token_id_matches_whats_on_square_map():
    assert whats_on_prompt_token_id("e4") == WHATS_ON_SQUARE["<whats_on_e4>"]


def test_whats_on_answer_token_id_empty_square():
    board = bulletchess.Board()
    assert whats_on_answer_token_id(board, "e4") == EMPTY_ID


def test_whats_on_answer_token_id_occupied_square():
    board = bulletchess.Board()
    board.apply(bulletchess.Move.from_uci("e2e4"))
    assert whats_on_answer_token_id(board, "e4") == COLORED_PIECE_TOKENS["<w:pawn>"]


def test_whats_on_probe_labels_matches_decomposed_helpers():
    board = bulletchess.Board.from_fen(_WHATS_ON_PROBE_FEN)
    sq_str, prompt_id, ans_id = whats_on_probe_labels(board, **_WHATS_ON_PROBE_KWARGS)

    sq_idx = whats_on_square_index(**_WHATS_ON_PROBE_KWARGS)
    assert sq_str == square_index_to_str(sq_idx)
    assert prompt_id == whats_on_prompt_token_id(sq_str)
    assert ans_id == whats_on_answer_token_id(board, sq_str)


def test_whats_on_probe_labels_deterministic():
    board = bulletchess.Board.from_fen(_WHATS_ON_PROBE_FEN)
    first = whats_on_probe_labels(board, **_WHATS_ON_PROBE_KWARGS)
    second = whats_on_probe_labels(board, **_WHATS_ON_PROBE_KWARGS)
    assert first == second
