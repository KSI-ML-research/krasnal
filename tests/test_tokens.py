import json

import pytest

from krasnal.tokens import (
    BISHOP_ID,
    BLACK_WON_ID,
    DRAW_ID,
    ELO_1000_1099_ID,
    ELO_1500_1599_ID,
    ELO_2000_2099_ID,
    ELO_2100_2199_ID,
    ELO_ABOVE_2200_ID,
    ELO_BELOW_1000_ID,
    ELO_TOKENS,
    ELO_UNKNOWN_ID,
    GAME_END_ID,
    GAME_START_ID,
    IS_CHECK_ID,
    KING_ID,
    KNIGHT_ID,
    MOVE_TO_ID,
    NO_CHECK_ID,
    PAD_ID,
    PAWN_ID,
    PIECE_TYPE_MOVED_ID,
    QUEEN_ID,
    ROOK_ID,
    SPECIAL_TOKENS,
    TC_BLITZ_INC_ID,
    TC_BLITZ_NO_INC_ID,
    TC_CLASSICAL_ID,
    TC_RAPID_INC_ID,
    TC_RAPID_NO_INC_ID,
    TC_TOKENS,
    TC_UNKNOWN_ID,
    WHITE_WON_ID,
    YES_CHECK_ID,
    build_move_key,
    get_elo_bucket,
    get_moves_only,
    get_time_control_bucket,
    load_move_vocab,
    make_move_vocab_artifact,
    move_key_for_ply,
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
    assert PIECE_TYPE_MOVED_ID in SPECIAL_TOKENS.values()
    assert PAWN_ID in SPECIAL_TOKENS.values()
    assert KNIGHT_ID in SPECIAL_TOKENS.values()
    assert BISHOP_ID in SPECIAL_TOKENS.values()
    assert ROOK_ID in SPECIAL_TOKENS.values()
    assert QUEEN_ID in SPECIAL_TOKENS.values()
    assert KING_ID in SPECIAL_TOKENS.values()


def test_special_tokens_in_vocab():
    assert all(tok_str in MOVE_TO_ID for tok_str in SPECIAL_TOKENS)


def test_elo_tokens_exist():
    assert ELO_BELOW_1000_ID in ELO_TOKENS.values()
    assert ELO_1000_1099_ID in ELO_TOKENS.values()
    assert ELO_1500_1599_ID in ELO_TOKENS.values()
    assert ELO_2000_2099_ID in ELO_TOKENS.values()
    assert ELO_ABOVE_2200_ID in ELO_TOKENS.values()
    assert ELO_UNKNOWN_ID in ELO_TOKENS.values()


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
