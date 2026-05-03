from krasnal.tokens import (
    BISHOP_ID,
    BLACK_WON_ID,
    DRAW_ID,
    ELO_1000_1499_ID,
    ELO_1500_1999_ID,
    ELO_2000_2499_ID,
    ELO_2500_2999_ID,
    ELO_ABOVE_3000_ID,
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
    THINK_END_ID,
    THINK_START_ID,
    WHITE_WON_ID,
    YES_CHECK_ID,
    get_elo_bucket,
    get_moves_only,
    move_key_for_ply,
    set_side_prefixed_moves,
)


def test_special_tokens_exist():
    assert GAME_START_ID in SPECIAL_TOKENS.values()
    assert GAME_END_ID in SPECIAL_TOKENS.values()
    assert PAD_ID in SPECIAL_TOKENS.values()
    assert WHITE_WON_ID in SPECIAL_TOKENS.values()
    assert BLACK_WON_ID in SPECIAL_TOKENS.values()
    assert DRAW_ID in SPECIAL_TOKENS.values()
    assert THINK_START_ID in SPECIAL_TOKENS.values()
    assert THINK_END_ID in SPECIAL_TOKENS.values()
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
    assert ELO_1000_1499_ID in ELO_TOKENS.values()
    assert ELO_1500_1999_ID in ELO_TOKENS.values()
    assert ELO_2000_2499_ID in ELO_TOKENS.values()
    assert ELO_2500_2999_ID in ELO_TOKENS.values()
    assert ELO_ABOVE_3000_ID in ELO_TOKENS.values()
    assert ELO_UNKNOWN_ID in ELO_TOKENS.values()


def test_elo_tokens_in_vocab():
    assert all(tok_str in MOVE_TO_ID for tok_str in ELO_TOKENS)


def test_elo_bucket_function():
    assert get_elo_bucket(999) == ELO_BELOW_1000_ID
    assert get_elo_bucket(1000) == ELO_1000_1499_ID
    assert get_elo_bucket(1499) == ELO_1000_1499_ID
    assert get_elo_bucket(1500) == ELO_1500_1999_ID
    assert get_elo_bucket(1999) == ELO_1500_1999_ID
    assert get_elo_bucket(2000) == ELO_2000_2499_ID
    assert get_elo_bucket(2499) == ELO_2000_2499_ID
    assert get_elo_bucket(2500) == ELO_2500_2999_ID
    assert get_elo_bucket(2999) == ELO_2500_2999_ID
    assert get_elo_bucket(3000) == ELO_ABOVE_3000_ID


def test_get_moves_only_basic():
    assert get_moves_only([GAME_START_ID, WHITE_WON_ID, 5000, 5001, 5002, GAME_END_ID]) == [
        5000,
        5001,
        5002,
    ]


def test_get_moves_only_with_elo():
    assert get_moves_only(
        [GAME_START_ID, WHITE_WON_ID, ELO_2000_2499_ID, ELO_1500_1999_ID, 5000, 5001, GAME_END_ID]
    ) == [5000, 5001]


def test_get_moves_only_strips_think_content():
    assert get_moves_only(
        [
            GAME_START_ID,
            WHITE_WON_ID,
            5000,
            THINK_START_ID,
            200,
            201,
            THINK_END_ID,
            5002,
            GAME_END_ID,
        ]
    ) == [5000, 5002]


def test_get_moves_only_all_special_tokens():
    assert get_moves_only(
        [
            GAME_START_ID,
            WHITE_WON_ID,
            ELO_BELOW_1000_ID,
            ELO_2500_2999_ID,
            THINK_START_ID,
            500,
            501,
            THINK_END_ID,
            5000,
            5001,
            GAME_END_ID,
        ]
    ) == [5000, 5001]


def test_side_prefixed_moves_toggle_changes_move_keys():
    set_side_prefixed_moves(True)
    assert move_key_for_ply("e2e4", 0) == "w:e2e4"
    assert move_key_for_ply("e7e5", 1) == "b:e7e5"
    try:
        set_side_prefixed_moves(False)
        assert move_key_for_ply("e2e4", 0) == "e2e4"
        assert move_key_for_ply("e7e5", 1) == "e7e5"
    finally:
        set_side_prefixed_moves(True)
