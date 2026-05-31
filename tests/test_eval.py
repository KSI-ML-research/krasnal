from krasnal.eval.parsers import GameTokens, parse_game_tokens
from krasnal.eval.replayer import replay_game_tokens
from krasnal.tokens import (
    BLACK_PREFIX,
    DRAW_ID,
    ELO_1500_1599_ID,
    ELO_2000_2099_ID,
    MOVE_TO_ID,
    OPP_MATERIAL_TOKENS,
    TC_RAPID_INC_ID,
    WHITE_PREFIX,
)


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
        body_tokens=[
            MOVE_TO_ID[WHITE_PREFIX + "e2e4"],
            OPP_MATERIAL_TOKENS["<opp_mat_39>"],
            MOVE_TO_ID[BLACK_PREFIX + "e7e5"],
            OPP_MATERIAL_TOKENS["<opp_mat_39>"],
        ],
    )

    contexts = replay_game_tokens(game_tokens)

    assert [ctx.player_elo_token for ctx in contexts] == [
        ELO_1500_1599_ID,
        ELO_2000_2099_ID,
    ]
    assert contexts[1].sequence[-1] == OPP_MATERIAL_TOKENS["<opp_mat_39>"]


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
    assert parsed.body_tokens == [500]
    assert parsed.initial_context == [
        0,
        TC_RAPID_INC_ID,
        DRAW_ID,
        ELO_1500_1599_ID,
        ELO_2000_2099_ID,
    ]


def test_parse_game_tokens_accepts_missing_outcome_conditioning():
    parsed = parse_game_tokens(
        [
            0,
            TC_RAPID_INC_ID,
            ELO_1500_1599_ID,
            ELO_2000_2099_ID,
            500,
            1,
        ]
    )

    assert parsed is not None
    assert parsed.outcome_conditioning_enabled is False
    assert parsed.body_tokens == [500]
    assert parsed.initial_context == [
        0,
        TC_RAPID_INC_ID,
        ELO_1500_1599_ID,
        ELO_2000_2099_ID,
    ]
