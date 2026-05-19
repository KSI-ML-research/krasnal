from krasnal.eval.parsers import GameTokens, parse_game_tokens
from krasnal.eval.replayer import replay_game_tokens
from krasnal.tokens import (
    BLACK_PREFIX,
    DRAW_ID,
    ELO_1500_1599_ID,
    ELO_2000_2099_ID,
    MOVE_TO_ID,
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
