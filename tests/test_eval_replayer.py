from krasnal.eval.parsers import GameTokens
from krasnal.eval.replayer import replay_game_tokens
from krasnal.tokens import ELO_UNKNOWN_ID, MOVE_TO_ID, WHITE_PREFIX, WHITE_WON_ID


def _move_token(uci: str) -> int:
    for key in (f"{WHITE_PREFIX}{uci}", uci):
        token_id = MOVE_TO_ID.get(key)
        if token_id is not None:
            return token_id
    raise KeyError(uci)


def test_replay_game_tokens_skips_illegal_move_without_crashing():
    e2e4 = _move_token("e2e4")

    game_tokens = GameTokens(
        outcome_token=WHITE_WON_ID,
        white_elo_token=ELO_UNKNOWN_ID,
        black_elo_token=ELO_UNKNOWN_ID,
        move_tokens=[e2e4, e2e4],
    )

    contexts = replay_game_tokens(game_tokens)

    assert len(contexts) == 1
    assert contexts[0].actual_token == e2e4
    assert contexts[0].sequence == game_tokens.initial_context
