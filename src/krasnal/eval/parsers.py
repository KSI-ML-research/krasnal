from dataclasses import dataclass

from krasnal.tokens import (
    DRAW_ID,
    ELO_TOKENS,
    GAME_END_ID,
    GAME_START_ID,
    OUTCOME_TOKENS,
)


@dataclass
class GameTokens:
    outcome_token: int
    white_elo_token: int | None
    black_elo_token: int | None
    move_tokens: list[int]

    @property
    def initial_context(self) -> list[int]:
        ctx = [GAME_START_ID, self.outcome_token]
        if self.white_elo_token is not None:
            ctx.append(self.white_elo_token)
        if self.black_elo_token is not None:
            ctx.append(self.black_elo_token)
        return ctx


def parse_game_tokens(token_ids: list[int]) -> GameTokens | None:
    if not token_ids or token_ids[0] != GAME_START_ID or token_ids[-1] != GAME_END_ID:
        return None

    outcome_token = None
    for token_id in token_ids[1:-1]:
        if token_id in OUTCOME_TOKENS.values():
            outcome_token = token_id
            break

    if outcome_token is None:
        outcome_token = DRAW_ID

    outcome_idx = token_ids.index(outcome_token)
    remaining_tokens = token_ids[outcome_idx + 1 : -1]

    elo_bucket_ids = set(ELO_TOKENS.values())
    white_elo_token = None
    black_elo_token = None

    for token_id in remaining_tokens:
        if token_id in elo_bucket_ids:
            if white_elo_token is None:
                white_elo_token = token_id
            elif black_elo_token is None:
                black_elo_token = token_id
                break

    return GameTokens(
        outcome_token=outcome_token,
        white_elo_token=white_elo_token,
        black_elo_token=black_elo_token,
        move_tokens=[],
    )
