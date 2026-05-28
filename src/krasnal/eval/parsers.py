from dataclasses import dataclass

import torch

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.tokens import (
    DRAW_ID,
    ELO_TOKENS,
    GAME_END_ID,
    GAME_START_ID,
    OUTCOME_TOKENS,
    TC_TOKENS,
    get_move_clock_pairs,
    get_moves_only,
)


@dataclass
class GameTokens:
    outcome_token: int
    time_control_token: int | None
    white_elo_token: int | None
    black_elo_token: int | None
    move_tokens: list[int]
    move_active_seconds: list[int] | None = None
    move_opponent_seconds: list[int] | None = None
    prefix_active_seconds: int | None = None
    prefix_opponent_seconds: int | None = None

    @property
    def initial_context(self) -> list[int]:
        ctx = [GAME_START_ID]
        if self.time_control_token is not None:
            ctx.append(self.time_control_token)
        ctx.append(self.outcome_token)
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

    time_control_ids = set(TC_TOKENS.values())
    elo_bucket_ids = set(ELO_TOKENS.values())
    time_control_token = None
    white_elo_token = None
    black_elo_token = None

    for token_id in token_ids[1:outcome_idx]:
        if token_id in time_control_ids:
            time_control_token = token_id
            break

    for token_id in remaining_tokens:
        if token_id in elo_bucket_ids:
            if white_elo_token is None:
                white_elo_token = token_id
            elif black_elo_token is None:
                black_elo_token = token_id
                break

    return GameTokens(
        outcome_token=outcome_token,
        time_control_token=time_control_token,
        white_elo_token=white_elo_token,
        black_elo_token=black_elo_token,
        move_tokens=[],
    )


def parse_row_to_game_tokens(row: tuple[torch.Tensor, ...] | torch.Tensor) -> GameTokens | None:
    """Parse a dataset row with token_ids and aligned clock columns into GameTokens."""
    token_ids = row[0].tolist() if isinstance(row, tuple) else row.tolist()

    game_tokens = parse_game_tokens(token_ids)
    if game_tokens is None:
        return None

    moves = get_moves_only(token_ids)
    game_tokens.move_tokens = moves

    if not isinstance(row, tuple) or len(row) < 3:
        return None

    active_list = row[1].tolist()
    opponent_list = row[2].tolist()
    pairs = get_move_clock_pairs(token_ids, active_list, opponent_list)
    if pairs is None or len(pairs) != len(moves):
        return None
    if any(a >= CLOCK_IGNORE_ID or o >= CLOCK_IGNORE_ID for a, o in pairs):
        return None

    game_tokens.move_active_seconds = [p[0] for p in pairs]
    game_tokens.move_opponent_seconds = [p[1] for p in pairs]
    if active_list[0] < CLOCK_IGNORE_ID:
        game_tokens.prefix_active_seconds = int(active_list[0])
        game_tokens.prefix_opponent_seconds = int(opponent_list[0])
    return game_tokens
