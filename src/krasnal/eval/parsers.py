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
    get_moves_only,
)


@dataclass
class GameTokens:
    outcome_token: int
    time_control_token: int | None
    white_elo_token: int | None
    black_elo_token: int | None
    move_tokens: list[int]
    body_tokens: list[int]
    outcome_conditioning_enabled: bool = True
    body_active_seconds: list[int] | None = None
    body_opponent_seconds: list[int] | None = None
    prefix_active_seconds: int | None = None
    prefix_opponent_seconds: int | None = None

    @property
    def initial_context(self) -> list[int]:
        ctx = [GAME_START_ID]
        if self.time_control_token is not None:
            ctx.append(self.time_control_token)
        if self.outcome_conditioning_enabled:
            ctx.append(self.outcome_token)
        if self.white_elo_token is not None:
            ctx.append(self.white_elo_token)
        if self.black_elo_token is not None:
            ctx.append(self.black_elo_token)
        return ctx


def parse_game_tokens(token_ids: list[int]) -> GameTokens | None:
    if not token_ids or token_ids[0] != GAME_START_ID or token_ids[-1] != GAME_END_ID:
        return None

    time_control_ids = set(TC_TOKENS.values())
    elo_bucket_ids = set(ELO_TOKENS.values())
    time_control_token = None
    white_elo_token = None
    black_elo_token = None

    cursor = 1
    if cursor < len(token_ids) - 1 and token_ids[cursor] in time_control_ids:
        time_control_token = token_ids[cursor]
        cursor += 1

    outcome_token = None
    if cursor < len(token_ids) - 1 and token_ids[cursor] in OUTCOME_TOKENS.values():
        outcome_token = token_ids[cursor]
        cursor += 1

    outcome_conditioning_enabled = outcome_token is not None
    if outcome_token is None:
        outcome_token = DRAW_ID

    if cursor < len(token_ids) - 1 and token_ids[cursor] in elo_bucket_ids:
        white_elo_token = token_ids[cursor]
        cursor += 1
    if cursor < len(token_ids) - 1 and token_ids[cursor] in elo_bucket_ids:
        black_elo_token = token_ids[cursor]
        cursor += 1

    body_tokens = token_ids[cursor:-1]

    return GameTokens(
        outcome_token=outcome_token,
        time_control_token=time_control_token,
        white_elo_token=white_elo_token,
        black_elo_token=black_elo_token,
        move_tokens=[],
        body_tokens=body_tokens,
        outcome_conditioning_enabled=outcome_conditioning_enabled,
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
    if len(active_list) != len(token_ids) or len(opponent_list) != len(token_ids):
        return None

    prefix_len = len(game_tokens.initial_context)
    body_end = prefix_len + len(game_tokens.body_tokens)
    game_tokens.body_active_seconds = [int(x) for x in active_list[prefix_len:body_end]]
    game_tokens.body_opponent_seconds = [int(x) for x in opponent_list[prefix_len:body_end]]
    body_clock_pairs = zip(
        game_tokens.body_active_seconds,
        game_tokens.body_opponent_seconds,
        strict=True,
    )
    if any(a >= CLOCK_IGNORE_ID or o >= CLOCK_IGNORE_ID for a, o in body_clock_pairs):
        return None
    if active_list[0] < CLOCK_IGNORE_ID:
        game_tokens.prefix_active_seconds = int(active_list[0])
        game_tokens.prefix_opponent_seconds = int(opponent_list[0])
    return game_tokens
